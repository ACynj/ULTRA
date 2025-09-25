import os
import sys
import math
import pprint
from itertools import islice

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
# 新的修改
from ultra.text_semantics import (
    load_or_generate_relation_semantics, 
    build_relation_embeddings, 
    make_inverse_by_negation
)
from ultra.tasks import build_text_relation_graph


separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    #model = pyg.compile(model, dynamic=True)
    model = model.to(device)
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    
    # SEMMA: 在线生成关系文本语义并构建文本关系图
    # 检查是否text配置不为空且text配置中开启了generate开关
    if hasattr(cfg, 'text') and cfg.text is not None and getattr(cfg.text, 'generate', False):
        print("\n" + separator)
        print("SEMMA: 开始生成关系文本语义...")
        print(separator)
        
        # 获取配置参数
        # 从配置中读取文本生成相关的参数，若配置中未定义则使用默认值。
        api_key = getattr(cfg.text, 'api_key', 'sk-dvJb2Dz7h6ikAnEAXP7morrMIIxTcpQN6EPKTUza7AqmSkNc')
        llm_model = getattr(cfg.text, 'llm_model', 'gpt-4o-2024-11-20')
        cache_dir = getattr(cfg.text, 'cache_dir', './cache')
        combine = getattr(cfg.text, 'combine', 'COMBINED_SUM')
        threshold = getattr(cfg.text, 'threshold', 0.8)
        top_percent = getattr(cfg.text, 'top_percent', None)
        
        # 为每个正向关系收集一个示例三元组
        # 获取正向关系的数量
        num_rel_direct = train_data.num_relations // 2
        # 尝试从训练数据中获取关系的文本的名称列表
        tokens = getattr(train_data, 'relation_tokens', None)
        # 如果有预定义的tokens（关系文本名称），就用tokens[i]作为第 i 个正向关系的名称；如果没有，就用r0、r1、r2... 这样的默认名称。
        rel_names = [tokens[i] if tokens is not None and i < len(tokens) else f"r{i}" for i in range(num_rel_direct)]
        
        rel_to_example = {} # 初始化一个空字典，用于存储每个关系类型的示例三元组
        etypes = train_data.target_edge_type # 从训练数据中提取目标边的类型
        eidx = train_data.target_edge_index # 从训练数据中提取目标边的索引 形状为 [2, num_edges]，其中第一行存储头实体的索引，第二行存储尾实体的索引。
        seen = set() # 初始化一个空集合，用于记录已经处理过的正向关系类型
        
        print(f"收集 {num_rel_direct} 个关系的示例三元组...")
        # 遍历边
        for col in range(etypes.shape[0]):
            r = int(etypes[col].item()) # 获取当前边的关系类型
            if r >= num_rel_direct: # 是否是正向关系
                continue
            if r in seen: # 是否处理过
                continue
            
            h = int(eidx[0, col].item()) # 头实体索引
            t = int(eidx[1, col].item()) # 尾实体索引
            # 后续替换为真实实体文字
            head_txt = "head_entity"
            tail_txt = "tail_entity"
            rel_to_example[rel_names[r]] = (head_txt, rel_names[r], tail_txt)
            seen.add(r)
            
            if len(seen) == num_rel_direct:
                break
        
        print(f"成功收集 {len(rel_to_example)} 个关系的示例")
        
        # 生成或加载关系语义
        # 缓存文件
        cache_file = os.path.join(cache_dir, f"relation_semantics_{cfg.dataset['class'].lower()}.pkl")
        cleaned, descs = load_or_generate_relation_semantics(
            rel_to_example=rel_to_example, 
            cache_file=cache_file, # 若缓存文件存在则直接加载缓存的语义
            api_key=api_key,
            model=llm_model,
            force_regenerate=getattr(cfg.text, 'force_regenerate', False)
        )
        
        # 生成关系文本嵌入
        print("使用 jina-embeddings-v3 生成文本嵌入...")
        rel_text_init = build_relation_embeddings(
            rel_names=rel_names,
            cleaned=cleaned,
            descs=descs,
            combine=combine,
        )
        
        # 处理逆关系嵌入
        if rel_text_init.shape[0] * 2 == train_data.num_relations:
            print("生成逆关系嵌入...")
            rel_text_full = make_inverse_by_negation(rel_text_init)
        else:
            rel_text_full = rel_text_init
        
        # 构建文本关系图
        print("构建文本关系图 G_R^{TEXT}...")
        train_data = build_text_relation_graph(train_data, rel_text_full, threshold=threshold, top_percent=top_percent)
        valid_data = build_text_relation_graph(valid_data, rel_text_full, threshold=threshold, top_percent=top_percent)
        test_data = build_text_relation_graph(test_data, rel_text_full, threshold=threshold, top_percent=top_percent)
        
        # SEMMA HYBRID: 验证集自适应开关
        if getattr(cfg.text, 'hybrid', False) and hasattr(model, 'relation_model') and hasattr(model.relation_model, 'alpha'):
            print("\n" + separator)
            print("SEMMA HYBRID: 验证集自适应开关...")
            print(separator)
            
            model.relation_model.alpha = 1.0
            print("测试文本分支 (alpha=1.0)...")
            mrr_text = test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            
            model.relation_model.alpha = 0.0
            print("测试结构分支 (alpha=0.0)...")
            mrr_struct = test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            
            chosen = 1.0 if mrr_text >= mrr_struct else 0.0
            model.relation_model.alpha = chosen
            
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(f"HYBRID选择: text_mrr={mrr_text:.4f}, struct_mrr={mrr_struct:.4f}, 选择alpha={chosen}")
                logger.warning(separator)
        
        print("\n" + separator)
        print("SEMMA 文本语义生成完成!")
        print(separator)
    
    train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger)
