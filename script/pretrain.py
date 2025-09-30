import os
import sys
import copy
import math
import pprint
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
from ultra.text_semantics import (
    load_or_generate_relation_semantics,
    build_relation_embeddings,
    make_inverse_by_negation,
)
from ultra.tasks import build_text_relation_graph


separator = ">" * 30
line = "-" * 30


def multigraph_collator(batch, train_graphs):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

    batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()
    return graph, batch

# here we assume that train_data and valid_data are tuples of datasets
def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, batch_per_epoch=None):

    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([
        torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
        for g in train_data
    ])
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler, collate_fn=partial(multigraph_collator, train_graphs=train_data))

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
                # now at each step we sample a new graph and edges from it
                train_graph, batch = batch
                batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_graph, batch)
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
        result = test(cfg, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    
    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = []
    for test_graph, filters in zip(test_data, filtered_data):

        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = math.factorial(num_sample - 1) / \
                                    math.factorial(i) / math.factorial(num_sample - i - 1)
                            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                logger.warning("%s: %g" % (metric, score))
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            logger.warning(separator)

    avg_metric = sum(all_metrics) / len(all_metrics)
    return avg_metric


if __name__ == "__main__":
    # 返回参数对象args和变量字典vars
    args, vars = util.parse_args()
    # 加载配置文件
    cfg = util.load_config(args.config, context=vars)
    # 根据配置文件创建工作目录，并返回路径，路径在时间命名的文件夹下
    working_dir = util.create_working_directory(cfg)
    # 配置随机种子，保证实验可重复性，util.get_rank()用于分布式训练时区分不同进程，每个进程的种子略有不同
    torch.manual_seed(args.seed + util.get_rank())
    # 获取主日志记录器，用于输出训练、验证等过程中的信息
    logger = util.get_root_logger()
    # 判断当前进程是否为主进程，分布式训练时只在主进程输出日志
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed) # 当前的随机种子
        logger.warning("Config file: %s" % args.config) # 当前使用的配置文件路径
        logger.warning(pprint.pformat(cfg)) # 输出完整的配置内容，格式化显示，方便后续流程控制
    
    # 获取任务名称，如预训练、微调等
    task_name = cfg.task["name"]
    # 根据配置cfg构建数据集对象，包含训练、验证、测试数据
    dataset = util.build_dataset(cfg)
    # 根据配置获取计算设备
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]
    
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = [copy.deepcopy(vd) for vd in valid_data]
        for graph in short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
        
        short_valid = [sv.to(device) for sv in short_valid]

    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)
    
    assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"

    # for transductive setting, use the whole graph for filtered ranking
    filtered_data = [
        Data(
            edge_index=torch.cat([trg.target_edge_index, valg.target_edge_index, testg.target_edge_index], dim=1), 
            edge_type=torch.cat([trg.target_edge_type, valg.target_edge_type, testg.target_edge_type,]),
            num_nodes=trg.num_nodes).to(device)
        for trg, valg, testg in zip(train_data, valid_data, test_data)
    ]

    # =========================
    # SEMMA: 在线生成关系文本语义并构建文本关系图（预训练，多图）
    # =========================
    if hasattr(cfg, 'text') and cfg.text is not None and getattr(cfg.text, 'generate', False):
        # 配置
        api_key = getattr(cfg.text, 'api_key', '')
        # llm_model = getattr(cfg.text, 'llm_model', 'gpt-4o-2024-11-20')
        llm_model = getattr(cfg.text, 'llm_model', 'Qwen3-8B')
        cache_dir = getattr(cfg.text, 'cache_dir', '../../../../cache')
        combine = getattr(cfg.text, 'combine', 'COMBINED_SUM')
        threshold = getattr(cfg.text, 'threshold', 0.8)
        top_percent = getattr(cfg.text, 'top_percent', None)
        force_regen = getattr(cfg.text, 'force_regenerate', False)


    def extract_relations_and_examples():
        import os
        import glob
        relation_set = set()  # 存储不重复的关系
        relation_first_triple = {}  # 存储每个关系第一次出现的三元组 (h, r, t)
        # 查找 kg-datasets 下所有层级的 train.txt（根据实际目录结构调整 glob 模式）
        train_files = glob.glob(os.path.join(os.getcwd(), 'kg-datasets', '**', 'train.txt'), recursive=True)
        valid_files = glob.glob(os.path.join(os.getcwd(), 'kg-datasets', '**', 'valid.txt'), recursive=True)
        test_files  = glob.glob(os.path.join(os.getcwd(), 'kg-datasets', '**', 'test.txt'), recursive=True)
        all_files = train_files + valid_files + test_files
        # 如果 train.txt 直接在数据集目录下（如 kg-datasets/fb15k237/train.txt），则改用：
        # train_files = glob.glob(os.path.join(os.getcwd(), 'kg-datasets', '*', 'train.txt'))

        for file in all_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # 假设 train.txt 格式为：头实体\t关系\t尾实体（若为空格分隔，将 \t 改为 空格）
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            head, rel, tail = parts[0], parts[1], parts[2]
                            relation_set.add(rel)
                            # 只记录每个关系第一次出现的三元组
                            if rel not in relation_first_triple:
                                relation_first_triple[rel] = (head, rel, tail)
            except Exception as e:
                print(f"读取 {file} 时出错: {e}")
        
        # 生成有序的关系列表（保证每次运行顺序一致）
        relations = sorted(list(relation_set))
        # 生成“关系 -> 示例三元组”的映射
        rel_to_example = {}
        for rel in relations:
            if rel in relation_first_triple:
                rel_to_example[rel] = relation_first_triple[rel]
            else:
                # 理论上不会触发，因为 relations 来自 relation_set
                rel_to_example[rel] = (f"head_entity_{rel}", rel, f"tail_entity_{rel}")
        return relations, rel_to_example

    # 从所有 train.txt 提取关系和示例
    rel_names, rel_to_example = extract_relations_and_examples()

    # 生成/加载关系语义（缓存）
    cache_file = os.path.join(cache_dir, f"relation_semantics_{cfg.dataset['class'].lower()}_pretrain.pkl")
    cleaned, descs = load_or_generate_relation_semantics(
        rel_to_example=rel_to_example,
        cache_file=cache_file,
        api_key=api_key,
        model=llm_model,
        force_regenerate=force_regen,
    )

    # 文本嵌入
    rel_text_init = build_relation_embeddings(
            rel_names=rel_names,
            cleaned=cleaned,
            descs=descs,
            combine=combine,
        )

    # 以第一个训练图为基准判断是否需要补全逆关系
    base_graph = train_data[0] if len(train_data) > 0 else None
    # 自动补全逆关系嵌入，确保嵌入数量和图的关系数量一致
    if base_graph is not None:
        if rel_text_init.shape[0] * 2 == base_graph.num_relations:
            rel_text_full = make_inverse_by_negation(rel_text_init)
        elif rel_text_init.shape[0] == base_graph.num_relations:
            rel_text_full = rel_text_init
        else:
            raise ValueError(f"关系嵌入数量({rel_text_init.shape[0]})与图的关系数量({base_graph.num_relations})不匹配，请检查关系id映射和嵌入生成逻辑！")
    else:
        rel_text_full = rel_text_init

    # 强制断言所有图的关系数量和嵌入数量一致
    for g in train_data + valid_data + test_data:
        assert g.num_relations == rel_text_full.shape[0], f"关系数量不一致: {g.num_relations} vs {rel_text_full.shape[0]}"

    # 为所有图构建文本关系图
    train_data = [build_text_relation_graph(g, rel_text_full, threshold=threshold, top_percent=top_percent) for g in train_data]
    valid_data = [build_text_relation_graph(g, rel_text_full, threshold=threshold, top_percent=top_percent) for g in valid_data]
    test_data = [build_text_relation_graph(g, rel_text_full, threshold=threshold, top_percent=top_percent) for g in test_data]

        # HYBRID: 在验证集上选择 alpha（使用第一对图做快速选择）
    if getattr(cfg.text, 'hybrid', False) and hasattr(model, 'relation_model') and hasattr(model.relation_model, 'alpha'):
        model.relation_model.alpha = 1.0
        mrr_text = test(cfg, model, valid_data, filtered_data=filtered_data)
        model.relation_model.alpha = 0.0
        mrr_struct = test(cfg, model, valid_data, filtered_data=filtered_data)
        model.relation_model.alpha = 1.0 if mrr_text >= mrr_struct else 0.0



    train_and_validate(cfg, model, train_data, valid_data if "fast_test" not in cfg.train else short_valid, filtered_data=filtered_data, batch_per_epoch=cfg.train.batch_per_epoch)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)