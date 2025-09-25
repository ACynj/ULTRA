from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph


''' 新增代码 '''
# 构建文本关系图
'''
graph：输入的图对象，最终会将构建的文本关系图添加到该对象中。
rel_text_init：关系的文本向量嵌入，形状为 (num_relations, dim)。
threshold：余弦相似度阈值，用于过滤相似度低于该阈值的边。
top_percent：可选参数，用于限制每个关系只保留最相似的前 x% 关系。
返回：修改后的graph 对象，其中包含构建的文本关系图。
'''
def build_text_relation_graph(graph, rel_text_init, threshold=0.8, top_percent=None):

    '''
    tau赋值为输入的rel_text_init，形状为(R, d)（R= 关系数，d= 嵌入维度）。
    对tau进行 L2 归一化（p=2），归一化维度为最后一维（dim=-1）。
    目的：归一化后，两个向量的点积等价于它们的余弦相似度（简化后续相似度计算）。
    '''

    tau = rel_text_init  # (R, d)
    # 对tau最后一维进行归一化，不改变形状，仍然是(R, d)
    tau = torch.nn.functional.normalize(tau, p=2, dim=-1)
    # 余弦相似度，(R, d) @ (d, R) = (R, R)
    sim = tau @ tau.t()  # (R, R)

    # 移除自环，填充0
    # 获取关系数量
    R = sim.shape[0]
    # 填充对角线为0，对角线元素sim[i][i]表示关系与自身的相似度（恒为 1，无意义），需要移除自环边。
    sim.fill_diagonal_(0.0)

    '''
    如果基于 top_percent 筛选边
    '''
    if top_percent is not None:
        k = max(1, int(R * top_percent / 100.0)) # 至少保留一个关系，最多保留 前top_percent% 个关系
        # 对相似度矩阵的每一行（每个关系）取前k个最大的相似度值（topk_vals）和对应的列索引，形状为(R, k)
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
        # 构建边的行索引row_idx：生成[0,1,...,R-1]的序列，扩展成与topk_idx相同的形状（每个行索引对应一行的k个列索引）。，形状为(R, k)
        row_idx = torch.arange(R, device=sim.device).unsqueeze(1).expand_as(topk_idx)
        # 构建边索引edge_index：将行索引和列索引堆叠成(2, E)的形状（E为边数，第一行为起点，第二行为终点）。形状为(2, R*k)
        edge_index = torch.stack([row_idx.reshape(-1), topk_idx.reshape(-1)])
        # 构建边权重edge_weight：将topk_vals展平成一维，与edge_index的形状匹配。形状为(R*k,)
        edge_weight = topk_vals.reshape(-1)
        # 根据阈值过滤边，只保留相似度不低于threshold的边。
        mask = edge_weight >= threshold
        # 根据mask过滤边索引和权重，只保留符合条件的边。形状为(2, R*k)
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
    else:
        # 如果基于阈值筛选边
        mask = sim >= threshold
        row, col = mask.nonzero(as_tuple=True)
        # 构建边索引edge_index：将行索引和列索引堆叠成(2, E)的形状（E为边数，第一行为起点，第二行为终点）。形状为(2, E)
        edge_index = torch.stack([row, col])
        # 构建边权重edge_weight：将相似度矩阵中对应位置的值作为权重。形状为(E,)
        edge_weight = sim[row, col]

    # 构建文本关系图
    # 文本关系图的节点数为关系数
    num_nodes = graph.num_relations
    # 如果边数为0，则构建空图
    if edge_index.numel() == 0:
        # 构建空图
        edge_index = torch.empty(2, 0, dtype=torch.long, device=tau.device)
        edge_weight = torch.empty(0, dtype=tau.dtype, device=tau.device)
    text_rel_graph = Data(
        edge_index=edge_index, # 边索引
        edge_type=torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device), # 边类型 这里统一设为 0（因为所有边都是 “语义相似” 类型）
        edge_weight=edge_weight, # 边权重
        num_nodes=num_nodes, # 节点数
        num_relations=1 # 关系数 边的类型数量，这里为 1（仅 “语义相似” 一种类型）
    )

    graph.text_relation_graph = text_rel_graph
    graph.rel_text_init = rel_text_init
    return graph
