import torch
from torch import nn

from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet

class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)

        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        try:
            relation_representations = self.relation_model(
                data.relation_graph,
                query=query_rels,
                text_rel_graph=getattr(data, 'text_relation_graph', None),
                rel_text_init=getattr(data, 'rel_text_init', None)
            )
        except TypeError:
            relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch)
        
        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    
    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output
    
    
#  定义TextRelNBFNet类，继承自BaseNBFNet（基础的 NBFNet 实现），因此会复用父类中关于图推理的核心逻辑。
class TextRelNBFNet(BaseNBFNet):
    # input_dim：输入维度，即关系文本向量的维度。
    # hidden_dims：隐藏层维度，是一个列表，包含多个中间层维度。
    # num_relation：关系数，即边的类型数量，这里为 1（仅 “语义相似” 一种类型）。
    # **kwargs：其他参数，传递给父类初始化方法。
    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        # 定义多层 NBFNet 层
        self.layers = nn.ModuleList()
        # 遍历每一层，创建 NBFNet 层
        for i in range(len(self.dims) - 1):
            # 定义一个广义的关系卷积层，用以处理关系信息的图数据
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )
            # self.dims[i]：当前层的输入维度
            # self.dims[i + 1]：当前层的输出维度
            # num_relation：关系数，即边的类型数量，这里为 1（仅 “语义相似” 一种类型）。
            # self.dims[0]：初始层的输入维度
            # self.message_func：消息函数，这里为 “distmult”。
            # self.aggregate_func：聚合函数，这里为 “sum”。
            # self.layer_norm：是否使用层归一化，这里为 False。
            # self.activation：激活函数，这里为 “relu”。
            # dependent：是否依赖关系，这里为 False。

        # 如果需要拼接隐藏层，则计算特征维度
        # 计算拼接后的特征维度，包括所有隐藏层维度之和，再加上初始输入维度
        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            # 定义一个多层感知机，用以将拼接后的特征维度映射到初始输入维度
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    # 定义bellmanford方法，用以在文本关系图上进行消息传递
    # data: 文本关系图
    # initial_node_features: 初始节点特征
    def bellmanford(self, data, initial_node_features):
        # initial_node_features: (batch_size, num_nodes, input_dim)
        batch_size = initial_node_features.shape[0]
        node_features = initial_node_features
        # 初始化查询向量，形状为 (batch_size, input_dim)
        query = torch.zeros(batch_size, self.dims[0], device=node_features.device, dtype=node_features.dtype)
        boundary = node_features
        size = (data.num_nodes, data.num_nodes)
        # 如果文本关系图data包含edge_weight（语义相似度），则直接使用；否则默认边权重为 1（表示不考虑权重差异）
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            edge_weight = torch.ones(data.edge_index.shape[1], device=node_features.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            hidden = layer(layer_input, query, boundary, data.edge_index, getattr(data, 'edge_type', None), size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        if self.concat_hidden:
            output = torch.cat(hiddens + [boundary], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        # 返回节点特征和边权重
        # 节点特征：(batch_size, num_nodes, input_dim)
        # 边权重：(batch_size, num_edges)
        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, text_rel_graph, rel_text_init):
        if rel_text_init.dim() == 2:
            # 如果初始节点特征的维度为 2，则扩展为 3 维，形状为 (1, R, d)
            rel_text_init = rel_text_init.unsqueeze(0)
        # 调用bellmanford方法，传入文本关系图和初始特征，获取更新后的节点特征（node_feature）并返回。这些特征是关系的语义表示，融合了文本关系图中的语义相似性信息。
        output = self.bellmanford(text_rel_graph, rel_text_init)["node_feature"]
        return output


class SemmaRelModel(nn.Module):
    # fusion: 融合方式，这里为 “mlp” 或 “attn”
    # alpha: 文本分支的权重，这里为 1.0
    # struct_kwargs: 结构分支的参数
    # text_kwargs: 文本分支的参数
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 fusion="mlp",
                 alpha=1.0,
                 struct_kwargs=None,
                 text_kwargs=None):
        super().__init__()
        struct_kwargs = struct_kwargs or {}
        text_kwargs = text_kwargs or {}
        self.struct = RelNBFNet(input_dim=input_dim, hidden_dims=hidden_dims, num_relation=4, **struct_kwargs)
        self.text = TextRelNBFNet(input_dim=input_dim, hidden_dims=hidden_dims, num_relation=1, **text_kwargs)
        self.alpha = alpha
        self.fusion = fusion
        if fusion == "mlp":
            self.fuse = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
            )
            
        # 第一层：输入维度为input_dim*2（结构特征和文本特征拼接后的维度），输出input_dim，接 ReLU 激活。
        # 第二层：输入维度为input_dim，输出input_dim（将拼接后的特征压缩回原始维度）。
        elif fusion == "attn":
            self.query_proj = nn.Linear(input_dim, input_dim)
            self.key_proj = nn.Linear(input_dim, input_dim)
            self.value_proj = nn.Linear(input_dim, input_dim)
            self.out_proj = nn.Linear(input_dim, input_dim)
        # query_proj：查询投影层，将结构特征映射到查询空间。
        # key_proj：键投影层，将文本特征映射到查询空间。
        # value_proj：值投影层，将文本特征映射到查询空间。
        # out_proj：输出投影层，将结构特征和文本特征融合后的结果映射回原始维度。
        else:
            raise ValueError("Unknown fusion type")

    def forward(self, rel_graph, query, text_rel_graph=None, rel_text_init=None):
        # 调用self.struct（RelNBFNet）处理结构关系图rel_graph和查询query，输出结构特征h。
        h = self.struct(rel_graph, query)  # (bs, R, d)
        # 如果文本关系图text_rel_graph为None，或者初始关系文本向量rel_text_init为None，或者文本分支权重alpha为0.0，则直接返回结构特征h。
        if text_rel_graph is None or rel_text_init is None or self.alpha == 0.0:
            return h
        # 如果初始关系文本向量rel_text_init的维度为2，则扩展为3维，形状为(bs, R, d)。
        if rel_text_init.dim() == 2:
            bs = h.shape[0]
            rel_text_init = rel_text_init.unsqueeze(0).expand(bs, -1, -1)
        # 调用self.text（TextRelNBFNet）处理文本关系图text_rel_graph和初始关系文本向量rel_text_init，输出文本特征z。
        z = self.text(text_rel_graph, rel_text_init)  # (bs, R, d)
        # 如果融合方式为 “mlp”，则将结构特征h和文本特征z拼接后，通过self.fuse（MLP）进行融合，输出融合后的特征fused。
        # self.fuse: 多层感知机，将结构特征和文本特征拼接后，通过MLP进行融合，输出融合后的特征fused。
        if self.fusion == "mlp":
            fused = self.fuse(torch.cat([h, self.alpha * z], dim=-1))
        else:
            # 如果融合方式为 “attn”，则将结构特征h和文本特征z拼接后，通过self.query_proj（查询投影层）进行投影，得到q。
            # 通过self.key_proj（键投影层）进行投影，得到k。
            # 通过self.value_proj（值投影层）进行投影，得到v。
            # 通过self.out_proj（输出投影层）进行投影，得到fused。
            q = self.query_proj(h)
            k = self.key_proj(z)
            v = self.value_proj(z)
            attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / (q.shape[-1] ** 0.5), dim=1)
            fused = self.out_proj(h + self.alpha * attn * v)
        return fused

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines
    Almost the same as EntityNBFNet except that 
    (1) we already get the initial node features at the forward pass time 
    and don't have to read the triples batch
    (2) we get `query` from the outer loop
    (3) we return a distribution over all nodes (assuming t_index = all nodes)
    """
    
    def bellmanford(self, data, node_features, query, separate_grad=False):
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, node_features, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(data, node_features, query)  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1) # (bs, num_nodes)
        return score  

    


