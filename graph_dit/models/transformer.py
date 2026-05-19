import torch
import torch.nn as nn

import utils
from models.layers import Attention, Mlp
from models.conditions import TimestepEmbedder, CategoricalEmbedder, ClusterContinuousEmbedder
import dgl

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def batch_shortest_path_matrix(graphs, max_nodes=None, max_distance=None, unreachable_value=-1):
    """
    批量计算多个DGL图中所有节点对之间的最短路径距离矩阵
    
    参数:
        graphs: DGL图对象的列表
        max_nodes: 最大节点数，用于填充。如果为None，使用最大图的节点数
        max_distance: 最大考虑距离，超过此距离的路径将设置为unreachable_value
        unreachable_value: 不可达路径的填充值
        
    返回:
        batch_dist_matrices: PyTorch张量，形状为(batch_size, max_nodes, max_nodes)
        actual_nodes: 每个图的实际节点数列表
    """
    batch_size = len(graphs)
    
    # 确定最大节点数
    if max_nodes is None:
        max_nodes = max([g.num_nodes() for g in graphs] + [0]) # Add [0] to handle empty graphs list
    
    # 初始化输出数组
    batch_dist_matrices = torch.full((batch_size, max_nodes, max_nodes), (unreachable_value), dtype=torch.long)
    
    for i, g in enumerate(graphs):
        num_nodes = g.num_nodes()
        
        if num_nodes == 0:
            continue
        # 初始化当前图的距离矩阵 (This line is from your selection)
        current_dist_matrix = torch.full((num_nodes, num_nodes), unreachable_value, dtype=torch.long)
        
        # 计算最短路径 (This comment is from your selection)
        if num_nodes > 0:
            dist_from_dgl = dgl.shortest_dist(g)
            current_dist_matrix = dist_from_dgl.long()
        
        # 如果设置了最大距离，将超过max_distance的值替换为unreachable_value
        if max_distance is not None:
            # 注意：需要先处理非unreachable_value的值
            # 创建一个掩码，标记那些不是unreachable_value且大于max_distance的路径
            mask = (current_dist_matrix != (unreachable_value)) & (current_dist_matrix > max_distance)
            current_dist_matrix[mask] = (unreachable_value)
        current_dist_matrix[current_dist_matrix == -1] = unreachable_value    
        
        # 将计算结果填入输出数组
        batch_dist_matrices[i, :num_nodes, :num_nodes] = current_dist_matrix
    batch_dist_matrices = batch_dist_matrices.to(graphs[0].device)
    return batch_dist_matrices

class DistanceEmbedder(nn.Module):
    def __init__(self, max_distance: int, embedding_dim: int):
        """
        Embeds shortest distances, typically from an adjacency matrix.

        Args:
            max_distance (int): The maximum distance value to be uniquely embedded.
                                Distances are expected to be integers from 0 to max_distance.
                                Values in the input distance_matrix greater than max_distance
                                will be clamped to max_distance.
                                Values less than 0 will be clamped to 0.
            embedding_dim (int): The dimension of the output embedding for each distance.

        """
        super().__init__()

        self.max_distance = max_distance
        # Number of embeddings: 0, 1, ..., max_distance. So, max_distance + 1 embeddings.
        self.embedding = nn.Embedding(
            num_embeddings=max_distance + 1,
            embedding_dim=embedding_dim
        )
        self._init_weights()

    def _init_weights(self):
        # Example: Initialize with a normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Transforms a distance matrix into an embedded representation.

        Args:
            distance_matrix (torch.Tensor): A tensor containing shortest distances.
                Expected shape: (..., N, N), where N is the number of entities.
                Values should be integers. Values will be clamped to the range
                [0, max_distance] before being passed to the embedding layer.

        Returns:
            torch.Tensor: The embedded distance matrix.
                Shape: (..., N, N, embedding_dim).
        """
        # Ensure input is long tensor for nn.Embedding
        distance_matrix_long = distance_matrix.long()

        # Clamp values to be within the range [0, self.max_distance] 
        # This handles distances > max_distance and < 0.
        clamped_distances = torch.clamp(distance_matrix_long, 0, self.max_distance)
        
        embedded_distances = self.embedding(clamped_distances)
        return embedded_distances  

class Denoiser(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.num_heads = self.cfg.model.num_heads
        self.ydim = self.cfg.model.ydim
        self.Xdim = self.cfg.model.Xdim
        self.Edim = self.cfg.model.Edim
        self.hidden_size = self.cfg.model.hidden_size
        self.max_n_nodes = self.cfg.model.max_n_nodes
        self.mlp_ratio = self.cfg.model.mlp_ratio
        self.depth = self.cfg.model.depth
        self.num_rel = self.cfg.num_rel
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        #### true ablation study ###############
        self.num_classes = self.num_rel*2+1    
        # self.num_classes = self.num_rel*2    
        #### true ablation study ###############
        
        self.distance = self.cfg.model.distance
        self.max_distance = self.cfg.model.max_distance
        self.onehot = self.cfg.model.onehot
        self.onehot = True
        
        if self.onehot:
            if self.distance=='onehot':
                self.x_embedder = nn.Linear(self.Xdim + self.max_n_nodes * (self.num_classes+1), self.hidden_size, bias=False)
            else:
                self.x_embedder = nn.Linear(self.Xdim + self.max_n_nodes * (self.num_classes), self.hidden_size, bias=False)
                # self.x_embedder = nn.Linear(self.max_n_nodes * (self.num_classes), self.hidden_size, bias=False)
        else:
            self.mask_rel_emb = nn.Parameter(torch.Tensor(1,self.Edim))
            self.x_embedder = nn.Linear(self.Xdim + self.max_n_nodes * self.Edim, self.hidden_size, bias=False)
            # self.project = nn.Linear(self.Edim, self.Edim, bias=False)
            # self.x_embedder = nn.Linear(self.max_n_nodes * self.Edim, self.hidden_size, bias=False)
        # self.y_embedding_list = torch.nn.ModuleList()

        # self.y_embedding_list.append(ClusterContinuousEmbedder(2, hidden_size, drop_condition))
        # for i in range(ydim - 2):
        #     if task_type == 'regression':
        #         self.y_embedding_list.append(ClusterContinuousEmbedder(1, hidden_size, drop_condition))
        #     else:
        #         self.y_embedding_list.append(CategoricalEmbedder(2, hidden_size, drop_condition))

        self.encoders = nn.ModuleList(
            [
                SELayer(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.depth)
            ]
        )

        self.out_layer = OutLayer(
            max_n_nodes=self.max_n_nodes,
            hidden_size=self.hidden_size,
            atom_type=self.Xdim,
            bond_type=self.Edim if not self.onehot else self.num_classes,
            mlp_ratio=self.mlp_ratio,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            distance=self.distance
        )
        if self.distance!='none':
            #待改进错误
            # self.distance_embedder = nn.Embedding(self.max_distance+2, self.Edim if not self.onehot else self.num_classes)
            self.distance_embedder = DistanceEmbedder(self.max_distance, self.Edim if not self.onehot else self.num_classes)
        #### rgcn ablation study ###############
        # self.node_embedding = nn.Parameter(torch.Tensor(self.max_n_nodes,self.Xdim))
        # torch.nn.init.xavier_normal_(self.node_embedding, gain=nn.init.calculate_gain('relu'))
        #### ablation study ###############
        ########## t condition ablation study ###############
        # self.tembedding = nn.Parameter(torch.Tensor(1,self.hidden_size))
        # torch.nn.init.xavier_normal_(self.tembedding, gain=nn.init.calculate_gain('relu'))
        ############ t condition ablation study ###############
        # 关系类型 -> attention bias 的标量参数（不含空边一类）
        # num_classes = num_rel * 2 + 1, 这里只对前 2*num_rel 施加 bias
        self.rel_bias = nn.Parameter(torch.zeros(self.num_rel * 2))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.encoders :
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.out_layer.adaLN_modulation[0], 0)
        if not self.onehot:
            torch.nn.init.xavier_normal_(self.mask_rel_emb, gain=nn.init.calculate_gain('relu'))
        
    def batch_graph_to_tensor(self, inputgraph,rel_head_emb, rel_tail_emb):
        """
        inputgraph: list of DGLGraph
        Returns:
            x: [bs, max_node, node_feat_dim]
            e: [bs, max_node, max_node, edge_feat_dim]
            nodemax: int
        """
        num_classes = self.num_classes
        node_feat_key='h'
        edge_feat_key='rel'
        nodemax = self.max_n_nodes
        
        bs = len(inputgraph)
        # device = inputgraph[0].ndata[node_feat_key].device
        device = inputgraph[0].edata[edge_feat_key].device
        
        node_feat_dim = self.Xdim
        edge_feat_dim = self.Edim
        if self.distance!='none':
            distance_matrixs = batch_shortest_path_matrix(inputgraph, max_nodes=nodemax, 
                                                        max_distance=self.max_distance,unreachable_value=self.max_distance+1)
            # 对distance_matrixs进行归一化
            # distance_matrixs = (distance_matrixs - distance_matrixs.min()) / (distance_matrixs.max() - distance_matrixs.min())
            # distance_matrixs = (distance_matrixs < self.max_distance).float() # Explicitly convert boolean to 0.0 or 1.0
            distance_embeddings = self.distance_embedder(distance_matrixs)
        # Collect node features with padding, preserving grad
        x_list = []
        # Collect edge features with padding, preserving grad，默认都=self.mask_rel_emb
        if self.onehot:
            # Initialize edge features as one-hot encoding
            if self.distance=='onehot':
                e = torch.zeros((bs, nodemax, nodemax, num_classes+1), device=device)
                e[:, :, :, self.num_rel*2] = 1
                # Assign normalized distances to the last channel of e
                e[:, :, :, -1] = distance_matrixs
            else:
                e = torch.zeros((bs, nodemax, nodemax, num_classes), device=device)
        #### true ablation study ###############
                e[:, :, :, self.num_rel*2] = 1
        #### true ablation study ###############
                
                
        else: 
            e = self.mask_rel_emb.repeat(bs, nodemax, nodemax, 1).clone()
            if self.distance=='emb':
                e+= distance_embeddings
           
        for i,g in enumerate(inputgraph):
            n = g.num_nodes()
            # feat = self.node_embedding
            #### rgcn ablation study###############
            feat = g.ndata[node_feat_key]
            if n < nodemax:
                pad = torch.zeros(nodemax - n, node_feat_dim, device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
            ####rgcn ablation study###############
            
            x_list.append(feat)
            
            # Collect edge features
            src, dst = g.edges()
            rel = g.edata[edge_feat_key]
            # rel是一个一维的tensor，表示边的类型，生成onehot编码
            if self.onehot:
                if self.distance=='onehot':
                    e[i, src, dst] += torch.nn.functional.one_hot(
                        rel, num_classes=num_classes + 1
                    ).to(e.device)
                else:
                    e[i, src, dst] += torch.nn.functional.one_hot(
                        rel, num_classes=num_classes
                    ).to(e.device)
        #### true ablation study ###############   
                e[i, src, dst,self.num_rel*2] = 0
        #### true ablation study ###############
                
            else:
                # 使用高效的平均池化处理所有该位置的 rel embedding
                rh_idx = rel < self.num_rel  # 关系属于 head embedding 的索引
                rt_idx = rel >= self.num_rel  # 关系属于 tail embedding 的索引
                
                # 清空特征
                e[i, src, dst] = torch.zeros_like(e[i, src, dst])
                
                # 使用scatter_add进行高效的累加和计数
                # 创建位置索引
                pos_idx = src * nodemax + dst  # 将2D位置转换为1D索引
                
                # 累加embedding
                flat_e = e[i].view(-1, self.Edim)
                
                # 处理 head embedding
                if rh_idx.any():
                    flat_e.scatter_add_(0, pos_idx[rh_idx].unsqueeze(1).expand(-1, self.Edim), 
                                      rel_head_emb[rel[rh_idx]])
                
                # 处理 tail embedding
                if rt_idx.any():
                    flat_e.scatter_add_(0, pos_idx[rt_idx].unsqueeze(1).expand(-1, self.Edim), 
                                      rel_tail_emb[rel[rt_idx] - self.num_rel])
                
                # 计算每个位置的边数进行平均
                edge_counts = torch.zeros(nodemax * nodemax, device=e.device, dtype=torch.float32)
                edge_counts.scatter_add_(0, pos_idx, torch.ones_like(pos_idx, dtype=torch.float32))
                
                # 应用平均池化：除以边数（避免除零）
                edge_counts = edge_counts.clamp(min=1.0)  # 防止除零
                flat_e = flat_e / edge_counts.unsqueeze(1)
                
                # 重新整形回原始形状
                e[i] = flat_e.view(nodemax, nodemax, self.Edim)  
        
        if self.onehot and self.distance=='emb':
            e+= distance_embeddings
        # import ipdb;ipdb.set_trace()
        x = torch.stack(x_list, dim=0)  # [bs, nodemax, node_feat_dim]
        
        # nodemask: [bs, nodemax]，True表示有效节点，False表示padding,device与x一致
        # nodemask = (x.abs().sum(dim=-1) != 0)
        # nodemask: [bs, nodemax]，True 表示有效节点，False 表示 padding
        nodemask = torch.arange(nodemax, device=device).unsqueeze(0) < torch.tensor(
            [g.num_nodes() for g in inputgraph], device=device
        ).unsqueeze(1)
        return x, e, nodemask
    
    def forward(self,inputgrpah, t, supportflag,
                rel_head_emb,rel_tail_emb, sample_mode=False):
        
        # if unconditioned:
        #     force_drop_id = torch.ones_like(y[:, 0])
        if isinstance(inputgrpah, list):
            x, e, node_mask = self.batch_graph_to_tensor(inputgrpah,rel_head_emb,rel_tail_emb)
        else:
            assert False, "inputgraph should be a list of DGLGraph"        
        x_in, e_in = x, e
        bs, n, _ = x.size()
        x = torch.cat([x, e.reshape(bs, n, -1)], dim=-1)
        # x=e.reshape(bs, n, -1)
        x = self.x_embedder(x)

        t_condition = self.t_embedder(t)
        # for i in range(1, self.ydim):
        #     if i == 1:
        #         c2 = self.y_embedding_list[i-1](y[:, :2], self.training, force_drop_id, t)
        #     else:
        #         c2 = c2 + self.y_embedding_list[i-1](y[:, i:i+1], self.training, force_drop_id, t)
        c = t_condition
        ######tcondition ablation study###############
        # c = self.tembedding.expand(bs, -1)
        ######ablation study###############
        # === 基于 inputgraph 构造每对节点 (i,j) 的关系 bias ===
        # e_in: [bs, n, n, num_classes] (one-hot, 最后一维: 2*num_rel + 1)
        attn_bias = None
        if self.onehot:
            
            # 只对真实关系类型施加 bias，忽略最后一维空边
            num_rel_classes = self.num_rel * 2
            rel_feat = e_in[..., :num_rel_classes]  # [bs, n, n, 2*num_rel]
            # self.rel_bias: [2*num_rel]
            # M_bias(b,i,j) = sum_k rel_feat[b,i,j,k] * rel_bias[k]
            attn_bias_matrix = (rel_feat * self.rel_bias.view(1, 1, 1, num_rel_classes)).sum(dim=-1)
            # [bs, 1, n, n]，对所有 head 共享
            attn_bias = attn_bias_matrix.unsqueeze(1)

        for i, block in enumerate(self.encoders):
            x = block(x, c, node_mask, attn_bias=attn_bias)

        # X: B * N * dx, E: B * N * N * de
        X, E, y = self.out_layer(x, x_in, e_in, c, t, node_mask,supportflag)
        return utils.PlaceHolder(X=X, E=E, y=y)
        # return utils.PlaceHolder(X=X, E=E, y=y),node_mask
        # return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class SELayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.dropout = 0.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, **block_kwargs
        )

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=self.dropout,
        )
        #####attention ablation study######
        # self.mlp1 = Mlp(
        #     in_features=hidden_size,
        #     hidden_features=hidden_size,
        #     drop=self.dropout,
        # )
        #####attention ablation study######

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c, node_mask, attn_bias=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * modulate(
            self.norm1(self.attn(x, node_mask=node_mask, attn_bias=attn_bias)),
            shift_msa,
            scale_msa,
        )
        ####attention ablation study######
        # x = x + gate_msa.unsqueeze(1) * modulate(self.norm1(self.mlp1(x)), shift_msa, scale_msa)
        ####ablation study######
        
        x = x + gate_mlp.unsqueeze(1) * modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x


class OutLayer(nn.Module):
    # Structure Output Layer
    def __init__(self, max_n_nodes, hidden_size, atom_type, bond_type, mlp_ratio, num_heads=None,num_classes=None,distance=True):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        if distance=='onehot':
            self.bond_type = bond_type + 1
        final_size = atom_type + max_n_nodes * self.bond_type
        # final_size =  max_n_nodes * self.bond_type
        self.xedecoder = Mlp(in_features=hidden_size, 
                            out_features=final_size, drop=0)

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.bond_type, self.num_classes , bias=False)

    def forward(self, x, x_in, e_in, c, t, node_mask,supportflag=None):
        x_all = self.xedecoder(x)
        B, N, D = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        
        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out

        bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)
        # bond_out = x_all[:, :, :].reshape(B, N, N, self.bond_type)
        bond_out = e_in + bond_out
        
        ##### standardize adj_out
        # edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        # edge_mask = ~((node_mask)[:, :, None] & (node_mask)[:, None, :])
        # diag_mask = (
        #     torch.eye(N, dtype=torch.bool)
        #     .unsqueeze(0)
        #     .expand(B, -1, -1)
        #     .type_as(edge_mask)
        # )
        
        # bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        # bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        
        #  # === 支持集边mask 进行loss ===
        # if supportflag is not None:
        #     for b, g in enumerate(supportflag):
        #         src, dst = g.edges()
        #         bond_out[b, src, dst, :] = 0  # 或其它你想填的值
        #         bond_out[b, dst, src, :] = 0  # 如果是无向图，双向mask

        # bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))
        bond_out = self.classifier(bond_out)

        return atom_out, bond_out, None,
