import itertools
from dataclasses import dataclass

import hydra.utils
import pytorch_lightning as L

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict
import metrics
from models.transformer import Denoiser
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete
from diffusion.noise_schedule2 import get_noise 
import utils
import numpy as np
import itertools
import dgl
from models.ent_init_model import EntInit
from models.rgcn_model import RGCN
from models.transformer import batch_shortest_path_matrix
def evaluate_link_prediction(model_out, true_src_nodes, true_dst_nodes, true_rel_types, threshold, apply_sigmoid=False):
    """
    评估模型在链接预测任务上的表现，包括召回率和精确率。

    参数:
    model_out (np.ndarray): 模型输出，形状为 (num_nodes, num_nodes, num_relation_types)。
                             值可以是原始 logits 或已经是概率。
    true_src_nodes (list or np.ndarray): 真实边的源节点列表。
    true_dst_nodes (list or np.ndarray): 真实边的目标节点列表。
    true_rel_types (list or np.ndarray): 真实边的关系类型列表 (整数索引)。
    threshold (float): 用于判断边是否被预测的阈值。
    apply_sigmoid (bool): 如果 model_out 包含 logits，则设置为 True 以应用 sigmoid 函数。

    返回:
    tuple: (recall, precision, true_positives, total_predicted_positives)
           recall (float): 召回率 (TP / Total Actual Positives)。
           precision (float): 精确率 (TP / Total Predicted Positives)。
           true_positives (int): 模型正确预测的真实边的数量。
           total_predicted_positives (int): 模型预测为存在的所有边的总数。
    """
    num_nodes_model, _, num_relations_model = model_out.shape
    num_actual_true_edges = len(true_src_nodes)

    true_positives = 0
    # 计算 True Positives (TP)
    # TP: 真实存在的边，并且模型也预测存在
    for i in range(num_actual_true_edges):
        u, v, rel = true_src_nodes[i], true_dst_nodes[i], true_rel_types[i]

        if not (0 <= u < num_nodes_model and \
                0 <= v < num_nodes_model and \
                0 <= rel < num_relations_model):
            # print(f"警告: 真实边 ({u}, {v}, type={rel}) 的索引超出 model_out 范围。跳过此边进行TP计数。")
            continue

        score = model_out[u, v, rel]
        if apply_sigmoid:
            score = 1 / (1 + np.exp(-score))

        if score > threshold:
            true_positives += 1

    # 计算模型预测为正的总数 (TP + FP)
    # TP + FP: 所有模型预测分数超过阈值的边 (无论它们是否真实存在)
    total_predicted_positives = 0
    processed_model_out = model_out.copy() # Avoid modifying the original array if it's passed by reference
    if apply_sigmoid:
        processed_model_out = 1 / (1 + np.exp(-processed_model_out))

    total_predicted_positives = np.sum(processed_model_out[...,:24] > threshold)


    # 计算召回率
    recall = true_positives / num_actual_true_edges if num_actual_true_edges > 0 else 0.0

    # 计算精确率
    precision = true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0

    return recall, precision, true_positives, total_predicted_positives
def filter_topk_edges(pred, threshold, mask_prob,step=None):
    """
    过滤每个 batch 的 pred 张量，仅保留 top-k 个边。
    pred: (B, N, N, DIM)
    threshold: float
    mask_prob: (B,1)
    返回: 仅保留 topk 的布尔张量
    """
    B, N, _, DIM = pred.shape
    device = pred.device
    pred_flat = pred.view(B, -1)  # (B, N*N*DIM)
    # 计算每个 batch 要保留的边数
    num_edges = (pred_flat > threshold).sum(dim=1)
    topk_mask = topk_mask.view_as(pred)
    return topk_mask
  
def symmetrize_pred_matrix_vectorized(pred, rel_num):
    """
    向量化版本的对称操作
    pred: (B, N, N, DIM)
    rel_num: 关系数量
    """
    B, N, _, DIM = pred.shape
    device = pred.device
    
    # 创建下三角矩阵掩码，不包括对角线
    tril_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=-1)
    
    # 为每个维度创建反向维度的映射
    reverse_dim_map = torch.arange(DIM, device=device)
    
    # 正向关系映射到反向关系
    reverse_dim_map[:rel_num] = torch.arange(rel_num, rel_num * 2, device=device)
    # 反向关系映射到正向关系
    reverse_dim_map[rel_num:rel_num*2] = torch.arange(rel_num, device=device)
    # no_edge类别保持不变
    if DIM > rel_num * 2:
        reverse_dim_map[rel_num*2:] = torch.arange(rel_num*2, DIM, device=device)
    
    # 获取下三角位置的索引
    i_indices, j_indices = torch.where(tril_mask)
    
    for dim in range(DIM):
        reverse_dim = reverse_dim_map[dim]
        
        # 获取 (i,j,dim) 和 (j,i,reverse_dim) 位置的值
        original_ij = pred[:, i_indices, j_indices, dim]  # (B, num_lower_triangular)
        original_ji = pred[:, j_indices, i_indices, reverse_dim]  # (B, num_lower_triangular)
        
        # 计算平均值
        avg_value = (original_ij + original_ji) / 2
        
        # 对称赋值
        pred[:, i_indices, j_indices, dim] = avg_value
        pred[:, j_indices, i_indices, reverse_dim] = avg_value
    
    return pred

def t_sample_categorical(pred,threshold=3,mask_prob=None,support=None,step=None):
  
  device = pred.device
  B, N, _, DIM = pred.shape

  # 1) 将支持图批处理到一个 DGLGraph 里
  batched_support = dgl.batch(support).to(device)  # 合并到单个 DGLGraph
  src, dst = batched_support.edges()                       # 所有支持边
  rel = batched_support.edata['rel']                       # 这些边的标签

  # 2) 计算每个支持图的边数和节点数，以便映射回各自在 pred_E 里的位置
  num_edges_per_graph = batched_support.batch_num_edges()  # 每个支持图的边数
  num_nodes_per_graph = batched_support.batch_num_nodes()  # 每个支持图的节点数
  # graph_idx[b] 表示批图中边对应的原图索引
  graph_idx = torch.arange(B, device=device).repeat_interleave(num_edges_per_graph)

  # 3) 为 src, dst 去掉各自图在批图中的节点偏移量
  node_offsets = torch.cumsum(
      torch.cat([torch.tensor([0], device=device), num_nodes_per_graph[:-1]]),
      dim=0
  )
  src_in_orig = src - node_offsets[graph_idx]
  dst_in_orig = dst - node_offsets[graph_idx]

  # 4) 用高级索引把支持图里的 (graph_idx, src_in_orig, dst_in_orig, rel) 置为非可能值
  pred[graph_idx, src_in_orig, dst_in_orig, rel] = -1e10
  
  # 计算nodemask 和 diagmask
  nodemask = torch.arange(N, device=device).unsqueeze(0) < num_nodes_per_graph.unsqueeze(1)
  pairmask = nodemask.unsqueeze(2) & nodemask.unsqueeze(1)  # (B, N, N)
  pairmask = pairmask.unsqueeze(-1).expand_as(pred)

    # 计算 diagmask
  diagmask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
  diagmask = diagmask.unsqueeze(-1).expand_as(pred)
  pred = pred.masked_fill(diagmask, -1e10)  # 对角线置为非可能值
  pred = pred.masked_fill(~pairmask, -1e10)  # ~pairmask padding位置置为非可能值
  # pred = pred.softmax(dim=-1)  # 在最后一个维度上进行 softmax
  
  # 对pred进行value对称化处理
  pred = symmetrize_pred_matrix_vectorized(pred, rel_num=(pred.shape[-1]-1)//2)
  
  # pred = pred.softmax(dim=-1)  # 在最后一个维度上进行 softmax
  pred = F.sigmoid(pred)
  # distance_matrix = batch_shortest_path_matrix(support,max_nodes=pred.shape[1])
  
  # # 根据distance_matrix条件修改pred
  # # 如果distance不是大于0且小于2，则对应pred位置置为0
  # valid_distance_mask = (distance_matrix > 0) & (distance_matrix <2)  # (B, N, N)
  # valid_distance_mask = ~valid_distance_mask  # 取反，获取无效距离的位置

  # # 将invalid_distance_mask扩展到pred的最后一个维度
  # valid_distance_mask = valid_distance_mask.unsqueeze(-1).expand_as(pred)  # (B, N, N, DIM)9

  # # 将无效距离位置的pred值置为0
  # pred = pred.masked_fill(valid_distance_mask, 0.0)
  
  
  # pred= filter_topk_edges(pred, 0.999, mask_prob,step=step)
  pred= filter_topk_edges(pred, 0.999, mask_prob,step=step)
  # pred= filter_topk_edges(pred, 0.9992, mask_prob,step=step)
  
  # pred_list=[]
  # t_list = torch.linspace(0.5, 1, steps=200).tolist()
  # for t_itm in t_list:
  #   premask= filter_topk_edges(pred, t_itm, mask_prob)
  #   pred_list.append((premask,t_itm))
  # return pred_list
  
  return pred

def _sample_categorical(pred,threshold=3,mask_prob=None,support=None,step=None):
  # pred = torch.bernoulli(probs)  # 得到0/1
  # 假设 pred_E 的形状为 (B, N, N, DIM)
  
  no_edge_index = pred.shape[-1] - 1  # 获取 no_edge_index 的索引
  max_probs, max_indices = torch.max(pred, dim=-1)  # 找到每个位置的最大值及其索引

  # 创建掩码，标记最大值索引为 no_edge_index 的位置
  mask = max_indices == no_edge_index  # 形状为 (B, N, N)

  # 将掩码扩展到最后一个维度，以便应用于整个概率张量
  mask_expanded = mask.unsqueeze(-1).expand_as(pred)

  # 将掩码位置的所有值置为 0
  pred[mask_expanded] = -1e10  # 或者使用其他适当的值
  pred[~mask][-1]=-1e10  # 确保 no_edge_index 的位置被置为 -1e10
  device = pred.device
  B, N, _, DIM = pred.shape

  # 1) 将支持图批处理到一个 DGLGraph 里
  batched_support = dgl.batch(support).to(device)  # 合并到单个 DGLGraph
  src, dst = batched_support.edges()                       # 所有支持边
  rel = batched_support.edata['rel']                       # 这些边的标签

  # 2) 计算每个支持图的边数和节点数，以便映射回各自在 pred_E 里的位置
  num_edges_per_graph = batched_support.batch_num_edges()  # 每个支持图的边数
  num_nodes_per_graph = batched_support.batch_num_nodes()  # 每个支持图的节点数
  # graph_idx[b] 表示批图中边对应的原图索引
  graph_idx = torch.arange(B, device=device).repeat_interleave(num_edges_per_graph)

  # 3) 为 src, dst 去掉各自图在批图中的节点偏移量
  node_offsets = torch.cumsum(
      torch.cat([torch.tensor([0], device=device), num_nodes_per_graph[:-1]]),
      dim=0
  )
  src_in_orig = src - node_offsets[graph_idx]
  dst_in_orig = dst - node_offsets[graph_idx]

  # 4) 用高级索引把支持图里的 (graph_idx, src_in_orig, dst_in_orig, rel) 置为非可能值
  pred[graph_idx, src_in_orig, dst_in_orig, rel] = -1e10
  pred[graph_idx, src_in_orig, dst_in_orig, no_edge_index] = -1e10
  
  # 计算nodemask 和 diagmask
  nodemask = torch.arange(N, device=device).unsqueeze(0) < num_nodes_per_graph.unsqueeze(1)
  pairmask = nodemask.unsqueeze(2) & nodemask.unsqueeze(1)  # (B, N, N)
  pairmask = pairmask.unsqueeze(-1).expand_as(pred)

    # 计算 diagmask
  diagmask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
  diagmask = diagmask.unsqueeze(-1).expand_as(pred)
  pred = pred.masked_fill(diagmask, -1e10)  # 对角线置为非可能值
  pred = pred.masked_fill(~pairmask, -1e10)  # ~pairmask padding位置置为非可能值
  # pred = pred.softmax(dim=-1)  # 在最后一个维度上进行 softmax
  
  # 对pred进行value对称化处理
  pred = symmetrize_pred_matrix_vectorized(pred, rel_num=(pred.shape[-1]-1)//2)
  
  pred = pred.softmax(dim=-1)  # 在最后一个维度上进行 softmax
  # pred = F.sigmoid(pred)
  # distance_matrix = batch_shortest_path_matrix(support,max_nodes=pred.shape[1])
  
  # # 根据distance_matrix条件修改pred
  # # 如果distance不是大于0且小于2，则对应pred位置置为0
  # valid_distance_mask = (distance_matrix > 0) & (distance_matrix <2)  # (B, N, N)
  # valid_distance_mask = ~valid_distance_mask  # 取反，获取无效距离的位置

  # # 将invalid_distance_mask扩展到pred的最后一个维度
  # valid_distance_mask = valid_distance_mask.unsqueeze(-1).expand_as(pred)  # (B, N, N, DIM)9

  # # 将无效距离位置的pred值置为0
  # pred = pred.masked_fill(valid_distance_mask, 0.0)
  
  
  # pred= filter_topk_edges(pred, 0.999, mask_prob,step=step)
  pred= filter_topk_edges(pred, 0.999, mask_prob,step=step)
  # pred= filter_topk_edges(pred, 0.9992, mask_prob,step=step)
  
  # pred_list=[]
  # t_list = torch.linspace(0.5, 1, steps=200).tolist()
  # for t_itm in t_list:
  #   premask= filter_topk_edges(pred, t_itm, mask_prob)
  #   pred_list.append((premask,t_itm))
  # return pred_list
  
  return pred

def calculate_metrics(pred_triples, test_triples,seentriples=None):
    """
    计算预测三元组相对于测试三元组的精确率、召回率和 F1 分数。

    Args:
        pred_triples (np.ndarray): 预测的三元组数组，形状为 (num_pred, 3)。
        test_triples (np.ndarray): 真实（测试）的三元组数组，形状为 (num_test, 3)。

    Returns:
        tuple: 包含精确率 (precision)、召回率 (recall) 和 F1 分数 (f1) 的元组。
               如果分母为零，则对应指标返回 0.0。
    """
    # 将 NumPy 数组转换为元组集合，以便进行高效的集合操作
    # 确保数组是整数类型，如果不是，根据需要转换
    pred_set = set(map(tuple, pred_triples.astype(int)))
    seentriples = set(map(tuple, seentriples.astype(int)))
    test_set = set(map(tuple, test_triples.astype(int)))

    pred_set = pred_set - seentriples
    
    # 计算交集（正确预测的三元组）
    correct_predictions = pred_set.intersection(test_set)
    num_correct = len(correct_predictions)

    # 计算预测的三元组数量和真实的三元组数量
    num_predicted = len(pred_set)
    num_ground_truth = len(test_set)

    # 计算精确率
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0

    # 计算召回率
    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0
    recall = recall**0.5

    # 计算 F1 分数
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, num_predicted

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  # nlls: torch.FloatTensor
  # token_mask: torch.FloatTensor

def check_graph_unique(graph):
  """
  Check if the graph has unique edges.
  """
  src, dst = graph.edges()
  rel = graph.edata['rel']
  all_triples = torch.stack([src, dst, rel], dim=1)
  unique_triples = torch.unique(all_triples, dim=0)
  
  if len(unique_triples) == len(all_triples):
    return True
  else:
    return False
  
def merge_two_graphs(querygraph, supportgraph,needunique=False):
    """
    Merges two DGL graphs, ensuring unique edges and preserving 'rel' edge features.
    Assumes both graphs have the same node set and node IDs correspond.
    """
    if '_ID' in supportgraph.edata:
      supportgraph.edata.pop('_ID')
    if not querygraph:
      return supportgraph
    if querygraph.num_edges() == 0:
      return supportgraph
    num_nodes = supportgraph.num_nodes()

    
    # 更保险的 device 获取方式
    device = supportgraph.edata['rel'].device

    # Get edges from both graphs
    q_src, q_dst = querygraph.edges()
    s_src, s_dst = supportgraph.edges()

    # Concatenate edges
    all_src = torch.cat([q_src, s_src])
    all_dst = torch.cat([q_dst, s_dst])

    # Concatenate 'rel' features if they exist
    q_rel = querygraph.edata['rel']
    s_rel = supportgraph.edata['rel']
    all_rel = torch.cat([q_rel, s_rel])

    
    all_triples = torch.stack([all_src, all_dst,all_rel], dim=1)
    
    if needunique:
      unique_triples = torch.unique(all_triples, dim=0)
    else:
      unique_triples = all_triples
    # Extract unique source and destination nodes
    unique_src = unique_triples[:, 0]
    unique_dst = unique_triples[:, 1]

    # Create the merged graph with unique edges
    merged_g = dgl.graph((unique_src, unique_dst), num_nodes=num_nodes, device=device)

    # Assign 'rel' features to the unique edges if they exist
    unique_rel = unique_triples[:, 2]
    merged_g.edata['rel'] = unique_rel
    # merge ndata['orig_id'] query and support share the same node id
    
    if 'orig_id' in querygraph.ndata:
      merged_g.ndata['orig_id'] = querygraph.ndata['orig_id']
    
    return merged_g

def graph_inverse(graphs, num_rel, in_place=False):
    """
    Add inverse edges to each graph in the list.
    Args:
        graphs: list of DGLGraph
        num_rel: int, number of relations (for inverse rel offset)
        in_place: bool, whether to modify in place
    """
    processed_graphs = []
    for i, g in enumerate(graphs):
        if not in_place:
            # Clone the graph if modification should not happen in place
            g_new = g.clone()
        else:
            g_new = g # Modify the original graph object

        # Get original edges and relation types
        src, dst = g.edges()
        if 'rel' not in g.edata:
             raise ValueError(f"Graph {i} does not have 'rel' edge data.")
        rel = g.edata['rel']

        # Calculate inverse relation types
        inv_rel = rel + num_rel
        g_new.add_edges(dst, src, {'rel': inv_rel})

        if not in_place:
            processed_graphs.append(g_new)

    return graphs if in_place else processed_graphs

    
                
class Diffusion(L.LightningModule):
  def __init__(self,config,test_triples=None,seentriples=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.antithetic_sampling = False
    self.mask_index = -1
    self.unit_sample = self.config.model.unit_sample
    self.ent_init = EntInit(self.config)
    self.rgcn = RGCN(self.config)
    self.backbone = Denoiser(self.config)
    self.drop_condition=self.config.model.drop_condition
    self.T = self.config.model.diffusion_steps
    self.noise_schedule = get_noise(self.config,self.config.model.diffusion_noise_schedule)
    self.num_rel = self.config.num_rel
    self.max_n_nodes = self.config.model.max_n_nodes
    self.loss_type = self.config.model.loss_type
    self.support_flag = self.config.model.support_flag
    self.test_triples = test_triples
    self.seentriples = seentriples
    self.first_inverse = self.config.dataset.first_inverse
    self.classes = self.num_rel*2+1
    self.softmax = True
    
    # self.noise_schedule = PredefinedNoiseScheduleDiscrete(config.model.diffusion_noise_schedule,
    #                                                           timesteps=config.model.diffusion_steps)
    # self.noise = noise_schedule.get_noise(self.config)
    # self.metrics = metrics.Metrics(config)

    # if self.config.train.ema > 0:
    #   self.ema = models.ema.ExponentialMovingAverage(
    #     self._get_parameters(),
    #     decay=self.config.train.ema)
    # else:
    #   self.ema = None
    self.ema=None
    self.var_min = None
    
   
      
    self.neg_infinity = -100000.0
    # self._validate_configuration()
  # 待修改
  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise_schedule.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False


  # 待修改
  # def to(self, *args, **kwargs):
  #   self = super().to(*args, **kwargs) 
  #   self.metrics.to(*args, **kwargs)
  #   if hasattr(self.backbone, "mask") and self.config.model.attn_backend == 'sdpa':
  #     self.backbone.mask = self.backbone.mask.to(*args, **kwargs)
  #   elif hasattr(self.backbone, "mask") and self.config.model.attn_backend == 'flex':
  #     self.backbone.mask = self.backbone.mask.to(self.device)
  #   if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
  #     self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
  #     self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
  #   return self


  def on_load_checkpoint(self, checkpoint):
    pass    

  def on_save_checkpoint(self, checkpoint):
    pass
  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
 
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    # Apply updates directly in the logits matrix.
    # 待改进，意思就是预测出来了非空边，那就预测出来，给概率最大值0，但真实为空边的其实没有
    
    if isinstance(xt, list):
      for i,itm in enumerate(xt):
        src,drc = itm.edges()
        rel = itm.edata['rel']
        #待改进，有可能有多种关系类型
        # logits[i,src,drc] = self.neg_infinity
        #0就是最大
        logits[i,src,drc,rel] = 0
        #反向边
        # logits[i,drc,src] = self.neg_infinity
        logits[i,drc,src,rel+self.num_rel] = 0

    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def get_ent_emb(self, graphs):
      self.ent_init(graphs)
      ent_emb = self.rgcn(graphs)

      return ent_emb
  def forward(self, input_graph, sigma:torch.Tensor,
            sample_mode=False):
    """Returns log score."""
    # sigma = self._process_sigma(sigma)
    batch_size = len(input_graph)
    device = input_graph[0].edata['rel'].device
    
    
    # input_graph = graph_inverse(input_graph, self.num_rel,inplace=False)
    # ####### ablation study###############
    batch_g = dgl.batch(input_graph).to(device)
    self.get_ent_emb(batch_g)
    input_graph = dgl.unbatch(batch_g)
    # ####### ablation study###############
    
    with torch.amp.autocast('cuda', dtype=torch.float32):
      
      logits= self.backbone(input_graph, sigma,
                             None if self.support_flag else None,
                            rel_head_emb=self.ent_init.rel_head_emb,
                            rel_tail_emb=self.ent_init.rel_tail_emb,
                              sample_mode=sample_mode)

    assert torch.isnan(logits.E).any() == False
    
    if self.loss_type == 'bd3':
      return self._subs_parameterization(logits=logits.E,
                                      xt=x)
    else:
      return logits.E
      
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise_schedule.train()
    self._train_epoch_losses = [] # Initialize loss accumulator

    # self.metrics.reset()
    # assert self.metrics.train_nlls.nll.mean_value == 0
    # assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch,batch_idx):
    del batch_idx
    losses = self._loss(batch)
    # self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    self._train_epoch_losses.append(losses.loss) # Accumulate loss
    return losses.loss
  
  def on_train_epoch_end(self):
      # Calculate average loss
      if self._train_epoch_losses:
        epoch_average_loss = torch.stack(self._train_epoch_losses).mean()
      else:
        epoch_average_loss = torch.tensor(0.0, device=self.device)
      # Log the average loss
      self.log(name='trainer/loss_epoch',
               value=epoch_average_loss,
               on_step=False,
               on_epoch=True,
               sync_dist=True, # Important for distributed training
               prog_bar=True) # Optional: don't show in progress bar
      # Clear the accumulator for the next epoch
      self._train_epoch_losses.clear()
      current_epoch = self.current_epoch
      if (current_epoch + 1) % 50 == 0: # 例如每 50 个 epoch 保存一次
        save_path = f"manual_checkpoints/epoch_{current_epoch+1}_manual.ckpt"
        print(f"Manually saving checkpoint at epoch {current_epoch+1} to {save_path}")
        self.trainer.save_checkpoint(save_path)
        print(f"Checkpoint saved to {save_path}")
      
  def on_validation_epoch_start(self):
    # self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise_schedule.eval()
    self.rgcn.eval()
    self.ent_init.eval()
    self._valid_graphs= []
    # self.sampling_eps = self.config.train.sampling_eps
    
  def validation_step(self, batch, batch_idx):
    clean_graphs= self.infer_batch(batch)
    if isinstance(clean_graphs[0], dgl.DGLGraph):
      for i in clean_graphs:
        self._valid_graphs.append(i.to('cpu'))
    else:
      for i in clean_graphs:
        self._valid_graphs.append((i[0],i[1]))
    
    return 
  
  def on_validation_epoch_end(self):
    # self.best_f1,self.best_recall,self.best_precision,self.best_threshold = 0,0,0,0
    best_f1,best_recall,best_precision,best_threshold,best_num_predict = 0,0,0,0,0
    if isinstance(self._valid_graphs[0], dgl.DGLGraph):
      self._valid_graphs=[(self._valid_graphs,0.98)]
    for _valid,_ther in self._valid_graphs:
      # _valid = self._valid_graphs
      # _ther = 0.98
      pred_triples=[]
      for itm in _valid:
        if itm.num_edges() == 0:
          continue
        src, dst = itm.edges()
        rel = itm.edata['rel']
        orig_id = itm.ndata['orig_id']
        pred_triples.append(torch.stack([orig_id[src],rel,orig_id[dst]], dim=1))
      # 和self.test_triples对比,计算precision recall f1
      if self.test_triples is not None:
        if len(pred_triples) == 0:
          return
        pred_triples = torch.cat(pred_triples, dim=0)
        pred_triples = pred_triples.numpy()
        test_triples = self.test_triples.numpy()
        # np.save('pred_triples.npy', pred_triples)
        # np.save('test_triples.npy', test_triples)
        # np.save('seen_triples.npy', self.seentriples.numpy())
        precision, recall, f1,num_predict = calculate_metrics(pred_triples, test_triples,self.seentriples.numpy())
        # if f1 > self.best_f1:
        #   self.best_f1 = f1
        #   self.best_precision = precision
        #   self.best_recall = recall
        #   self.best_threshold = _ther
        if f1 > best_f1:
          best_f1 = f1
          best_precision = precision
          best_recall = recall
          best_threshold = _ther
          best_num_predict = num_predict
      self.log(name='valid/p',
               value=best_precision,
               on_step=False,
               on_epoch=True,
               sync_dist=True,
               prog_bar=True)
      self.log(name='valid/r',
               value=best_recall,
               on_step=False,
               on_epoch=True,
               sync_dist=True,
               prog_bar=True)
      self.log(name='valid/f1',
               value=best_f1,
               on_step=False,
               on_epoch=True,
               sync_dist=True,
               prog_bar=True)
      self.log(name='valid/th',
                value=best_threshold,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True)
      self.log(name='valid/num_predict',
          value=best_num_predict,
          on_step=False,
          on_epoch=True,
          sync_dist=True,
          prog_bar=True)
    return 
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
     
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.train.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      


  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
      # betas=(self.config.optim.beta1,
      #        self.config.optim.beta2),
      # eps=self.config.optim.eps,
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.train.lr,
      weight_decay=self.config.train.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    assert self.train
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    # ... (之前的代码) ...
   
  def q_xt(
      self, x, p, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    
    assert isinstance(x[0], dgl.DGLGraph)
    noise_graph=[]
    batch_size=len(x)
    device=x[0].device
    for itm in range(batch_size):
      #待改进 均匀mask可能不太好，或者只mask query graph中的三元组
      if self.first_inverse:
        num_edges = x[itm].num_edges()
        move_indices = torch.rand(num_edges//2, device=device) <= p[itm]
        move_indices = torch.cat([move_indices,move_indices],dim=0)
        # g = x[itm]
        # rel = g.edata['rel']
        # src, dst = g.edges()
        # num_edges = g.num_edges()
        # # 只对正向边(rel < num_rel)做mask
        # is_forward = rel < self.num_rel
        # move_indices = torch.zeros(num_edges, dtype=torch.bool, device=device)
        # # 随机mask正向边
        # rand_mask = torch.rand(num_edges, device=device) <= p[itm]
        # move_indices[is_forward] = rand_mask[is_forward]
        # # 找到被mask的正向边的(src, dst, rel)
        # masked_src = src[move_indices]
        # masked_dst = dst[move_indices]
        # masked_rel = rel[move_indices]
        # # 找到对应的inverse边 (dst, src, rel+num_rel)
        # inv_rel = masked_rel + num_rel
        # # 标记inverse边
        # for s, d, r in zip(masked_dst, masked_src, inv_rel):
        #     inv_mask = (src == s) & (dst == d) & (rel == r)
        #     move_indices = move_indices | inv_mask
      else:
        move_indices = torch.rand(x[itm].num_edges(), device=device) <= p[itm]
      # noise_itm=dgl.edge_subgraph(x[itm], ~move_indices, preserve_nodes=True)
      noise_itm = dgl.edge_subgraph(x[itm], ~move_indices, relabel_nodes=False)
      missing_nodes = torch.arange(x[itm].num_nodes(), device=x[itm].device)[~torch.isin(torch.arange(x[itm].num_nodes(), device=x[itm].device), noise_itm.nodes())]
      noise_itm.add_nodes(len(missing_nodes))
      noise_itm.ndata['orig_id'] = x[itm].ndata['orig_id']
      noise_graph.append(noise_itm)

    if sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    
    return noise_graph
  
 
  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      p_x0 = self.forward(x,
                        sigma_t,
                        sample_mode=True).to(torch.float64)
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0[:, -self.block_size:])
      # randomly and uniformly select an index in the block (among masked tokens)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs[:, -self.block_size:])  
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # compute kv cache if all tokens in a block are sampled
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      _ = self.forward(x_new, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      tokens_sampled = (x_new - x) != 0
      return None, x_new
    else:
      return p_x0, x_new

  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    samples = []
    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(batch_size_per_gpu)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.gen_nfes.append(self.config.model.length)
      samples = torch.cat(samples, dim=0) 
      return self.tokenizer.batch_decode(samples)
    if self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=(seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    samples = torch.cat(samples, dim=0) 
    return self.tokenizer.batch_decode(samples)

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max):
     #保证每个batch能见到0-1平均的t
    if self.unit_sample:
      n_half = batch_dims // 2
      _eps_b1 = torch.rand(n_half, device=device)
      _eps_b2 = 1 - _eps_b1
      _eps_b = torch.cat((_eps_b1, _eps_b2), dim=0)
      # 如果 batch_dims 是奇数，额外添加一个随机数
      if batch_dims % 2 == 1:
          _eps_b_extra = torch.rand(1, device=device)
          _eps_b = torch.cat((_eps_b, _eps_b_extra), dim=0)
    else:
      _eps_b = torch.rand((batch_dims), device=device)
      
    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims , device=device) / (batch_dims)
      offset_b = offset_b.view(batch_dims)
      _eps_b = (_eps_b / (batch_dims ) + offset_b) % 1
    t = _eps_b

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def batch_graph_to_tensor(self, inputgraph):
        """
        inputgraph: list of DGLGraph
        Returns:
            x: [bs, max_node, 1]
            e: [bs, max_node, max_node, 1]
        """
        
        edge_feat_key='rel'
        nodemax = self.max_n_nodes
        
        bs = len(inputgraph)
        device = inputgraph[0].edata[edge_feat_key].device
        

        x_list = []
        # Collect edge  with padding, 默认都=self.num_rel*2+1
         # Define an index for "no edge" or padding
        no_edge_index = self.num_rel * 2
        # Initialize with the "no edge" index, shape is (bs, nodemax, nodemax)
        e = torch.full((bs, nodemax, nodemax), no_edge_index, dtype=torch.long, device=device)

        
        for i,g in enumerate(inputgraph):
            n = g.num_nodes()
            if n < nodemax:
                pad = torch.zeros(nodemax - n, 1, device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
            x_list.append(feat)
            
            src, dst = g.edges()
            rel = g.edata[edge_feat_key]
            e[i, src, dst] = rel
            
        x = torch.stack(x_list, dim=0)  # [bs, nodemax, node_feat_dim]
        
        # nodemask: [bs, nodemax]，True表示有效节点，False表示padding,device与x一致
        nodemask = (x.abs().sum(dim=-1) != 0)
   
        return x, e, nodemask
 
  def _build_pair_index_cache(self, graphs):
    """Pre-compute mapping from global entity pairs to (batch, i, j) indices.

    graphs: list of DGLGraph; each graph must have ndata['orig_id']
    Returns: dict[(h_id, t_id)] -> list of (b, i, j)
    """
    pair_index = {}
    for b, g in enumerate(graphs):
      if 'orig_id' not in g.ndata:
        continue
      orig_ids = g.ndata['orig_id']
      num_nodes = g.num_nodes()
      for i in range(num_nodes):
        h_id = int(orig_ids[i].item())
        for j in range(num_nodes):
          if i == j:
            continue
          t_id = int(orig_ids[j].item())
          key = (h_id, t_id)
          if key not in pair_index:
            pair_index[key] = [(b, i, j)]
          else:
            pair_index[key].append((b, i, j))
    return pair_index

  def _avg_pair_probs_across_subgraphs(self, logits_E, pair_index):
    """Average relation probabilities for cached repeated global entity pairs.

    logits_E: Tensor of shape (B, N, N, C)
    pair_index: dict[(h_id, t_id)] -> list of (b, i, j)
    """
    if not pair_index:
      return logits_E


    new_probs = logits_E.clone()
    for _, index_list in pair_index.items():
      if len(index_list) <= 1:
        continue
      # Stack all probability vectors for this global pair
      stacked = torch.stack([logits_E[b, i, j] for (b, i, j) in index_list], dim=0)
      mean_vec = stacked.mean(dim=0)
      # Write averaged probs back to all occurrences
      for (b, i, j) in index_list:
        new_probs[b, i, j] = mean_vec

    return new_probs

  @torch.no_grad()
  def sample_batch(self, batch_id, batch_size, y, keep_chain, number_chain_steps, save_final, num_nodes=None):
      """
      :param batch_id: int
      :param batch_size: int
      :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
      :param save_final: int: number of predictions to save to file
      :param keep_chain: int: number of chains to save to file (disabled)
      :param keep_chain_steps: number of timesteps to save for each chain (disabled)
      :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
      """
      
      arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
      node_mask = arange < n_nodes.unsqueeze(1)
      
      z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
      X, E = z_T.X, z_T.E

      assert (E == torch.transpose(E, 1, 2)).all()

      # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
      for s_int in reversed(range(0, self.T)):
          s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
          t_array = s_array + 1
          s_norm = s_array / self.T
          t_norm = t_array / self.T

          # Sample z_s
          sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
          X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

      # Sample
      sampled_s = sampled_s.mask(node_mask, collapse=True)
      X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
      
      molecule_list = []
      for i in range(batch_size):
          n = n_nodes[i]
          atom_types = X[i, :n].cpu()
          edge_types = E[i, :n, :n].cpu()
          molecule_list.append([atom_types, edge_types])
      
      return molecule_list
  

  def sample_p_zs_given_zt(self, s, t, noise_graph,support):
      """Samples from zs ~ p(zs | zt). Only used during sampling.
          if last_step, return the graph prediction as well"""
      if not noise_graph:
        pass
      bs=len(support)
      
      loss_scale, beta_t  = self.noise_schedule(t)
      loss_scale, beta_s  = self.noise_schedule(s)
      
      # alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
      # alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

      
      def get_prob(noisy_data, condition=True):
          pred = self.forward(noisy_data, condition=condition)

          # Normalize predictions
          pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
          pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

          

          Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
          p_s_and_t_given_0 = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=Xt_all,
                                                                                          Qt=Qt.X,
                                                                                          Qsb=Qsb.X,
                                                                                          Qtb=Qtb.X)
          predX_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)
          weightedX_all = predX_all.unsqueeze(-1) * p_s_and_t_given_0
          unnormalized_probX_all = weightedX_all.sum(dim=2)                     # bs, n, d_t-1

          unnormalized_prob_X = unnormalized_probX_all[:, :, :self.Xdim_output]
          unnormalized_prob_E = unnormalized_probX_all[:, :, self.Xdim_output:].reshape(bs, n*n, -1)

          unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
          unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5

          prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1
          prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)  # bs, n, d_t-1
          prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

          return prob_X, prob_E

      prob_X, prob_E = get_prob(noisy_data)

      ### Guidance
      if self.guidance_target is not None and self.guide_scale is not None and self.guide_scale != 1:
          uncon_prob_X, uncon_prob_E = get_prob(noisy_data, unconditioned=True)
          prob_X = uncon_prob_X *  (prob_X / uncon_prob_X.clamp_min(1e-10)) ** self.guide_scale  
          prob_E = uncon_prob_E * (prob_E / uncon_prob_E.clamp_min(1e-10)) ** self.guide_scale  
          prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-10)
          prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-10)

      assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
      assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

      sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask, step=s[0,0].item())

      X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
      E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

      assert (E_s == torch.transpose(E_s, 1, 2)).all()
      assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

      out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)
      out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)

      return out_discrete
  

  def check_graph_inverse(self,g):
    # --- Check and add inverse edges ---
    src_nodes, dst_nodes = g.edges()
    if src_nodes.numel() > 0: # Only proceed if there are edges
        rel_types = g.edata['rel']
        device = rel_types.device
        # Create a set of existing triples for quick lookup
        existing_triples = set(map(tuple, torch.stack([src_nodes, dst_nodes, rel_types], dim=1).cpu().numpy()))
        
        new_src, new_dst, new_rel = [], [], []
        
        for s, d, r_int in zip(src_nodes.tolist(), dst_nodes.tolist(), rel_types.tolist()):
            inv_s, inv_d = d, s
            if r_int < self.num_rel:
                inv_r = r_int + self.num_rel
            else:
                inv_r = r_int - self.num_rel
                
            # Check if the inverse triple exists
            if (inv_s, inv_d, inv_r) not in existing_triples:
                new_src.append(inv_s)
                new_dst.append(inv_d)
                new_rel.append(inv_r)
                # Add the newly added inverse to the set to avoid duplicates if the original list had inverses already
                existing_triples.add((inv_s, inv_d, inv_r))

        # Add the missing inverse edges
        if new_src:
            g.add_edges(
                torch.tensor(new_src, dtype=torch.long, device=device),
                torch.tensor(new_dst, dtype=torch.long, device=device),
                {'rel': torch.tensor(new_rel, dtype=torch.long, device=device)}
            )
    return g
  def build_graphs_from_predE(self,pred_E,num_nodes_per_graph,support):
      """
      根据多标签预测张量 pred_E 构建对应的 DGLGraph 列表。
      参数：
        pred_E: (B, N, N, DIM)，对于每个样本 (b)，在 [N, N] 的节点对上存在 DIM 个标签。
                若 pred_E[b, i, j, k] != 0 表示存在标签 k 的边 (i->j)。
      返回：
        graph_list: 包含 B 个 dgl.DGLGraph 的列表
      """
      device = pred_E.device
      B, nn, _, DIM = pred_E.shape
      graph_list = []
      # 假设 pred_E 的形状为 (B, N, N, DIM)
      no_edge_index = DIM - 1

      # 对于 batch 中的每个样本，逐一构建图
      for b in range(B):
          # 根据预测结果获取该样本所有 (i, j, k) 满足 pred_E[b, i, j, k] != 0
          # 这里假设 > 0 表示存在该边标签
          # （若是 logits，需先做阈值判断或 argmax 等）
          num_nodes=support[b].num_nodes()
          pred_slice = pred_E[b][:num_nodes,:num_nodes] # (num_node,num_node, DIM)

          # 仅选择非空边类别的切片 pred_E[..., :no_edge_index]
          # row_idx, col_idx, label_idx = torch.nonzero(pred_slice[..., :no_edge_index] != 0, as_tuple=True)
          # 生成掩码并转换为稀疏张量
          pred_mask = pred_slice[..., :no_edge_index] != 0
          sparse_tensor = pred_mask.to_sparse()
          # 直接获取稀疏张量的索引
          row_idx, col_idx, label_idx = sparse_tensor.indices()
          # 去掉在支持图中存在的边
          # support_row_idx, support_col_idx = support[b].edges()
          # if support_row_idx.numel() > 0: # If there are support edges
          #     pred_triples = torch.stack([row_idx, col_idx, label_idx], dim=1) # Shape: (num_pred, 3)
          #     support_rel = support[b].edata['rel']
          #     support_triples = torch.stack([support_row_idx, support_col_idx, support_rel], dim=1) # Shape: (num_support, 3)

          #     # Check which predicted triples exist in support triples using broadcasting
          #     # Expand dims for comparison: pred -> (num_pred, 1, 3), support -> (1, num_support, 3)
          #     # matches will be (num_pred, num_support), True if pred_triple[i] == support_triple[j]
          #     matches = (pred_triples.unsqueeze(1) == support_triples.unsqueeze(0)).all(dim=2)

          #     # is_in_support will be (num_pred,), True if pred_triple[i] is found anywhere in support_triples
          #     is_in_support = matches.any(dim=1)

          #     # Keep only predicted triples that are NOT in the support graph
          #     mask = ~is_in_support
          #     if is_in_support.sum() > 1:
          #       assert False, "All predicted edges are not in the support graph, no new edges to add."

          #     new_row_idx = row_idx[mask]
          #     new_col_idx = col_idx[mask]
          #     new_label_idx = label_idx[mask]
          # else:
          #     # No support edges, so all predicted edges are new
          #     new_row_idx, new_col_idx, new_label_idx = row_idx, col_idx, label_idx
          new_row_idx, new_col_idx, new_label_idx = row_idx, col_idx, label_idx

          # 创建空图初始化
          g = dgl.DGLGraph().to(device)
          g.add_nodes(num_nodes)  # 添加节点
          if row_idx.numel() > 0: # Check if there are any edges
            g.add_edges(new_row_idx, new_col_idx)
            g.edata['rel'] = new_label_idx.to(torch.long) # Assign labels directly
          
          if 'orig_id' in support[b].ndata:
              g.ndata['orig_id'] = support[b].ndata['orig_id']
          # check一下graph是否每条边都有对应的逆边存在，如果无则添加
          self.check_graph_inverse(g)
          graph_list.append(g)
  
      return graph_list
  def tbuild_graphs_from_predE(self,pred_E,num_nodes_per_graph,support):
      """
      根据多标签预测张量 pred_E 构建对应的 DGLGraph 列表。
      参数：
        pred_E: (B, N, N, DIM)，对于每个样本 (b)，在 [N, N] 的节点对上存在 DIM 个标签。
                若 pred_E[b, i, j, k] != 0 表示存在标签 k 的边 (i->j)。
      返回：
        graph_list: 包含 B 个 dgl.DGLGraph 的列表
      """
      device = pred_E.device
      B, nn, _, DIM = pred_E.shape
      graph_list = []

      for b in range(B):
          # 根据预测结果获取该样本所有 (i, j, k) 满足 pred_E[b, i, j, k] != 0
          # 这里假设 > 0 表示存在该边标签
          # （若是 logits，需先做阈值判断或 argmax 等）
          num_nodes=support[b].num_nodes()
          pred_slice = pred_E[b][:num_nodes,:num_nodes] # (num_node,num_node, DIM)

          # 仅选择非空边类别的切片 pred_E[..., :no_edge_index]
          # row_idx, col_idx, label_idx = torch.nonzero(pred_slice[..., :no_edge_index] != 0, as_tuple=True)
          # 生成掩码并转换为稀疏张量
          pred_mask = pred_slice[...] != 0
          sparse_tensor = pred_mask.to_sparse()
          # 直接获取稀疏张量的索引
          row_idx, col_idx, label_idx = sparse_tensor.indices()
          new_row_idx, new_col_idx, new_label_idx = row_idx, col_idx, label_idx

          # 创建空图初始化
          g = dgl.DGLGraph().to(device)
          g.add_nodes(num_nodes)  # 添加节点
          if row_idx.numel() > 0: # Check if there are any edges
            g.add_edges(new_row_idx, new_col_idx)
            g.edata['rel'] = new_label_idx.to(torch.long) # Assign labels directly
          
          if 'orig_id' in support[b].ndata:
              g.ndata['orig_id'] = support[b].ndata['orig_id']
          # check一下graph是否每条边都有对应的逆边存在，如果无则添加
          self.check_graph_inverse(g)
          graph_list.append(g)
  
      return graph_list
    
  def generate_graphs_from_prediction(self,pred_E, support_graphs):
      """
      将 support_graphs 中的每条边 (src, dst) 所对应的多标签索引 (rel) 在 pred_E 中置为 0。
      pred_E: (B, N, N, DIM)，多标签预测张量
      support_graphs: 包含 B 个支持图的列表，每个图用 edata['rel'] 存储标签索引
      返回：置 0 后的 pred_E
      """
      
      if isinstance(pred_E, list):
        generated_graphs = []
        for i in range(len(pred_E)):
          generated_graphs.append((self.build_graphs_from_predE(pred_E[i][0], None, support_graphs),pred_E[i][1])) 
      else:
          generated_graphs = self.build_graphs_from_predE(pred_E, None, support_graphs)
        
      return generated_graphs

  @torch.no_grad()
  def infer_batch(self, batch):
    support=batch['support']
    # target_graph=batch['graph']
    device = batch['support'][0].device
    batch_size = batch['support'].__len__()
    t = torch.linspace(1,1e-3, steps=self.T, device=device).unsqueeze(0).repeat(batch_size, 1)
    # t = torch.tensor(0.9,device=device).repeat(batch_size, 10)
    result=[]
    # Pre-compute cache of repeated global entity pairs for this batch
    # pair_index_cache = self._build_pair_index_cache(support)
    for step in tqdm(range(self.T)):
      input_graph = []
      noise_graph=[]   
      if step == self.T-1:
        break
      current_t = t[:,step]
      next_t = t[:,step+1]
      # sample noise
      _, move_chance_t = self.noise_schedule(current_t)
      _, move_chance_s = self.noise_schedule(next_t)
      sigma = current_t
      move_chance_t = move_chance_t[:, None]
      move_chance_s = move_chance_s[:, None]
      mask_prob = move_chance_s / move_chance_t
      if noise_graph:
        exit(-1)
        for i in range(batch_size):
          input_graph.append(merge_two_graphs(noise_graph[i], support[i]))
      else:
        input_graph = support
        # input_graph = self.q_xt(support,move_chance_s)
      logits_E= self.forward(input_graph=input_graph, sigma=sigma)
      # On the final denoising step, enforce cross-subgraph consistency
      # by averaging probabilities for repeated global entity pairs.
      # if step >= self.T//2:
      #   logits_E = self._avg_pair_probs_across_subgraphs(logits_E, pair_index_cache)
      # for idx in range(batch_size):
      #   logit=logits_E[idx,:target_graph[idx].num_nodes(),:target_graph[idx].num_nodes()]
      #   src,drc=target_graph[idx].edges()
      #   rel=target_graph[idx].edata['rel']
      #   src,drc,rel,logit = np.array(src.cpu()), np.array(drc.cpu()), np.array(rel.cpu()), np.array(logit.cpu())
      #   print(evaluate_link_prediction(logit, src, drc, rel, 2))
      # if noise_graph:
      #   logits_uncond = self.forward(input_graph=input_graph, sigma=sigma)
      # 统计一下哪些不是0
      # mask_logits = (logits_E != 0.).any(dim=-1)
      # logits_E = torch.softmax(logits_E, dim=-1)  # bs, n, n, d0
      # Sample noise_graph
      # q_xs = logits_E * (1 - mask_prob).view(logits_E.shape[0], 1, 1, 1)
      # q_xs[:, :,:, self.num_rel*2] = mask_prob.squeeze(-1)[:, None, None]
      
      pred_E = _sample_categorical(logits_E,threshold=0.99,mask_prob=mask_prob,support=support,step=step) 
      
      noise_graph = self.generate_graphs_from_prediction(pred_E, support)
      #更新support
      # if step==0:
      #   torch.save(support, f'origin.pt')  
      if isinstance(pred_E, list):
        pass
      else:
        for i in range(batch_size):
          # 获取 noise_graph 的边和关系
          src, dst = noise_graph[i].edges()
          if src.numel() == 0:
            continue
          rel = noise_graph[i].edata['rel']
          # 将 noise_graph 的边和关系添加到 support_graph
          support[i].add_edges(src, dst, {'rel': rel})
          if result:
            result[i].add_edges(src, dst, {'rel': rel})
      if not result:
        result=noise_graph    
      ################# 用于可视化 ################  
      
      # if step%4 == 0:
      #   # 每隔4步保存一下result
      #   if isinstance(result, list):
      #       # 如果result是图列表，可以保存图的基本信息
      #       print(f"Step {step}: Saving {len(result)} graphs")
      #       # 可以选择保存到文件或进行其他处理
      #       torch.save(result, f'result_step_{step}.pt')
      #   else:
      #       print(f"result must be list")
      #       # torch.save(result, f'result_step_{step}.pt')
      # elif step==self.T-2:
      #     torch.save(result, f'final_{step}.pt')
      # ################# 用于可视化 ################  
        
      # noise_graph = self.sample_p_zs_given_zt(current_t, next_t, noise_graph, support)
      
    return result 
    
    
  def cross_loss(self, model_output, graph, support,no_edge_weight=0.01):       
      '''
      仅恢复无边位置的预测, ground truth是graph，support也在ground truth中不过在计算loss的时候去掉
      Use Binary Cross-Entropy for multi-label edge prediction
      model_output # Logits shape: (B, N, N, DIM)
      graph # List of ground truth graphs
      support # List of support graphs
      node_mask # Shape: (B, N), True for valid nodes
      '''
      B, N, _, DIM = model_output.shape
      device = model_output.device
      
      target_graph = graph
      # target_graph = []
      # for itm in range(B):
      #   target_graph.append(merge_two_graphs(graph[itm], support[itm],needunique=False))

      # DIM should be self.num_rel * 2 + 1 (forward, inverse, no_edge)
      num_classes = self.num_rel * 2 + 1
      assert DIM == num_classes, f"Output dimension {DIM} doesn't match num_classes {num_classes}"
      no_edge_index = num_classes-1 # Index for no edge 

      # 1. Create Binary Target Tensor
      target_bce = F.one_hot(torch.full((B, N, N), no_edge_index, dtype=torch.long, device=device), 
                             num_classes=num_classes).float()

      # Populate target using batched ground truth graph (including inverses)
      # Create a temporary list with inverse edges added to ground truth
      temp_graph_with_inverse = target_graph
      batch_g_truth_inv = dgl.batch(temp_graph_with_inverse).to(device)
      b_src_inv, b_dst_inv = batch_g_truth_inv.edges()
      b_rel_inv = batch_g_truth_inv.edata['rel'] # These now include original + inverse rel indices
      
      # Map batched indices back to B, N, N space
      num_edges_per_graph_inv = batch_g_truth_inv.batch_num_edges()
      graph_indices_inv = torch.arange(len(graph), device=device).repeat_interleave(num_edges_per_graph_inv)
      num_nodes_per_graph_inv = batch_g_truth_inv.batch_num_nodes() # Should be same as original
      node_offsets_inv = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes_per_graph_inv[:-1]]), dim=0)
      orig_src_inv = b_src_inv - node_offsets_inv[graph_indices_inv]
      orig_dst_inv = b_dst_inv - node_offsets_inv[graph_indices_inv]

      # Use advanced indexing to set 1.0 for true edges (forward and inverse)
      target_bce[graph_indices_inv, orig_src_inv, orig_dst_inv, b_rel_inv] = 1.0
      target_bce[graph_indices_inv, orig_src_inv, orig_dst_inv, no_edge_index] = 0.0
      

      # 2. Create Effective Mask (B, N, N)
      # Mask for valid node pairs
      # Mask for valid node pairs (vectorized)
      row_indices = torch.arange(N, device=device).view(1, N, 1)
      col_indices = torch.arange(N, device=device).view(1, 1, N)
      num_nodes_b = num_nodes_per_graph_inv.view(B, 1, 1)
      valid_node_pair_mask = (row_indices < num_nodes_b) & (col_indices < num_nodes_b) # Shape (B, N, N)
      
      # Mask for self-loops
      diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
      # Pair-level mask: valid nodes and not self-loop (B, N, N)
      pair_mask = valid_node_pair_mask & ~diag_mask


      # 3. Calculate Pos Weight (Optional but Recommended)
      # Consider only elements covered by the pair_mask
      num_valid_pairs = pair_mask.sum()
      if num_valid_pairs > 0:
          # Exclude support triples from positive/negative counts for weighting
          target_bce_masked_pairs = target_bce[pair_mask] # Shape: (num_valid_pairs, DIM)

          num_positives = (target_bce_masked_pairs).sum()
          num_negatives = ((1.0 - target_bce_masked_pairs)).sum()

          if num_positives > 0:
              pos_weight_value = num_negatives / num_positives
          else:
              pos_weight_value = 1.0 # Avoid division by zero
          pos_weight_tensor = torch.full_like(model_output, pos_weight_value)
          pos_weight_tensor[..., no_edge_index]=no_edge_weight
      else:
          pos_weight_tensor = torch.ones_like(model_output) # Default if no valid pairs

      # 4. Calculate BCE Loss (Element-wise)
      loss_elementwise = F.binary_cross_entropy_with_logits(
          model_output,
          target_bce,
          reduction='none',
          pos_weight=pos_weight_tensor # Apply positive weight
      )
# for idx in range(B):
#   input_graph = target_graph
#   logit=model_output[idx,:input_graph[idx].num_nodes(),:input_graph[idx].num_nodes()]
#   src,drc=input_graph[idx].edges()
#   rel=input_graph[idx].edata['rel']
#   src,drc,rel,logit = np.array(src.detach().cpu()), np.array(drc.detach().cpu()), np.array(rel.detach().cpu()), np.array(logit.detach().cpu())
#   print(evaluate_link_prediction(logit, src, drc, rel, 2))
      # 5. Apply Masks and Reduce

      # Apply pair-level mask (valid nodes, no self-loops) and sum over relation dimension
      masked_loss_per_pair = (loss_elementwise * pair_mask.unsqueeze(-1)).sum(dim=-1) 
      # masked_loss_per_pair = (loss_elementwise[pair_mask]).sum(dim=-1)
      
      if num_valid_pairs > 0:
          # Average the summed loss over the number of valid pairs
          loss = masked_loss_per_pair.sum() / num_valid_pairs
      else:
          # Handle case with no valid pairs to avoid division by zero
          loss = torch.tensor(0.0, device=device, requires_grad=True)
      return loss
 
  def bce_loss(self, model_output, graph, support,no_edge_weight=0.005):       
      '''
      Use Binary Cross-Entropy for multi-label edge prediction
      model_output # Logits shape: (B, N, N, DIM)
      graph # List of ground truth graphs
      support # List of support graphs
      '''
      B, N, _, DIM = model_output.shape
      device = model_output.device
      # DIM should be self.num_rel * 2 + 1 (forward, inverse, no_edge)
      num_classes = self.num_rel * 2 + 1
      
      assert DIM == num_classes, f"Output dimension {DIM} doesn't match num_classes {num_classes}"
      no_edge_index = num_classes-1 # Index for no edge (padding)
      
      target_bce = F.one_hot(torch.full((B, N, N), no_edge_index, dtype=torch.long, device=device), 
                             num_classes=num_classes).float()
      
      temp_graph_with_inverse = graph
      batch_g_truth_inv = dgl.batch(temp_graph_with_inverse).to(device)
      b_src_inv, b_dst_inv = batch_g_truth_inv.edges()
      b_rel_inv = batch_g_truth_inv.edata['rel'] # These now include original + inverse rel indices

      # Map batched indices back to B, N, N space
      num_edges_per_graph_inv = batch_g_truth_inv.batch_num_edges()
      graph_indices_inv = torch.arange(len(graph), device=device).repeat_interleave(num_edges_per_graph_inv)
      num_nodes_per_graph_inv = batch_g_truth_inv.batch_num_nodes() # Should be same as original
      node_offsets_inv = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes_per_graph_inv[:-1]]), dim=0)
      orig_src_inv = b_src_inv - node_offsets_inv[graph_indices_inv]
      orig_dst_inv = b_dst_inv - node_offsets_inv[graph_indices_inv]

      # Use advanced indexing to set 1.0 for true edges (forward and inverse)
      target_bce[graph_indices_inv, orig_src_inv, orig_dst_inv, b_rel_inv] = 1.0
      target_bce[graph_indices_inv, orig_src_inv, orig_dst_inv, no_edge_index] = 0.0

      # 2. Create Effective Mask (B, N, N)
      # Mask for valid node pairs
      row_indices = torch.arange(N, device=device).view(1, N, 1)
      col_indices = torch.arange(N, device=device).view(1, 1, N)
      num_nodes_b = num_nodes_per_graph_inv.view(B, 1, 1)
      valid_node_pair_mask = (row_indices < num_nodes_b) & (col_indices < num_nodes_b) # Shape (B, N, N)
      diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
      # Pair-level mask: valid nodes and not self-loop (B, N, N)
      pair_mask = valid_node_pair_mask & ~diag_mask
      
      support_triple_mask = torch.zeros_like(model_output, dtype=torch.bool)
      temp_support_with_inverse = support
      batch_support_inv = dgl.batch(temp_support_with_inverse).to(device)
      if batch_support_inv.num_edges() > 0:
          bs_src_inv, bs_dst_inv = batch_support_inv.edges()
          bs_rel_inv = batch_support_inv.edata['rel'] # Get relation types from support
          # Map support indices back
          num_edges_per_support_inv = batch_support_inv.batch_num_edges()
          support_indices_inv = torch.arange(len(support), device=device).repeat_interleave(num_edges_per_support_inv)
          num_nodes_per_support_inv = batch_support_inv.batch_num_nodes()
          support_node_offsets_inv = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes_per_support_inv[:-1]]), dim=0)
          orig_s_src_inv = bs_src_inv - support_node_offsets_inv[support_indices_inv]
          orig_s_dst_inv = bs_dst_inv - support_node_offsets_inv[support_indices_inv]
          # Mark True only for the specific triples present in support
          support_triple_mask[support_indices_inv, orig_s_src_inv, orig_s_dst_inv, bs_rel_inv] = True

      # 3. Calculate Pos Weight (Optional but Recommended)
      # Consider only elements covered by the pair_mask
      num_valid_pairs = pair_mask.sum()
      if num_valid_pairs > 0:
        
          # Exclude support triples from positive/negative counts for weighting
          target_bce_masked_pairs = target_bce[pair_mask] # Shape: (num_valid_pairs, DIM)
          support_triple_masked_pairs = support_triple_mask[pair_mask] # Shape: (num_valid_pairs, DIM)
          # Calculate positives and negatives for all classes vectorized
          valid_elements_mask = ~support_triple_masked_pairs # Shape: (num_valid_pairs, DIM)
          
          # Sum over the num_valid_pairs dimension (dim=0)
          positives_per_class = (target_bce_masked_pairs * valid_elements_mask).sum(dim=0) # Shape: (DIM,)
          negatives_per_class = ((1.0 - target_bce_masked_pairs) * valid_elements_mask).sum(dim=0) # Shape: (DIM,)

          # Calculate pos_weight per class, handling division by zero
          pos_weight_values = torch.ones_like(positives_per_class) # Default to 1.0
          valid_pos_mask = positives_per_class > 0
          pos_weight_values[valid_pos_mask] = negatives_per_class[valid_pos_mask] / positives_per_class[valid_pos_mask]

          # Broadcast the per-class weights to the full tensor shape
          # Reshape pos_weight_values to (1, 1, 1, DIM) for broadcasting
          pos_weight_tensor = pos_weight_values.view(1, 1, 1, -1).expand_as(model_output)

          
          # 例如对空边 no_edge_index，单独调整权重
          pos_weight_tensor[..., no_edge_index] = no_edge_weight
      else:
          pos_weight_tensor = torch.ones_like(model_output) # Default if no valid pairs
          # pos_weight_value = min(num_negatives / max(num_positives, 1), some_clamp_value)
      # 4. Calculate BCE Loss (Element-wise)
      loss_elementwise = F.binary_cross_entropy_with_logits(
          model_output,
          target_bce,
          reduction='none',
          pos_weight=pos_weight_tensor # Apply positive weight
      )

      # 5. Apply Masks and Reduce
      # Mask out specific support triples *before* summing over relations
      masked_loss_elementwise = loss_elementwise * ~support_triple_mask

      # Apply pair-level mask (valid nodes, no self-loops) and sum over relation dimension
      masked_loss_per_pair = (masked_loss_elementwise * pair_mask.unsqueeze(-1)).sum(dim=-1) # Shape (B, N, N)

      if num_valid_pairs > 0:
          # Average the summed loss over the number of valid pairs
          loss = masked_loss_per_pair.sum() / num_valid_pairs
      else:
          # Handle case with no valid pairs to avoid division by zero
          loss = torch.tensor(0.0, device=device, requires_grad=True)
      return loss
    
  def tbce_loss(self, model_output, graph, support,no_edge_weight=0.005):       
      '''
      Use Binary Cross-Entropy for multi-label edge prediction
      model_output # Logits shape: (B, N, N, DIM)
      graph # List of ground truth graphs
      support # List of support graphs
      '''
      B, N, _, DIM = model_output.shape
      device = model_output.device
      num_classes = self.num_rel * 2
      
      assert DIM == num_classes, f"Output dimension {DIM} doesn't match num_classes {num_classes}"
      
      target_bce = torch.zeros(B, N, N,DIM).float().to(device)
      
      temp_graph_with_inverse = graph
      batch_g_truth_inv = dgl.batch(temp_graph_with_inverse).to(device)
      b_src_inv, b_dst_inv = batch_g_truth_inv.edges()
      b_rel_inv = batch_g_truth_inv.edata['rel'] # These now include original + inverse rel indices

      # Map batched indices back to B, N, N space
      num_edges_per_graph_inv = batch_g_truth_inv.batch_num_edges()
      graph_indices_inv = torch.arange(len(graph), device=device).repeat_interleave(num_edges_per_graph_inv)
      num_nodes_per_graph_inv = batch_g_truth_inv.batch_num_nodes() # Should be same as original
      node_offsets_inv = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes_per_graph_inv[:-1]]), dim=0)
      orig_src_inv = b_src_inv - node_offsets_inv[graph_indices_inv]
      orig_dst_inv = b_dst_inv - node_offsets_inv[graph_indices_inv]

      # Use advanced indexing to set 1.0 for true edges (forward and inverse)
      target_bce[graph_indices_inv, orig_src_inv, orig_dst_inv, b_rel_inv] = 1.0

      # 2. Create Effective Mask (B, N, N)
      # Mask for valid node pairs
      row_indices = torch.arange(N, device=device).view(1, N, 1)
      col_indices = torch.arange(N, device=device).view(1, 1, N)
      num_nodes_b = num_nodes_per_graph_inv.view(B, 1, 1)
      valid_node_pair_mask = (row_indices < num_nodes_b) & (col_indices < num_nodes_b) # Shape (B, N, N)
      diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
      # Pair-level mask: valid nodes and not self-loop (B, N, N)
      pair_mask = valid_node_pair_mask & ~diag_mask
      
      support_triple_mask = torch.zeros_like(model_output, dtype=torch.bool)
      temp_support_with_inverse = support
      batch_support_inv = dgl.batch(temp_support_with_inverse).to(device)
      if batch_support_inv.num_edges() > 0:
          bs_src_inv, bs_dst_inv = batch_support_inv.edges()
          bs_rel_inv = batch_support_inv.edata['rel'] # Get relation types from support
          # Map support indices back
          num_edges_per_support_inv = batch_support_inv.batch_num_edges()
          support_indices_inv = torch.arange(len(support), device=device).repeat_interleave(num_edges_per_support_inv)
          num_nodes_per_support_inv = batch_support_inv.batch_num_nodes()
          support_node_offsets_inv = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes_per_support_inv[:-1]]), dim=0)
          orig_s_src_inv = bs_src_inv - support_node_offsets_inv[support_indices_inv]
          orig_s_dst_inv = bs_dst_inv - support_node_offsets_inv[support_indices_inv]
          # Mark True only for the specific triples present in support
          support_triple_mask[support_indices_inv, orig_s_src_inv, orig_s_dst_inv, bs_rel_inv] = True
          
      # 3. Calculate Pos Weight (Optional but Recommended)
      # Consider only elements covered by the pair_mask
      num_valid_pairs = pair_mask.sum()
      if num_valid_pairs > 0:
        
          # Exclude support triples from positive/negative counts for weighting
          target_bce_masked_pairs = target_bce[pair_mask] # Shape: (num_valid_pairs, DIM)
          support_triple_masked_pairs = support_triple_mask[pair_mask] # Shape: (num_valid_pairs, DIM)
          # Calculate positives and negatives for all classes vectorized
          valid_elements_mask = ~support_triple_masked_pairs # Shape: (num_valid_pairs, DIM)
          
          # Sum over the num_valid_pairs dimension (dim=0)
          positives_per_class = (target_bce_masked_pairs * valid_elements_mask).sum(dim=0) # Shape: (DIM,)
          negatives_per_class = ((1.0 - target_bce_masked_pairs) * valid_elements_mask).sum(dim=0) # Shape: (DIM,)

          # Calculate pos_weight per class, handling division by zero
          pos_weight_values = torch.ones_like(positives_per_class) # Default to 1.0
          valid_pos_mask = positives_per_class > 0
          pos_weight_values[valid_pos_mask] = negatives_per_class[valid_pos_mask] / positives_per_class[valid_pos_mask]

          # Broadcast the per-class weights to the full tensor shape
          # Reshape pos_weight_values to (1, 1, 1, DIM) for broadcasting
          pos_weight_tensor = pos_weight_values.view(1, 1, 1, -1).expand_as(model_output)
          
      else:
          pos_weight_tensor = torch.ones_like(model_output) # Default if no valid pairs
          # pos_weight_value = min(num_negatives / max(num_positives, 1), some_clamp_value)
      # 4. Calculate BCE Loss (Element-wise)
      loss_elementwise = F.binary_cross_entropy_with_logits(
          model_output,
          target_bce,
          reduction='none',
          pos_weight=pos_weight_tensor # Apply positive weight
      )

      # 5. Apply Masks and Reduce
      # Mask out specific support triples *before* summing over relations
      masked_loss_elementwise = loss_elementwise * ~support_triple_mask

      # Apply pair-level mask (valid nodes, no self-loops) and sum over relation dimension
      masked_loss_per_pair = (masked_loss_elementwise * pair_mask.unsqueeze(-1)).sum(dim=-1) # Shape (B, N, N)

      if num_valid_pairs > 0:
          # Average the summed loss over the number of valid pairs
          loss = masked_loss_per_pair.sum() / num_valid_pairs
      else:
          # Handle case with no valid pairs to avoid division by zero
          loss = torch.tensor(0.0, device=device, requires_grad=True)
      return loss
  def _forward_pass_diffusion(self, batch, t=None, sampling_eps_min=None, sampling_eps_max=None,condition=True):
    
    support=batch['support']
    query_graph=batch['graph']
    batch_size = support.__len__()
    
    # ground_truth_graph=[]
    # for itm in range(batch_size):
    #   ground_truth_graph.append(merge_two_graphs(query_graph[itm], support[itm], needunique=False))
    
    if t is None:
      t = self._sample_t(batch_size,
                         support[0].device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise_schedule(t)
    # sigma = torch.round(self.T*t)
    sigma = t

    """
    目前是把整个graph做mask，support作为condition,使用class-free guidance
    还有方法可以把support+mask后的query作为输入，使用uncondition，但是这个输入分布有可能和测试集不符
    目前把query作为maskgraph，support作为condition
    """
    x_input = self.q_xt(query_graph,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    input_graph=[]
    if condition:
      if self.training:
        # train for class-free guidence
        # drop = np.random.rand(batch_size) < self.drop_condition 
        for itm in range(batch_size):
          # if drop[itm]:
          #   if '_ID' in x_input[itm].edata:
          #     x_input[itm].edata.pop('_ID')
          #     input_graph.append(x_input[itm])
          # else:
            input_graph.append(merge_two_graphs(x_input[itm], support[itm],needunique=False))
      else:
        # eval for class-free guidence
        for itm in range(batch_size):
          if x_input:
            input_graph.append(merge_two_graphs(x_input[itm], support[itm],needunique=False)) 
          else:
            input_graph.append(support[itm])
    else:
      input_graph = support 
    
    model_output = self.forward(input_graph=input_graph,sigma=sigma)
    assert torch.isnan(model_output).any() == False

    if self.loss_type == 'bd3':
       # 1. Batch the graphs to easily get all edges and map them back
        batch_g = dgl.batch(graph) # Use the original graph here
        b_src, b_dst = batch_g.edges()
        b_rel = batch_g.edata['rel']

        # 2. Create mapping from batched edge index to original graph index
        num_edges_per_graph = batch_g.batch_num_edges()
        graph_indices = torch.arange(len(graph), device=model_output.device).repeat_interleave(num_edges_per_graph)

        # 3. Create mapping from batched node index to original node index within each graph
        num_nodes_per_graph = batch_g.batch_num_nodes()
        node_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=model_output.device), num_nodes_per_graph[:-1]]), dim=0)
        orig_src = b_src - node_offsets[graph_indices]
        orig_dst = b_dst - node_offsets[graph_indices]
        # loss_scale:B,1
        noedge_scale = loss_scale.clone()
        
        loss_scale = loss_scale.repeat_interleave(num_edges_per_graph)
        # 4. Use advanced indexing to gather the log probabilities for true edges and relations
        # model_output shape: (B, N, N, R)
        # Indices shape: (num_total_edges,)
        # true_log_probs_flat = model_output[graph_indices, orig_src, orig_dst, b_rel]
        # true_log_probs_flat2 = model_output[graph_indices, orig_dst, orig_src, b_rel+self.num_rel]
        # true_log_probs_flat3 = model_output[graph_indices, , self.num_rel*2]
         # 4.1 Log probability mean for true forward edges (orig_src -> orig_dst, rel = b_rel)
        true_log_probs_forward = loss_scale*model_output[graph_indices, orig_src, orig_dst, b_rel]
        mean_log_probs_forward = true_log_probs_forward.mean()

        # 4.2 Log probability mean for true inverse edges (orig_dst -> orig_src, rel = b_rel + num_rel)
        inverse_rel = b_rel + self.num_rel
        true_log_probs_inverse = loss_scale*model_output[graph_indices, orig_dst, orig_src, inverse_rel]
        mean_log_probs_inverse = true_log_probs_inverse.mean()

        # 4.3 Log probability mean for "no edge" where no true edge (forward or inverse) exists
        no_edge_index = self.num_rel * 2

        # Create masks for true edges in the (B, N, N) space to identify non-edges
        B, N, _, R = model_output.shape
        forward_mask = torch.zeros((B, N, N), dtype=torch.bool, device=model_output.device)
        inverse_mask = torch.zeros((B, N, N), dtype=torch.bool, device=model_output.device)

        # Mark locations of true forward and inverse edges
        forward_mask[graph_indices, orig_src, orig_dst] = True
        inverse_mask[graph_indices, orig_dst, orig_src] = True

        # Identify pairs that are neither forward nor inverse true edges
        non_edge_mask = ~(forward_mask | inverse_mask)
        
        valid_pair_mask = (model_output != 0.).any(dim=-1) # Shape (B, N, N)
        non_edge_mask = non_edge_mask & valid_pair_mask
        log_probs_no_edge_all = model_output[non_edge_mask]
        # Select only those where 
        noedge_scale = noedge_scale.repeat_interleave(torch.sum(non_edge_mask, dim=(1, 2)))
        log_probs_no_edge = noedge_scale*log_probs_no_edge_all[:,no_edge_index] 
        log_probs_no_edge = log_probs_no_edge.mean()
        
       
       
       

        # 5. Calculate the final loss
        # Minimize the negative sum of scaled log likelihoods
        loss = ((mean_log_probs_forward + mean_log_probs_inverse + log_probs_no_edge)).mean()

    

        # The variable 'loss' now holds the computed loss value.

    else:
        # return self.cross_loss(model_output=model_output, graph=query_graph, support=support)
        return self.bce_loss(model_output=model_output, graph=query_graph, support=input_graph)
        # return self.bce_loss(model_output=model_output, graph=ground_truth_graph, support=input_graph)
        B, N, _, DIM = model_output.shape
        device = model_output.device
        no_edge_index = self.num_rel * 2 # Index representing no edge

        # 1. Create target tensor initialized with 'no edge' index
        target = torch.full((B, N, N), no_edge_index, dtype=torch.long, device=device)

        # 2. Populate target using batched ground truth graph
        # It's crucial to use the original 'graph' list here for ground truth labels
        batch_g_truth = dgl.batch(graph).to(device)
        b_src, b_dst = batch_g_truth.edges()
        b_rel = batch_g_truth.edata['rel']

        # Map batched indices back to B, N, N space
        num_edges_per_graph = batch_g_truth.batch_num_edges()
        graph_indices = torch.arange(len(graph), device=device).repeat_interleave(num_edges_per_graph)
        num_nodes_per_graph = batch_g_truth.batch_num_nodes()
        node_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes_per_graph[:-1]]), dim=0)
        orig_src = b_src - node_offsets[graph_indices]
        orig_dst = b_dst - node_offsets[graph_indices]

     
        # Fill target with forward and inverse edges
        # target[graph_indices, orig_src, orig_dst] = b_rel
        # target[graph_indices, orig_dst, orig_src] = b_rel + self.num_rel # Inverse relations

        edge_mask = ((node_mask)[:, :, None] & (node_mask)[:, None, :])
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        mask_E = edge_mask & ~diag_mask       
         # === 支持集边mask 进行loss ===
        if self.support_flag:
            for b, g in enumerate(support):
                src, dst = g.edges()
                mask_E[b, src, dst] = False
                # mask_E[b, dst, src] = False
        # 3. Reshape for F.cross_entropy
        # Input: (B, N, N, DIM) -> (B*N*N, DIM)
        model_output_flat = model_output[mask_E].view(-1, DIM)
        # Target: (B, N, N) -> (B*N*N)
        target_flat = target[mask_E].view(-1)

        num_classes = self.num_rel * 2 + 1
        weights = torch.ones(num_classes, device=device)
        # 空边占比99%，非空边占比1%
        weights[no_edge_index] = 0.005  # 空边权重极小
        # 其余类别权重为1, 待改进，support的边是不是就不用计算loss了？
        loss = F.cross_entropy(model_output_flat, target_flat,weight=weights,reduction='mean')

      
    
    return loss

  def _loss(self, batch, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    
   
    loss = self._forward_pass_diffusion(
      batch,
      None,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max,)
    
    return Loss(loss=loss)

 
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
