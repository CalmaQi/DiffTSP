import os
import random
import pickle
import itertools
from collections import defaultdict
from dataclasses import dataclass
from collections import Counter
import math
import dgl
import torch
import pickle
import tqdm
import numpy as np
import torch.nn.functional as F
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def batch_shortest_path_sparse_tensor(graphs, max_nodes=None, max_distance=None, unreachable_value=-1):
    """
    args:
        graphs: DGL
        max_nodes: 
        max_distance: 
        unreachable_value: 
    return:
        indices:(3, nnz) - (batch_idx, row, col)
        values: (nnz,)
        shape:(batch_size, max_nodes, max_nodes)
        actual_nodes: 
        max_dist: 
    """
    batch_size = len(graphs)
    
    # Determine maximum number of nodes
    if max_nodes is None:
        max_nodes = max([g.num_nodes() for g in graphs])
    
    # Collect all indices and values
    all_indices = []
    all_values = []
    actual_nodes = []
    max_dist = 0
    
    for i, g in enumerate(graphs):
        num_nodes = g.num_nodes()
        actual_nodes.append(num_nodes)
        
        # Convert DGL graph to NetworkX graph
        nx_g = dgl.to_networkx(g, edge_attrs=['weight'] if 'weight' in g.edata else None)
        
        # Get edge weights (if they exist)
        if 'weight' in g.edata:
            adj_matrix = nx.to_scipy_sparse_array(nx_g, weight='weight')
        else:
            adj_matrix = nx.to_scipy_sparse_array(nx_g)
        
        # Calculate shortest paths
        dist_matrix = shortest_path(
            csgraph=adj_matrix,
            method='auto',
            directed=g.is_multigraph,
            return_predecessors=False
        )
        
        # Process reachable paths
        rows, cols = np.where(np.isfinite(dist_matrix))
        
        # Filter out distances exceeding max_distance
        if max_distance is not None:
            valid_indices = dist_matrix[rows, cols] <= max_distance
            rows = rows[valid_indices]
            cols = cols[valid_indices]
        
        # Get corresponding distance values
        values = dist_matrix[rows, cols]
        
        # Update maximum distance
        if len(values) > 0:
            current_max = np.max(values)
            max_dist = max(max_dist, current_max)
        
        # Add batch index
        batch_indices = np.full_like(rows, i)
        
        # Collect indices and values
        if len(rows) > 0:
            batch_rows_cols = np.stack([batch_indices, rows, cols], axis=0)
            all_indices.append(batch_rows_cols)
            all_values.append(values)
    
    # If no valid paths
    if not all_indices:
        # Create an empty sparse tensor
        indices = torch.zeros((3, 0), dtype=torch.long)
        values = torch.zeros(0, dtype=torch.float)
        return indices, values, (batch_size, max_nodes, max_nodes), actual_nodes, 0.0
    
    # Merge indices and values from all batches
    indices = np.concatenate(all_indices, axis=1)
    values = np.concatenate(all_values)
    
    # Convert to PyTorch tensor
    indices = torch.tensor(indices, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float)
    
    return indices, values, (batch_size, max_nodes, max_nodes), actual_nodes, max_dist

def sparse_to_dense(indices, values, shape, default_value=-1):
    """
    Convert sparse representation to dense tensor (for visualization or specific operations that require it)
    
    Args:
        indices: Sparse tensor indices, shape (3, nnz)
        values: Corresponding distance values, shape (nnz,)
        shape: Shape of sparse tensor (batch_size, max_nodes, max_nodes)
        default_value: Default value for unstored positions
        
    Returns:
        Dense tensor
    """
    dense = torch.full(shape, default_value, dtype=values.dtype)
    if indices.shape[1] > 0:  # If there are elements
        dense[indices[0], indices[1], indices[2]] = values
    return dense

def get_distance_at_indices(indices, values, batch_idx, src_nodes, dst_nodes):
    """
    Get distance values at specific indices
    
    Args:
        indices: Sparse tensor indices, shape (3, nnz)
        values: Corresponding distance values, shape (nnz,)
        batch_idx: Batch index
        src_nodes: Source node indices
        dst_nodes: Target node indices
        
    Returns:
        Distance values at corresponding indices
    """
    if not isinstance(src_nodes, torch.Tensor):
        src_nodes = torch.tensor(src_nodes)
    if not isinstance(dst_nodes, torch.Tensor):
        dst_nodes = torch.tensor(dst_nodes)
    
    # Ensure consistent dimensions
    if len(src_nodes.shape) == 0:
        src_nodes = src_nodes.unsqueeze(0)
    if len(dst_nodes.shape) == 0:
        dst_nodes = dst_nodes.unsqueeze(0)
    
    # Build query indices
    batch_indices = torch.full_like(src_nodes, batch_idx)
    query_indices = torch.stack([batch_indices, src_nodes, dst_nodes], dim=0)
    
    # Find matches in sparse indices
    distances = []
    for i in range(query_indices.shape[1]):
        # Check each query point individually
        query = query_indices[:, i:i+1]
        
        # Find positions where all dimensions match
        matches = (indices == query).all(dim=0)
        
        if matches.any():
            # Take the value at the first matching position
            match_idx = torch.where(matches)[0][0]
            distances.append(values[match_idx].item())
        else:
            # If no match, it means unreachable
            distances.append(-1)
    
    return distances


def graph_inverse(graphs, num_rel, in_place=True):
    """
    Add inverse edges to each graph in the list.
    Args:
        graphs: list of DGLGraph
        num_rel: int, number of relations (for inverse rel offset)
        in_place: bool, whether to modify in place
    """
    g=graphs    
    src, dst = g.edges()
    rel = g.edata['rel']
    inverse_rel = rel + num_rel
    g.add_edges(dst, src, data={'rel': inverse_rel})
    return g  

@dataclass
class Graph:
    edge_index: torch.Tensor
    num_nodes: int
    num_relations: int
    
    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        return self

class KGSubgraph:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file_path = os.path.join(self.cfg.dataset.datadir,'split-subgraph-0.8_J3.pkl')
        self.file_path = os.path.join(self.cfg.dataset.datadir,'mysplit-500.pkl')
        self.subgraphs = None
        self.max_n_nodes = cfg.model.max_n_nodes
        self.setup()
        
    def setup(self):
         # Read pkl file
        with open(self.file_path, 'rb') as file:
            self.subgraphs = pickle.load(file)

    def build_dgl_graph_from_tensor(self,triples: torch.Tensor):
        """
        triples is a tensor of shape (N, 3):
        Each row is [orig_src, rel, orig_dst] in order,
        where orig_src/orig_dst themselves may not start from 0, nor be consecutive.
        """
        # Original node and relation indices
        orig_src = triples[:, 0].long()
        rel = triples[:, 1].long()
        orig_dst = triples[:, 2].long()
        
        # Get all original nodes that appear
        unique_ids = torch.cat([orig_src, orig_dst]).unique()
        # Sort them and map to [0, 1, 2, ..., num_nodes-1]
        sorted_ids = unique_ids.sort()[0]
        node_to_idx = {orig_id.item(): i for i, orig_id in enumerate(sorted_ids)}
        
        # Replace old indices with new indices
        new_src = torch.tensor([node_to_idx[x.item()] for x in orig_src], dtype=torch.long)
        new_dst = torch.tensor([node_to_idx[x.item()] for x in orig_dst], dtype=torch.long)
        num_nodes = len(sorted_ids)  # Total number of new nodes
        
        # Build DGL graph
        g = dgl.graph((new_src, new_dst), num_nodes=num_nodes)
        
        # Store relation information of edges
        g.edata['rel'] = rel
        
        # Keep original id information for each new node in ndata
        # Index i corresponds to sorted_ids[i] which is the original node id
        g.ndata['orig_id'] = sorted_ids
        
        return g
    
    def get_subtriple(self, triples: torch.Tensor, ratio: float = 0.8,unique_relations=None,target_counts=None):
        """
        Sample support and query subsets from triples.
        Try to sample proportionally by relation type to better preserve graph structure.

        Args:
            triples: Tensor of shape (N, 3) [src, rel, dst].
            ratio: Proportion of support set should account for total triple count.

        Returns:
            (support_triples, query_triples)
        """
        
        num_triples = triples.shape[0]
        
        # 4. For each relation, sample the corresponding number of triple indices
        support_indices_list = []
        all_indices = torch.arange(num_triples, device=triples.device)
        for r in unique_relations:
            count_to_sample = target_counts[r]
            if count_to_sample > 0:
                # Find indices of all triples with relation r
                rel_mask = triples[:, 1] == r
                rel_indices = all_indices[rel_mask]
                
                # Ensure sampling count doesn't exceed actual available count
                count_to_sample = min(count_to_sample, len(rel_indices))

                if count_to_sample > 0:
                  # Random sampling
                  perm = torch.randperm(len(rel_indices), device=triples.device)
                  sampled_rel_indices = rel_indices[perm[:count_to_sample]]
                  support_indices_list.append(sampled_rel_indices)

        # 5. Merge indices and get support and query sets
        if not support_indices_list:
            support_indices = torch.empty((0,), dtype=torch.long, device=triples.device)
        else:
            support_indices = torch.cat(support_indices_list)

        # Since previous adjustments may not be perfect, confirm support_indices count again
        # If count is insufficient for final_num_support, consider random supplementation, but this will slightly break proportionality
        # Here we choose to accept the current sampled count
        
        support_mask = torch.zeros(num_triples, dtype=torch.bool, device=triples.device)
        if len(support_indices) > 0:
             support_mask[support_indices] = True
  
        return support_indices, support_mask
    
    def get_supportq(self, dglgraph,ratio=0.8):
        """
        ratio: 0.0 ~ 1.0
        Calculate support set size based on ratio
        Returns support set and query set
        If tensor, returns tensor
        If dglgraph, returns dglgraph
        """
        
        
        if isinstance(dglgraph, torch.Tensor):
            triples = dglgraph
            num_triples = triples.shape[0]
           
            
            # 1. Count the number of each relation
            relations = triples[:, 1].tolist()
            relation_counts = Counter(relations)
            unique_relations = sorted(relation_counts.keys())

            # 2. Calculate target count for each relation in support set (sample proportionally within each relation)
            target_counts = {}
            for r in unique_relations:
                # Directly calculate sampling count for this relation by proportion
                target_counts[r] = math.ceil(relation_counts[r] * ratio)

            support_indices, support_mask = self.get_subtriple(triples,ratio=ratio,
                                                unique_relations=unique_relations,target_counts=target_counts)
            # 3. Get support and query sets
            # support = triples[support_mask]
            # query = triples[~support_mask]
            all_indices = torch.arange(num_triples, device=triples.device)
            query_indices = all_indices[~support_mask]
            
            
            return support_indices, query_indices
        else:
            raise ValueError("Unsupported graph type. Please provide a DGLGraph or a tensor.")    
    
    def add_inverse_relation(self, triples, rel_num):
        """
        Add inverse relations
        """
        inverse_triples = []
        for h, r, t in triples:
            inverse_triples.append((t, r + rel_num, h))
        return torch.cat([triples, torch.tensor(inverse_triples)])
    
    def get_train_data(self,ratio=0.8,num_repeat=500,add_inverse=False,rel_num=0):
        """
        ratio: 0.0 ~ 1.0
        """
        train_data=[]
        for itm in tqdm.tqdm(self.subgraphs, desc="Subgraphs"):
            triples = self.subgraphs[itm]
            if add_inverse:
                # Add inverse relations
                triples = self.add_inverse_relation(triples,rel_num)
            dgl_g=self.build_dgl_graph_from_tensor(triples)
            assert self.cfg.model.max_n_nodes >= dgl_g.num_nodes() 
            kggraph_triple=self.dgl2tensor(dgl_g)
            for rep in tqdm.tqdm(range(num_repeat), desc="Repeats", leave=False):
                # Random sampling, TODO: improve - are ratios all the same, are repeats all the same, if memory usage is high, get support during each batch later
                support_idx, query_idx = self.get_supportq(kggraph_triple, ratio)
                # support_graph = dgl_g.edge_subgraph(support_idx, preserve_nodes=True)
                # query_graph = dgl_g.edge_subgraph(query_idx, preserve_nodes=True)
                support_graph = dgl_g.edge_subgraph(support_idx, relabel_nodes=False)
                missing_nodes = torch.arange(dgl_g.num_nodes(), device=dgl_g.device)[~torch.isin(torch.arange(dgl_g.num_nodes(), device=dgl_g.device), support_graph.nodes())]
                support_graph.add_nodes(len(missing_nodes))
                support_graph.ndata['orig_id'] = dgl_g.ndata['orig_id']
                
                query_graph = dgl_g.edge_subgraph(query_idx, relabel_nodes=False)
                missing_nodes = torch.arange(dgl_g.num_nodes(), device=dgl_g.device)[~torch.isin(torch.arange(dgl_g.num_nodes(), device=dgl_g.device), query_graph.nodes())]
                query_graph.add_nodes(len(missing_nodes))
                query_graph.ndata['orig_id'] = dgl_g.ndata['orig_id']
                # query_graph = dgl_g.edge_subgraph(query_idx, preserve_nodes=True)
                # train_data.append((support_graph,query_graph))
                
                # train_data.append((support_graph,dgl_g))
                support_graph=graph_inverse(support_graph,self.cfg.num_rel)
                query_graph=graph_inverse(query_graph,self.cfg.num_rel)
                train_data.append((support_graph,query_graph ) )
        return train_data
    
    
            
    def dgl2tensor(self, dglgraph): 
        """
        Convert DGL graph to tensor
        """
        src = dglgraph.edges()[0]
        dst = dglgraph.edges()[1]
        rel = dglgraph.edata['rel']
        #         # If original ID is needed, can restore using g.ndata['orig_id']
        orig_src = dglgraph.ndata['orig_id'][src]
        orig_dst = dglgraph.ndata['orig_id'][dst]

        # Here src and dst are original node ids
        triples = torch.stack([src, rel, dst], dim=1)
        
        return triples
    
    
    
    
class TransductiveKnowledgeGraph:
    def __init__(self, cfg):
        self.data_path = cfg.dataset.datadir
        self.cfg = cfg
        self.entity2id = dict()
        self.id2entity = dict()
        self.relation2id = dict()
        self.id2relation = dict()
        
        with open(os.path.join(self.data_path, 'entities.dict'), 'r', encoding='utf-8') as fread:
            for line in fread:
                eid,entity= line.strip().split('\t')
                self.entity2id[entity] = int(eid)
                self.id2entity[int(eid)] = entity
                
        with open(os.path.join(self.data_path, 'relations.dict'), 'r', encoding='utf-8') as fread:
            for line in fread:
                rid,relation = line.strip().split('\t')
                self.relation2id[relation] = int(rid)
                self.id2relation[int(rid)] = relation
                
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        assert self.num_relation == self.cfg.num_rel
        
        raw_train_triplets = self.read_triplets(os.path.join(self.data_path, 'train.txt'))
        raw_valid_triplets = self.read_triplets(os.path.join(self.data_path, 'valid.txt'))
        raw_test_triplets = self.read_triplets(os.path.join(self.data_path, 'test.txt'))
        
        self.train_triplets = self.encode_triplets(raw_train_triplets)
        self.valid_triplets = self.encode_triplets(raw_valid_triplets)
        self.test_triplets = self.encode_triplets(raw_test_triplets)
        
        # self.train_filters = self.get_filters(self.train_triplets)
        # self.valid_filters = self.get_filters(self.train_triplets + self.valid_triplets + self.test_triplets)
        # self.test_filters = self.get_filters(self.train_triplets + self.valid_triplets + self.test_triplets)
        
        # self.train_answers = self.get_filters(self.train_triplets)
        # self.valid_answers = self.get_filters(self.valid_triplets)
        # self.test_answers = self.get_filters(self.test_triplets)
        
        self.train_triplets = torch.tensor(self.train_triplets)
        self.valid_triplets = torch.tensor(self.valid_triplets)
        self.test_triplets = torch.tensor(self.test_triplets)
        self.test_triplets = self.test_triple_addverse(self.test_triplets)
        self.train_triplets = self.test_triple_addverse(self.train_triplets)
        self.valid_triplets = self.test_triple_addverse(self.valid_triplets)
        self.seen_triplets = torch.cat([self.train_triplets, self.valid_triplets], dim=0)
        # self.test_triplets =  torch.cat([self.test_triplets, self.valid_triplets], dim=0)
        # self.infer_data = torch.cat([self.train_triplets, self.test_triplets], dim=0)
        # self.infer_data2 = torch.cat([self.train_triplets, self.test_triplets, self.valid_triplets], dim=0)
        # self.edge_index = self.train_triplets.clone()
        
        # self.train_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        # self.valid_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        # self.test_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.kgsubgraph = KGSubgraph(self.cfg)
        self.subgraphs = self.kgsubgraph.subgraphs
        self.train_data = self.kgsubgraph.get_train_data(ratio=self.cfg.dataset.ratio, num_repeat=self.cfg.dataset.repeat,
                                                         add_inverse=False,rel_num=self.num_relation)
        self.test_data = self.get_vaild_data()
        self.valid_data = self.get_vaild_data()
        # self.get_rel_ratio(self.train_triplets)

    def get_rel_ratio(self,triples):
        """
        Calculate the proportion of each relation (including empty edges)
        Args:
            triples: torch.Tensor, shape (N, 3), each row is [src, rel, dst]
        Returns:
            dict: {rel: ratio}
        """
        # Count all appearing relations
        rels = triples[:, 1].tolist()
        total = self.num_entity*self.num_entity
        num_classes = self.num_relation * 2 + 1
        count = [0] * num_classes
        for r in rels:
            count[r] += 1
        count[-1] = total - len(rels)  # Count of empty edge category
        # Calculate proportions
        ratio = {rel: count[rel] / total for rel in range(num_classes)}
        return ratio
    
    def restore_full_graph_from_subgraph(self, dglsubgraph, alltriples):
        """
        subgraph: dgl.DGLGraph
        alltriples: torch.Tensor, shape (K, 3)
        Directly add missing edges to subgraph (without introducing new entities)
        """
        # 1. Get existing entity set (original ids)
        subgraph = dglsubgraph.clone()
        entities = set(subgraph.ndata['orig_id'].tolist())
        # 2. Get existing edge set
        src, dst = subgraph.edges()
        rel = subgraph.edata['rel']
        sub_edges = set((subgraph.ndata['orig_id'][src[i]].item(),
                         rel[i].item(),
                         subgraph.ndata['orig_id'][dst[i]].item()) for i in range(subgraph.num_edges()))
        # 3. Find edges in alltriples where both entities are in entities but not in sub_edges
        mask_h = torch.tensor([h.item() in entities for h in alltriples[:, 0]])
        mask_t = torch.tensor([t.item() in entities for t in alltriples[:, 2]])
        mask = mask_h & mask_t
        filtered = alltriples[mask]
        # 4. Add missing edges
        new_src, new_dst, new_rel = [], [], []
        orig_id_to_idx = {oid.item(): idx for idx, oid in enumerate(subgraph.ndata['orig_id'])}
        for h, r, t in filtered:
            key = (h.item(), r.item(), t.item())
            if key not in sub_edges:
                new_src.append(orig_id_to_idx[h.item()])
                new_dst.append(orig_id_to_idx[t.item()])
                new_rel.append(r.item())
               
        if new_src:
            subgraph.add_edges(
                torch.tensor(new_src, device=subgraph.device),
                torch.tensor(new_dst, device=subgraph.device),
                data={'rel': torch.tensor(new_rel, device=subgraph.device)}
            )
        return subgraph
    
    def test_triple_addverse(self, triples):
        """
        Add inverse relations
        """
        inverse_triples = []
        for h, r, t in triples:
            inverse_triples.append((t, r + self.num_relation, h))
            assert r + self.num_relation < self.num_relation * 2
        return torch.cat([triples, torch.tensor(inverse_triples)])
    def get_vaild_data(self):
        """
        Get validation set data
        """
        valid_data = []
        infer_ground=[]
        for itm in tqdm.tqdm(self.subgraphs, desc="Valid_Subgraphs"):
            triples = self.subgraphs[itm]
            dgl_g=self.kgsubgraph.build_dgl_graph_from_tensor(triples)
            dgl_g= graph_inverse(dgl_g,self.num_relation)
            # infer_g = self.restore_full_graph_from_subgraph(dgl_g, self.test_triplets)
            # infer_ground.append(infer_g)
            valid_data.append((dgl_g, None))
        # import ipdb; ipdb.set_trace()    
        # with open('infer_ground.pkl', 'wb') as f:
        #     pickle.dump(infer_ground, f)
        return valid_data
    def read_triplets(self, file_name):
        triplets = list()
        with open(file_name, 'r', encoding='utf-8') as fread:
            for line in fread:
                h, r, t = line.strip().split('\t')
                triplets.append((h, r, t))
        return triplets
    
    def encode_triplets(self, raw_triplets):
        encoded_triplets = list()
        for triplet in raw_triplets:
            h, r, t = self.entity2id[triplet[0]], self.relation2id[triplet[1]], self.entity2id[triplet[2]]
            # rev_r = self.relation2id['-' + triplet[1]]
            encoded_triplets.append((h, r, t))
            # encoded_triplets.append((t, rev_r, h))
        return encoded_triplets
    
    
    
    def get_filters(self, triplets):
        filters = defaultdict(set)
        for h, r, t in triplets:
            filters[(h, r)].add(t)
        return filters

    
    
    def train_collate_fn(self, batch):
        # batch_size = len(batch)
        
        x_list, y_list,mx_list,my_list = [], [],[],[]
        for (x, y) in batch:
            x_list.append(x)
            y_list.append(y)
         
        
        return {
            'support': x_list,
            'graph':y_list,
           
        }
    
    def valid_collate_fn(self, batch):
        x_list, y_list = [], []
        for (x, y) in batch:
            x_list.append(x)
            y_list.append(y)
        
        return {
            'support': x_list,
            'graph':y_list
        }
    
    def test_collate_fn(self, batch):
        return self.valid_collate_fn(batch)
        
def dgl2edge_index(dglgraph:list,max_n_nodes=180,DIM=25,no_edge_index=24):
    """
    Convert DGL graph to edge_index
    """
    B = len(dglgraph)
    N = max_n_nodes
    # edge_index = torch.zeros((B, N, N, DIM)) 
    # 1. Create Binary edgeindex Tensor
    edge_index = F.one_hot(torch.full((B, N, N), no_edge_index, dtype=torch.long), 
                             num_classes=no_edge_index+1)

    # Populate target using batched ground truth graph (including inverses)
    batch_g = dgl.batch(dglgraph)
    b_src, b_dst = batch_g.edges()
    b_rel = batch_g.edata['rel'] # These include original + inverse rel indices
    
    # Map batched indices back to B, N, N space
    num_edges_per_graph = batch_g.batch_num_edges()
    graph_indices = torch.arange(B).repeat_interleave(num_edges_per_graph)
    num_nodes_per_graph = batch_g.batch_num_nodes() # Should be same as original
    node_offsets = torch.cumsum(torch.cat([torch.tensor([0]), num_nodes_per_graph[:-1]]), dim=0)
    orig_src = b_src - node_offsets[graph_indices]
    orig_dst = b_dst - node_offsets[graph_indices]

    edge_index[graph_indices, orig_src, orig_dst, b_rel] = 1
    edge_index[graph_indices, orig_dst, orig_src, no_edge_index] = 0
    # 2. Create Node Mask  
    row_indices = torch.arange(N).view(1, N, 1)
    num_nodes_b = num_nodes_per_graph.view(B, 1, 1)
    node_mask = row_indices < num_nodes_b
    return  edge_index,node_mask

def edge_index_to_batch_g(edge_index, node_mask, no_edge_index):
    """
    Reconstructs a batched DGL graph from a dense edge_index tensor and node_mask.

    Args:
        edge_index: torch.Tensor, shape (B, N, N, DIM). Value 1 indicates an edge.
        node_mask: torch.Tensor, shape (B, N). True for valid nodes.
        no_edge_index: int. The index in the DIM dimension representing 'no edge'.

    Returns:
        dgl.DGLGraph: The batched graph.
        list: List containing the number of nodes for each graph in the batch.
    """
    B, N, _, DIM = edge_index.shape
    device = edge_index.device
    graph_list = []
    num_nodes_list = []

    for b in range(B):
        # 1. Determine actual number of nodes for this graph
        num_nodes_b = node_mask[b].sum().item()
        num_nodes_list.append(num_nodes_b)
        if num_nodes_b == 0:
            # Handle empty graph case
            g = dgl.graph(([], []), num_nodes=0, device=device)
            graph_list.append(g)
            continue

        # 2. Extract relevant slice and find edges (excluding 'no edge' type)
        # Slice edge_index for valid nodes and edge types
        edge_slice = edge_index[b, :num_nodes_b, :num_nodes_b, :no_edge_index] 
        
        # Find indices where an edge exists (value is 1 or > 0)
        src_indices, dst_indices, rel_indices = torch.nonzero(edge_slice > 0, as_tuple=True) 

        # 3. Create individual DGL graph
        g = dgl.graph((src_indices, dst_indices), num_nodes=num_nodes_b, device=device)

        # 4. Assign edge relations if any edges exist
        if src_indices.numel() > 0:
            # Ensure relation indices are long type for DGL
            g.edata['rel'] = rel_indices.long() 

        graph_list.append(g)

    # 5. Batch the individual graphs
    if not graph_list:
        # Handle case where the entire batch might be empty
        return dgl.batch([]), [] 
        
    batch_g = dgl.batch(graph_list)
    
    # You might also need the num_nodes_per_graph info which was calculated
    # num_nodes_per_graph = torch.tensor(num_nodes_list, device=device, dtype=torch.long)

    return batch_g 
