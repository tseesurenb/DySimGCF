'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import torch
import torch.nn.functional as F

import numpy as np

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from world import config
from torch_geometric.utils import softmax
                   

# NGCF Convolutional Layer
class NGCFConv(MessagePassing):
  def __init__(self, emb_dim, dropout, bias=True, **kwargs):  
    super(NGCFConv, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout

    self.lin_1 = nn.Linear(emb_dim, emb_dim, bias=bias)
    self.lin_2 = nn.Linear(emb_dim, emb_dim, bias=bias)

    self.init_parameters()

  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)

  def forward(self, x, edge_index, edge_attrs, scale):
    # Compute normalization
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=norm)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)

  def message(self, x_j, x_i, norm):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) 

# LightGCN Convolutional Layer     
class lightGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        self.norm = None
            
    def forward(self, x, edge_index, edge_attrs):
      
        if self.norm is None:
          # Compute normalization
          from_, to_ = edge_index
          deg = degree(to_, x.size(0), dtype=x.dtype)
          deg_inv_sqrt = deg.pow(-0.5)
          deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
          self.norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# DySimGCF Convolutional Layer
class DySimGCF(MessagePassing):
    def __init__(self, self_loop = False, device = 'cpu', **kwargs):  
        super().__init__(aggr='add')
        
        self.graph_norms = None
        self.edge_attrs = None
        self.add_self_loops = self_loop
        
    def forward(self, x, edge_index, edge_attrs):
        
        if self.graph_norms is None:
          
          from_, to_ = edge_index      
          incoming_norm = softmax(edge_attrs, to_)
          outgoing_norm = softmax(edge_attrs, from_)
          
          if config['abl_study'] == -1:
            norm = outgoing_norm
          elif config['abl_study'] == 1:
            norm = incoming_norm
          else:
            norm = torch.sqrt(incoming_norm * outgoing_norm)
            
          self.graph_norms = norm
                    
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
      
class RecSysGNN(nn.Module):
  def __init__(
      self,
      emb_dim, 
      n_layers,
      n_users,
      n_items,
      model, # 'NGCF' or 'LightGCN' or 'hyperGCN'
      dropout=0.1, # Only used in NGCF
      device = 'cpu',
      self_loop = False
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'lightGCN') or model == 'DySimGCF', 'Model must be NGCF or LightGCN or DySimGCF'
    self.model = model
    self.n_users = n_users
    self.n_items = n_items
    self.n_layers = n_layers
    self.emb_dim = emb_dim
      
    self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim, dtype=torch.float32)
        
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
    elif self.model == 'lightGCN':
      self.convs = nn.ModuleList(lightGCN() for _ in range(self.n_layers))
    elif self.model == 'DySimGCF':
      self.convs = nn.ModuleList(DySimGCF(self_loop=self_loop, device=device) for _ in range(self.n_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or DySimGCF')
    
    self.init_parameters()

  def init_parameters(self):
        
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      nn.init.normal_(self.embedding.weight, std=0.1)

  def forward(self, edge_index, edge_attrs):
    
    emb0 = self.embedding.weight
    embs = [emb0]
     
    emb = emb0
    for conv in self.convs:
      emb = conv(x=emb, edge_index=edge_index, edge_attrs=edge_attrs)
      embs.append(emb)
      
    
    if self.model == 'NGCF':
      out = torch.cat(embs, dim=-1)
    else:
      out = torch.mean(torch.stack(embs, dim=0), dim=0)
        
    return emb0, out


  def encode_minibatch(self, users, pos_items, neg_items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)
    
    return (
        out[users], 
        out[pos_items], 
        out[neg_items],
        emb0[users],
        emb0[pos_items],
        emb0[neg_items],
    )
    
  def predict(self, users, items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)    
    return torch.matmul(out[users], out[items].t())

# define a function that compute all users scoring for all items and then save it to a file. later, I can be able to get top-k for a user by user_id
def get_all_predictions(model, edge_index, edge_attrs, device):
    model.eval()
    users = torch.arange(model.n_users).to(device)
    items = torch.arange(model.n_items).to(device)
    predictions = model.predict(users, items, edge_index, edge_attrs)
    return predictions.cpu().detach().numpy()
  
# define a function that get top-k items for a user by user_id after sorting the predictions
def get_top_k(user_id, predictions, k):
    return np.argsort(predictions[user_id])[::-1][:k]