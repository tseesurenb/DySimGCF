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
# Then import it in your code
from torchdiffeq import odeint
                   

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
          # incoming_norm = softmax(edge_attrs, to_)
          # outgoing_norm = softmax(edge_attrs, from_)
        
          # You could add a temperature parameter to control softmax sharpness
          temperature = config['s_temp']
          incoming_norm = softmax(edge_attrs / temperature, to_)
          outgoing_norm = softmax(edge_attrs / temperature, from_)
          
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


# DyODE-GCF: Dynamic ODE-based Graph Convolutional Filtering
class DyODE_GCF(MessagePassing):
    def __init__(self, 
                 latent_dim, 
                 solver='dopri5', 
                 time_split=3, 
                 max_time=10.0, 
                 learnable_time=True, 
                 dual_res=False, 
                 self_loop=False, 
                 device='cpu', 
                 **kwargs):
        super().__init__(aggr='add')
        self.graph_norms = None
        self.edge_attrs = None
        self.add_self_loops = self_loop
        self.device = device
        
        # ODE specific parameters
        self.latent_dim = latent_dim
        self.solver = solver
        self.time_split = time_split
        self.max_time = max_time
        self.learnable_time = learnable_time
        self.dual_res = dual_res
        
        # Initialize ODE components
        self.__init_ode()
        
    def __init_ode(self):
        if self.learnable_time:
            # Learnable time points
            self.odetimes = ODETimeSetter(self.time_split, self.max_time)
            
            # ODE blocks with learnable time points
            self.ode_block_1 = ODEBlockTimeFirst(ODEFunction(), self.time_split, self.solver)
            self.ode_block_2 = ODEBlockTimeMiddle(ODEFunction(), self.time_split, self.solver)
            self.ode_block_3 = ODEBlockTimeMiddle(ODEFunction(), self.time_split, self.solver)
            self.ode_block_4 = ODEBlockTimeLast(ODEFunction(), self.time_split, self.solver, self.max_time)
        else:
            # Fixed time points - make sure they're strictly increasing
            times = torch.linspace(0, self.max_time, self.time_split + 1)
            self.odetime_splitted = times
            
            # ODE blocks with fixed time intervals
            self.ode_block_1 = ODEBlock(ODEFunction(), self.solver, times[0], times[1])
            self.ode_block_2 = ODEBlock(ODEFunction(), self.solver, times[1], times[2])
            self.ode_block_3 = ODEBlock(ODEFunction(), self.solver, times[2], times[3])
            self.ode_block_4 = ODEBlock(ODEFunction(), self.solver, times[3], self.max_time)
        
    def get_times(self):
        if self.learnable_time:
            ode_times = list(self.odetime_1) + list(self.odetime_2) + list(self.odetime_3)
            return ode_times
        else:
            return self.odetime_splitted
    
    def setup_ode_function(self, edge_index, edge_attrs, x):
        # Calculate normalized edge weights similar to your DySimGCF approach
        from_, to_ = edge_index
        temperature = config['s_temp']
        incoming_norm = softmax(edge_attrs / temperature, to_)
        outgoing_norm = softmax(edge_attrs / temperature, from_)
        
        if config['abl_study'] == -1:
            norm = outgoing_norm
        elif config['abl_study'] == 1:
            norm = incoming_norm
        else:
            norm = torch.sqrt(incoming_norm * outgoing_norm)
            
        self.graph_norms = norm
        
        # Update the ODEFunction with the current graph structure
        sparse_size = (x.size(0), x.size(0))
        edge_index_with_weights = torch.sparse.FloatTensor(edge_index, self.graph_norms, sparse_size)
        
        # Set the graph for each ODE function
        if self.learnable_time:
            self.ode_block_1.odefunc.set_graph(edge_index_with_weights)
            self.ode_block_2.odefunc.set_graph(edge_index_with_weights)
            self.ode_block_3.odefunc.set_graph(edge_index_with_weights)
            self.ode_block_4.odefunc.set_graph(edge_index_with_weights)
        else:
            self.ode_block_1.odefunc.set_graph(edge_index_with_weights)
            self.ode_block_2.odefunc.set_graph(edge_index_with_weights)
            self.ode_block_3.odefunc.set_graph(edge_index_with_weights)
            self.ode_block_4.odefunc.set_graph(edge_index_with_weights)
    
    def forward(self, x, edge_index, edge_attrs):
        # Setup the ODE function with the current graph structure
        self.setup_ode_function(edge_index, edge_attrs, x)
        
        # Create list to store embeddings at each layer
        embs = [x]
        
        # Apply ODE integration layers
        if self.learnable_time:
            # Get sorted time points
            times = self.odetimes.forward()
            t1 = times[0:1]
            t2 = times[1:2]
            t3 = times[2:3]
            
            # First ODE block - from 0 to t1
            out_1 = self.ode_block_1(x, t1)
            if not self.dual_res:
                out_1 = out_1 - x
            embs.append(out_1)
            
            # Second ODE block - from t1 to t2
            out_2 = self.ode_block_2(out_1, t1, t2)
            if not self.dual_res:
                out_2 = out_2 - out_1
            embs.append(out_2)
            
            # Third ODE block - from t2 to t3
            out_3 = self.ode_block_3(out_2, t2, t3)
            if not self.dual_res:
                out_3 = out_3 - out_2
            embs.append(out_3)
            
            # Fourth ODE block - from t3 to max_time
            out_4 = self.ode_block_4(out_3, t3)
            if not self.dual_res:
                out_4 = out_4 - out_3
            embs.append(out_4)
        else:
            # Fixed time intervals
            all_emb_1 = self.ode_block_1(x)
            all_emb_1 = all_emb_1 - x
            embs.append(all_emb_1)
            
            all_emb_2 = self.ode_block_2(all_emb_1)
            all_emb_2 = all_emb_2 - all_emb_1
            embs.append(all_emb_2)
            
            all_emb_3 = self.ode_block_3(all_emb_2)
            all_emb_3 = all_emb_3 - all_emb_2
            embs.append(all_emb_3)
            
            all_emb_4 = self.ode_block_4(all_emb_3)
            all_emb_4 = all_emb_4 - all_emb_3
            embs.append(all_emb_4)
        
        # Aggregate embeddings from all layers
        embs = torch.stack(embs, dim=1)
        return torch.mean(embs, dim=1)
        
    def message(self, x_j, norm):
        # This is called during propagate() but we're using ODE integration instead
        return norm.view(-1, 1) * x_j


# ODE Function for graph dynamics
class ODEFunction(nn.Module):
    def __init__(self):
        super(ODEFunction, self).__init__()
        self.graph = None
    
    def set_graph(self, graph):
        self.graph = graph
    
    def forward(self, t, x):
        # ODE dynamics: dx/dt = -L*x where L is the graph Laplacian
        # In our case, we use the normalized adjacency matrix directly
        return torch.sparse.mm(self.graph, x)

# Fix for ODETimeSetter to ensure time parameters are strictly increasing
class ODETimeSetter(nn.Module):
    def __init__(self, time_split, max_time):
        super(ODETimeSetter, self).__init__()
        # Initialize with evenly spaced time points
        initial_times = torch.linspace(0.1 * max_time, 0.9 * max_time, time_split)
        self.time_params = nn.Parameter(initial_times)
    
    def forward(self):
        # Use softplus to ensure all times are positive, then sort to ensure strictly increasing
        positive_times = F.softplus(self.time_params)
        return torch.sort(positive_times)[0]
    
    def __getitem__(self, idx):
        times = self.forward()
        return times[idx]

# Fixed time splitter
class ODETimeSplitter:
    def __init__(self, time_split, max_time):
        self.times = torch.linspace(0, max_time, time_split+1)
    
    def __getitem__(self, idx):
        return self.times[idx]

# Base ODE Block for fixed time integration
class ODEBlock(nn.Module):
    def __init__(self, odefunc, solver, t0, t1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, x):
        # Integration from t0 to t1
        t_span = torch.tensor([self.t0, self.t1]).to(x.device)
        out = odeint(self.odefunc, x, t_span, method=self.solver)[1]
        return out

# ODE Blocks for learnable time integration
class ODEBlockTimeFirst(nn.Module):
    def __init__(self, odefunc, time_split, solver):
        super(ODEBlockTimeFirst, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        
    def forward(self, x, t1):
        t_span = torch.tensor([0.0, t1[0]]).to(x.device)
        out = odeint(self.odefunc, x, t_span, method=self.solver)[1]
        return out

class ODEBlockTimeMiddle(nn.Module):
    def __init__(self, odefunc, time_split, solver):
        super(ODEBlockTimeMiddle, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        
    def forward(self, x, t0, t1):
        t_span = torch.tensor([t0[0], t1[0]]).to(x.device)
        out = odeint(self.odefunc, x, t_span, method=self.solver)[1]
        return out

class ODEBlockTimeLast(nn.Module):
    def __init__(self, odefunc, time_split, solver, max_time):
        super(ODEBlockTimeLast, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.max_time = max_time
        
    def forward(self, x, t0):
        # Ensure t0 is less than max_time
        last_t = t0[0].item()
        end_t = max(last_t + 0.1, self.max_time) # Ensure it's strictly greater
        
        t_span = torch.tensor([last_t, end_t]).to(x.device)
        out = odeint(self.odefunc, x, t_span, method=self.solver)[1]
        return out

class RecSysGNN(nn.Module):
    def __init__(
        self,
        emb_dim,
        n_layers,
        n_users,
        n_items,
        model, # 'NGCF' or 'LightGCN' or 'DySimGCF' or 'DyODE_GCF'
        dropout=0.1, # Only used in NGCF
        device='cpu',
        self_loop=False,
        # ODE specific parameters
        solver='dopri5',
        time_split=3,
        max_time=10.0,
        learnable_time=True,
        dual_res=False
    ):
        super(RecSysGNN, self).__init__()
        assert model in ['NGCF', 'lightGCN', 'DySimGCF', 'DyODE_GCF'], 'Model must be NGCF, LightGCN, DySimGCF, or DyODE_GCF'
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.device = device
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim, dtype=torch.float32)
        
        if self.model == 'NGCF':
            self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
        elif self.model == 'lightGCN':
            self.convs = nn.ModuleList(lightGCN() for _ in range(self.n_layers))
        elif self.model == 'DySimGCF':
            self.convs = nn.ModuleList(DySimGCF(self_loop=self_loop, device=device) for _ in range(self.n_layers))
        elif self.model == 'DyODE_GCF':
            self.convs = nn.ModuleList(
                DyODE_GCF(
                    latent_dim=emb_dim,
                    solver=solver,
                    time_split=time_split,
                    max_time=max_time,
                    learnable_time=learnable_time,
                    dual_res=dual_res,
                    self_loop=self_loop,
                    device=device
                ) for _ in range(1)  # Note: For ODE, we only need one layer as it already handles multi-hop propagation
            )
        else:
            raise ValueError('Model must be NGCF, LightGCN, DySimGCF, or DyODE_GCF')
            
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
        
        if self.model == 'DyODE_GCF':
            # For ODE-based model, we only need one layer call
            # as the ODE integration handles multi-hop propagation internally
            emb = self.convs[0](x=emb, edge_index=edge_index, edge_attrs=edge_attrs)
            out = emb  # The ODE already aggregates across layers
        else:
            # Standard GNN propagation
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
    

class RecSysGNN_old(nn.Module):
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