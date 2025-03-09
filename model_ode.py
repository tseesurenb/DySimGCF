'''
Created on Sep 1, 2024
Pytorch Implementation of DySimODE: A Continuous-Time Neural ODE Version of Similarity-Centric Graph Networks for Adaptive Collaborative Filtering
'''

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch_geometric.utils import softmax
from torchdiffeq import odeint
from world import config

class DySimODEFunc(nn.Module):
    """ODE function for the DySimODE model that defines how features evolve over time"""
    def __init__(self, self_loop=False, device='cpu', **kwargs):
        super().__init__()
        
        self.device = device
        self.add_self_loops = self_loop
        self.edge_index = None
        self.graph_norms = None
        
    def set_graph(self, edge_index, edge_attrs):
        """Set the graph structure and precompute norms"""
        self.edge_index = edge_index
        
        # Compute normalized weights similar to the original GCN
        from_, to_ = edge_index
        
        # Compute softmax normalization like in the original model
        incoming_norm = softmax(edge_attrs, to_)
        outgoing_norm = softmax(edge_attrs, from_)
        
        # Choose normalization based on ablation study
        if config['abl_study'] == -1:
            norm = outgoing_norm
        elif config['abl_study'] == 1:
            norm = incoming_norm
        else:
            norm = torch.sqrt(incoming_norm * outgoing_norm)
            
        self.graph_norms = norm
    
    def forward(self, t, x):
        """Define the ODE dynamics: dx/dt = f(x, t)"""
        # Prepare result container
        result = torch.zeros_like(x).to(self.device)
        
        # Propagate messages according to graph structure
        from_, to_ = self.edge_index
        for i in range(from_.size(0)):
            # Get source node, target node, and normalized weight
            source_idx = from_[i]
            target_idx = to_[i]
            weight = self.graph_norms[i]
            
            # Add weighted message from source to target
            result[target_idx] += weight * x[source_idx]
        
        # Add self-loops if specified
        if self.add_self_loops:
            result += x
            
        return result

class DySimODE(nn.Module):
    def __init__(self, self_loop=False, device='cpu', **kwargs):
        super().__init__()
        
        # ODE function that defines the dynamics
        self.ode_func = DySimODEFunc(self_loop, device, **kwargs)
        self.device = device
        self.integration_time = torch.tensor([0, 1]).to(device)
        
    def forward(self, x, edge_index, edge_attrs):
        """
        x: Node features [num_nodes, feature_dim]
        edge_index: Graph connectivity [2, num_edges]
        edge_attrs: Edge attributes/weights [num_edges]
        """
        # Store graph structure in the ODE function
        self.ode_func.set_graph(edge_index, edge_attrs)
        
        # Solve ODE to obtain the final representations
        solution = odeint(
            self.ode_func, 
            x, 
            self.integration_time, 
            method='dopri5',
            rtol=1e-3,
            atol=1e-3
        )
        
        # Return the final state (at t=1)
        return solution[-1]

class NGCFConv(nn.Module):
    """Original NGCF convolutional layer (kept for comparison)"""
    def __init__(self, emb_dim, dropout=0.0):
        super(NGCFConv, self).__init__()
        
        self.dropout = dropout
        
        # Weight matrices for NGCF message propagation
        self.W1 = nn.Linear(emb_dim, emb_dim, bias=True)
        self.W2 = nn.Linear(emb_dim, emb_dim, bias=True)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
    
    def forward(self, x, edge_index, edge_attrs):
        from_, to_ = edge_index
        
        # Element-wise product embeddings for each edge
        x_i = x[from_]
        x_j = x[to_]
        
        # Normalization based on degree
        # Calculate degrees for each node in the graph
        deg = torch.zeros(x.size(0), device=x.device)
        deg = deg.scatter_add(0, to_, torch.ones_like(to_, dtype=torch.float32))
        deg = torch.sqrt(deg)
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        
        # Normalize edge weights
        norm = deg_inv[from_] * deg_inv[to_]
        norm = norm.view(-1, 1)
        
        # Message computation
        m_j = norm * (x_j + self.W1(x_j * x_i))
        
        # Aggregate messages
        output = torch.zeros_like(x)
        output = output.scatter_add(0, to_.view(-1, 1).expand(-1, x.size(1)), m_j)
        
        # Apply activation and dropout
        output = F.leaky_relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output

class lightGCN(nn.Module):
    """Original LightGCN layer (kept for comparison)"""
    def __init__(self):
        super(lightGCN, self).__init__()
    
    def forward(self, x, edge_index, edge_attrs=None):
        from_, to_ = edge_index
        
        # Normalization based on degree
        deg = torch.zeros(x.size(0), device=x.device)
        deg = deg.scatter_add(0, to_, torch.ones_like(to_, dtype=torch.float32))
        deg = torch.sqrt(deg)
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        
        # Normalize edge weights
        norm = deg_inv[from_] * deg_inv[to_]
        norm = norm.view(-1, 1)
        
        # Message passing
        m_j = norm * x[from_]
        
        # Aggregate messages
        output = torch.zeros_like(x)
        output = output.scatter_add(0, to_.view(-1, 1).expand(-1, x.size(1)), m_j)
        
        return output

class RecSysGNN(nn.Module):
    def __init__(
        self,
        emb_dim, 
        n_layers,
        n_users,
        n_items,
        model, # 'NGCF' or 'LightGCN' or 'DySimODE'
        dropout=0.1, # Only used in NGCF
        device='cpu',
        self_loop=False
    ):
        super(RecSysGNN, self).__init__()

        assert model in ['NGCF', 'lightGCN', 'DySimGCF', 'DySimODE'], 'Model must be NGCF, LightGCN, DySimGCF, or DySimODE'
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
        elif self.model == 'DySimODE':
            # For ODE model, we only need one module since depth is controlled by integration time
            self.ode_model = DySimODE(self_loop=self_loop, device=device)
        else:
            raise ValueError('Model must be NGCF, LightGCN, DySimGCF or DySimODE')
        
        self.init_parameters()

    def init_parameters(self):
        if self.model == 'NGCF':
            nn.init.xavier_uniform_(self.embedding.weight, gain=1)
        else:
            nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index, edge_attrs):
        emb0 = self.embedding.weight
        
        if self.model == 'DySimODE':
            # For ODE model, we integrate through continuous time
            # The integration depth is determined by the integration time
            out = self.ode_model(emb0, edge_index, edge_attrs)
            return emb0, out
        else:
            # For traditional GNN models, we use discrete layers
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

# For backward compatibility, keep the DySimGCF class for comparison
class DySimGCF(nn.Module):
    def __init__(self, self_loop=False, device='cpu', **kwargs):  
        super().__init__()
        
        self.graph_norms = None
        self.edge_attrs = None
        self.add_self_loops = self_loop
        self.device = device
        
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
        
        # Manual message passing implementation
        result = torch.zeros_like(x).to(self.device)
        from_, to_ = edge_index
        
        for i in range(from_.size(0)):
            source_idx = from_[i]
            target_idx = to_[i]
            weight = self.graph_norms[i]
            result[target_idx] += weight * x[source_idx]
            
        return result