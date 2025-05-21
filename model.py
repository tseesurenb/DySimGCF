'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import torch
import torch.nn.functional as F
from torch import nn # Ensure nn is imported
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, degree # Ensure degree is imported
from world import config # Assuming 'world' and 'config' are correctly set up

# Then import it in your code
# from torchdiffeq import odeint # This import seems unused in the provided GNN code, can be removed if not needed elsewhere


# NGCF Convolutional Layer (Copied from user's context for completeness)
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
    if self.lin_1.bias is not None:
        nn.init.zeros_(self.lin_1.bias)
    if self.lin_2.bias is not None:
        nn.init.zeros_(self.lin_2.bias)


  def forward(self, x, edge_index, edge_attrs, scale): # scale seems unused here
    # Compute normalization
    from_, to_ = edge_index
    deg_val = degree(to_, x.size(0), dtype=x.dtype) # Renamed to avoid conflict
    deg_inv_sqrt = deg_val.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=norm)

    # Perform update after aggregation
    out += self.lin_1(x) # This line might be specific to NGCF's interpretation of residual connection
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)

  def message(self, x_j, x_i, norm):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) 

# LightGCN Convolutional Layer (Copied from user's context for completeness)
class lightGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        self.norm_cache = None # Renamed to avoid conflict with message's norm
            
    def forward(self, x, edge_index, edge_attrs): # edge_attrs seems unused here
        if self.norm_cache is None or self.norm_cache.device != x.device or self.norm_cache.size(0) != edge_index.size(1) : # Basic cache check
          from_, to_ = edge_index
          deg_val = degree(to_, x.size(0), dtype=x.dtype) # Renamed
          deg_inv_sqrt = deg_val.pow(-0.5)
          deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
          self.norm_cache = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        return self.propagate(edge_index, x=x, norm=self.norm_cache)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# DySimGCF (Copied from user's context for completeness - assuming it's a separate model option)
class DySimGCF(MessagePassing):
    def __init__(self, self_loop=False, device='cpu', dropout_rate=0.5, **kwargs):
        super().__init__(aggr='add')
        self.graph_norms = None
        # self.edge_attrs = None # This was in original user code, but not used.
        self.add_self_loops = self_loop # Not used in current forward logic
        self.last_edge_key = None
        self.dropout_rate = dropout_rate # This is for edge dropout
        
    def forward(self, x, edge_index, edge_attrs):
        edge_index_prop, edge_attrs_prop = edge_index, edge_attrs # For clarity
        norm_to_propagate = None

        if self.training:
            if self.dropout_rate > 0 and edge_index.numel() > 0:
                edge_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.dropout_rate
                edge_index_prop = edge_index[:, edge_mask]
                edge_attrs_prop = edge_attrs[edge_mask]
            
            if edge_index_prop.numel() > 0 : # Only compute norm if there are edges
                from_, to_ = edge_index_prop
                temperature = config['s_temp'] # Assumes global config
                num_nodes = x.size(self.node_dim)
                incoming_norm = softmax(edge_attrs_prop / temperature, to_, num_nodes=num_nodes)
                outgoing_norm = softmax(edge_attrs_prop / temperature, from_, num_nodes=num_nodes)
                
                if config['abl_study'] == -1:
                    norm_to_propagate = outgoing_norm
                elif config['abl_study'] == 1:
                    norm_to_propagate = incoming_norm
                else:
                    norm_to_propagate = torch.sqrt(incoming_norm * outgoing_norm)
            # else norm_to_propagate remains None
            
        else: # Evaluation mode
            current_edge_key = f"{edge_index.data_ptr()}_{edge_attrs.data_ptr()}"
            if self.graph_norms is None or current_edge_key != self.last_edge_key:
                if edge_index_prop.numel() > 0:
                    from_, to_ = edge_index_prop # Use edge_index_prop (which is edge_index in eval)
                    temperature = config['s_temp']
                    num_nodes = x.size(self.node_dim)
                    incoming_norm = softmax(edge_attrs_prop / temperature, to_, num_nodes=num_nodes) # Use edge_attrs_prop
                    outgoing_norm = softmax(edge_attrs_prop / temperature, from_, num_nodes=num_nodes)
                    
                    if config['abl_study'] == -1:
                        self.graph_norms = outgoing_norm
                    elif config['abl_study'] == 1:
                        self.graph_norms = incoming_norm
                    else:
                        self.graph_norms = torch.sqrt(incoming_norm * outgoing_norm)
                    self.last_edge_key = current_edge_key
                else: # No edges, no norm to cache
                    self.graph_norms = None 
                    self.last_edge_key = current_edge_key # Update key even if norm is None

            norm_to_propagate = self.graph_norms
        
        # Propagate. If norm_to_propagate is None (no edges), message passing effectively does nothing or returns zeros.
        return self.propagate(edge_index_prop, x=x, norm=norm_to_propagate)
        
    def message(self, x_j, norm):
        if norm is None: # Handle case where no edges led to norm being None
            return torch.zeros_like(x_j)
        return norm.view(-1, 1) * x_j

# trying to add sharpening part to DySimGCF
class DySimGCF_sharp(MessagePassing):
    def __init__(self, self_loop=False, device='cpu', dropout_rate=0.5, 
                 enable_sharpening=True, sharpening_strength=1.0, **kwargs):
        super().__init__(aggr='add')
        self.graph_norms = None
        self.add_self_loops = self_loop
        self.last_edge_key = None
        self.dropout_rate = dropout_rate
        
        # Sharpening parameters
        self.enable_sharpening = enable_sharpening
        self.sharpening_strength = sharpening_strength
    
    def forward(self, x, edge_index, edge_attrs):
        # Step 1: Blurring process (your original implementation)
        edge_index_prop, edge_attrs_prop = edge_index, edge_attrs
        norm_to_propagate = None
        
        if self.training:
            if self.dropout_rate > 0 and edge_index.numel() > 0:
                edge_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.dropout_rate
                edge_index_prop = edge_index[:, edge_mask]
                edge_attrs_prop = edge_attrs[edge_mask]
                
            if edge_index_prop.numel() > 0:
                from_, to_ = edge_index_prop
                temperature = config['s_temp']
                num_nodes = x.size(self.node_dim)
                incoming_norm = softmax(edge_attrs_prop / temperature, to_, num_nodes=num_nodes)
                outgoing_norm = softmax(edge_attrs_prop / temperature, from_, num_nodes=num_nodes)
                
                if config['abl_study'] == -1:
                    norm_to_propagate = outgoing_norm
                elif config['abl_study'] == 1:
                    norm_to_propagate = incoming_norm
                else:
                    norm_to_propagate = torch.sqrt(incoming_norm * outgoing_norm)
        else:
            current_edge_key = f"{edge_index.data_ptr()}_{edge_attrs.data_ptr()}"
            if self.graph_norms is None or current_edge_key != self.last_edge_key:
                if edge_index_prop.numel() > 0:
                    from_, to_ = edge_index_prop
                    temperature = config['s_temp']
                    num_nodes = x.size(self.node_dim)
                    incoming_norm = softmax(edge_attrs_prop / temperature, to_, num_nodes=num_nodes)
                    outgoing_norm = softmax(edge_attrs_prop / temperature, from_, num_nodes=num_nodes)
                    
                    if config['abl_study'] == -1:
                        self.graph_norms = outgoing_norm
                    elif config['abl_study'] == 1:
                        self.graph_norms = incoming_norm
                    else:
                        self.graph_norms = torch.sqrt(incoming_norm * outgoing_norm)
                    self.last_edge_key = current_edge_key
                else:
                    self.graph_norms = None
                    self.last_edge_key = current_edge_key
            norm_to_propagate = self.graph_norms
        
        # Propagate (blurring process)
        blurred_output = self.propagate(edge_index_prop, x=x, norm=norm_to_propagate)
        
        # Step 2: Sharpening process
        if self.enable_sharpening:
            # Implement sharpening as negative propagation to emphasize differences
            # This is based on s(S(t)) = -S(t)P̃ from the paper
            sharp_norm_to_propagate = None
            
            if norm_to_propagate is not None:
                # For sharpening, we use negative values to emphasize differences
                sharp_norm_to_propagate = -self.sharpening_strength * norm_to_propagate
                
            # Apply sharpening process (negatively weighted propagation)
            sharpening_output = self.propagate(edge_index_prop, x=blurred_output, norm=sharp_norm_to_propagate)
            
            # Combine blurred and sharpened outputs
            return blurred_output + sharpening_output
        else:
            return blurred_output
    
    def message(self, x_j, norm):
        if norm is None:
            return torch.zeros_like(x_j)
        return norm.view(-1, 1) * x_j

class RecSysGNN(nn.Module):
  def __init__(
      self,
      emb_dim, 
      n_layers,
      n_users,
      n_items,
      model, 
      dropout=0.1, 
      device = 'cpu',
      self_loop = False # self_loop for DySimGCF or JGCF_DySimGCF if they use it
  ):
    super(RecSysGNN, self).__init__()

    assert model in ['NGCF', 'lightGCN', 'DySimGCF'], \
        'Model must be NGCF, LightGCN, or DySimGCF'
    self.model = model
    self.n_users = n_users
    self.n_items = n_items
    self.n_layers = n_layers
    self.emb_dim = emb_dim
      
    self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim) # dtype removed, defaults to float32
        
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
    elif self.model == 'lightGCN':
      self.convs = nn.ModuleList(lightGCN() for _ in range(self.n_layers))
    elif self.model == 'DySimGCF':
      # Pass relevant params from config or RecSysGNN args to DySimGCF
      self.convs = nn.ModuleList(
          DySimGCF(
              self_loop=self_loop, 
              device=device, 
              dropout_rate=config.get('dsm_dropout_rate', 0.5) # Example: use a specific config key
          ) for _ in range(self.n_layers)
      )
    
    self.init_parameters()

  def init_parameters(self):
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else: # For LightGCN, DySimGCF, JGCF_DySimGCF
      nn.init.normal_(self.embedding.weight, std=0.1)
    # Initialization for convs' own parameters (like JGCF_DySimGCF.final_proj) happens in their respective classes

  def forward(self, edge_index, edge_attrs):
    emb0 = self.embedding.weight
    embs = [emb0]
    current_emb = emb0

    for conv_layer in self.convs:
      if self.model == 'NGCF': # NGCFConv has a different signature
          current_emb = conv_layer(x=current_emb, edge_index=edge_index, edge_attrs=edge_attrs, scale=None) # Assuming scale isn't critical or handled internally
      else: # lightGCN, DySimGCF, JGCF_DySimGCF
          current_emb = conv_layer(x=current_emb, edge_index=edge_index, edge_attrs=edge_attrs)
      embs.append(current_emb)
      
    if self.model == 'NGCF':
      # NGCF concatenates all layer embeddings including the initial one
      out = torch.cat(embs, dim=-1) 
    else:
      # LightGCN, DySimGCF, JGCF_DySimGCF typically average layer embeddings
      # Due to the fix in JGCF_DySimGCF, all embs will have the same feature dimension (self.emb_dim)
      out = torch.mean(torch.stack(embs, dim=0), dim=0)
        
    return emb0, out # Return initial embeddings and final aggregated embeddings

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
    # items in RecSysGNN usually refers to item indices relative to the start of item embeddings
    # e.g., if items are 0 to n_items-1, they map to n_users to n_users+n_items-1 in the full embedding table
    emb0, out = self(edge_index, edge_attrs)    
    user_embeds = out[users]
    item_embeds = out[items + self.n_users] # Adjust item indices for full embedding table
    return torch.matmul(user_embeds, item_embeds.t())
  

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