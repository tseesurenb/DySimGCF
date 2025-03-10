'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import os
import torch
import numpy as np
import torch.nn.functional as F
import utils as ut

from tqdm import tqdm
from model import RecSysGNN, get_all_predictions
from world import config
from data_prep import get_edge_index, create_uuii_adjmat, create_uuii_adjmat_from_feature_data
from spectral_transform import construct_laplacian, spectral_transform, compute_spectral_similarities, create_edges_from_similarities

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="tseesuren-novelsoft",
    # Set the wandb project where this run will be logged.
    project="dysimgcf",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": config['lr'],
        "architecture": "DySimGCF",
        "dataset": "Ml-100k",
        "epochs": config['epochs'],
    },
)

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"
 
def compute_bpr_loss_multi_sample(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    
    if config['samples'] == 1:
        neg_reg_loss = neg_emb0.norm(2).pow(2)
    else:
        neg_reg_loss = neg_emb0.norm(2).pow(2).sum() / neg_emb0.shape[1]  # Sum over negatives and average by N
        
    # Compute regularization loss
    reg_loss = (1 / 2) * (
        user_emb0.norm(2).pow(2) + 
        pos_emb0.norm(2).pow(2)  +
        neg_reg_loss
    ) / float(len(users))
    
    # Compute positive and negative scores
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    
    if config['samples'] == 1:
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))  # Using softplus for stability
    else:
        # Neg scores for each user and N negative items: [batch_size, N]
        neg_scores = torch.mul(users_emb.unsqueeze(1), neg_emb).sum(dim=2)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores.unsqueeze(1)))  # Using softplus for stability
        
    return bpr_loss, reg_loss

def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    
    # Compute regularization loss
    reg_loss = (1 / 2) * (
        user_emb0.norm(2).pow(2) + 
        pos_emb0.norm(2).pow(2)  +
        neg_emb0.norm(2).pow(2)
    ) / float(len(users))
    
    # Compute positive and negative scores
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))  # Using softplus for stability
        
    return bpr_loss, reg_loss, None

# Compute BPR loss and Contrastive Loss
def compute_bpr_loss_contrast(users, users_emb, pos_emb, neg_emb, 
                                 user_emb0, pos_emb0, neg_emb0, 
                                 temperature=0.1, lambda_contrastive=0.05):
    
    # Compute BPR loss (ranking)
    reg_loss = (1 / 2) * (
        user_emb0.norm(2).pow(2) + 
        pos_emb0.norm(2).pow(2)  +
        neg_emb0.norm(2).pow(2)
    ) / float(len(users))
    
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)  # Positive interactions
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)  # Negative interactions
    
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))  # BPR loss
    
    # Compute Contrastive Loss (InfoNCE)
    pos_sim = F.cosine_similarity(users_emb, pos_emb) / temperature  # Positive pair similarity
    neg_sim = F.cosine_similarity(users_emb, neg_emb) / temperature  # Negative pair similarity
    
    # Contrastive loss using InfoNCE (pull positives, push negatives)
    contrastive_loss = -torch.mean(torch.log(
        torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))
    ))
    
    # Final loss = BPR Loss + Contrastive Loss (weighted by λ)
    #total_loss = bpr_loss + lambda_contrastive * contrastive_loss
    
    return bpr_loss, reg_loss, contrastive_loss


def train_and_eval(model, optimizer, train_df, test_df, edge_index, edge_attrs, adj_list, device, g_seed):
   
    epochs = config['epochs']
    b_size = config['batch_size']
    topK = config['top_K']
    decay = config['decay']
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    
    losses = { 'bpr_loss': [], 'reg_loss': [], 'total_loss': [] }
    metrics = { 'recall': [], 'precision': [], 'f1': [], 'ncdg': [] }
        
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    
    v = torch.ones(len(train_df), dtype=torch.float32).to(device)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()
    
    max_ncdg = 0.0
    max_recall = 0.0
    max_prec = 0.0
    max_epoch = 0
    
    neg_sample_time = 0.0
    
    pbar = tqdm(range(epochs), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    
    for epoch in pbar:
    
        total_losses, bpr_losses, reg_losses, contrast_losses  = [], [], [], []
        
        # Shuffle the DataFrame
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        if config['samples'] == 1:
            S = ut.neg_uniform_sample(train_df, adj_list, n_users)
        else:
            S = ut.multiple_neg_uniform_sample(train_df, adj_list, n_users)
         
        users = torch.Tensor(S[:, 0]).long().to(device)
        pos_items = torch.Tensor(S[:, 1]).long().to(device)
        neg_items = torch.Tensor(S[:, 2]).long().to(device)
                
        if config['shuffle']: 
            users, pos_items, neg_items = ut.shuffle(users, pos_items, neg_items)
        
        n_batches = len(users) // b_size + 1
        
        if epoch % config["epochs_per_eval"] == 0:
            model.eval()
            with torch.no_grad():
                _, out = model(edge_index, edge_attrs)
                final_u_emb, final_i_emb = torch.split(out, (n_users, n_items))
                recall,  prec, ncdg = ut.get_metrics(final_u_emb, final_i_emb, test_df, topK, interactions_t, device)
            
            if ncdg > max_ncdg:
                max_ncdg = ncdg
                max_recall = recall
                max_prec = prec
                max_epoch = epoch
            
            pbar.set_postfix_str(f"prec {br}{prec:.4f}{rs} | recall {br}{recall:.4f}{rs} | ncdg {br}{ncdg:.4f} ({max_ncdg:.4f}, {max_recall:.4f}, {max_prec:.4f} at {max_epoch}) {rs}")
            pbar.refresh()
                                
        model.train()
        for (b_i, (b_users, b_pos, b_neg)) in enumerate(ut.minibatch(users, pos_items, neg_items, batch_size=b_size)):
                                     
            u_emb, pos_emb, neg_emb, u_emb0,  pos_emb0, neg_emb0 = model.encode_minibatch(b_users, b_pos, b_neg, edge_index, edge_attrs)
            bpr_loss, reg_loss, contrast_loss = compute_bpr_loss(b_users, u_emb, pos_emb, neg_emb, u_emb0,  pos_emb0, neg_emb0)
            
            reg_loss = decay * reg_loss
            
            if contrast_loss != None:
                lambda_contrastive = 0.05
                total_loss = bpr_loss + reg_loss + lambda_contrastive * contrast_loss
            else:
                total_loss = bpr_loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bpr_losses.append(bpr_loss.detach().item())
            reg_losses.append(reg_loss.detach().item())
            total_losses.append(total_loss.detach().item())
            
            if contrast_loss != None:
                contrast_losses.append(contrast_loss.detach().item())
            
            # Update the description of the outer progress bar with batch information
            pbar.set_description(f"{config['model']}({g_seed:2}) | #ed {len(edge_index[0]):6} | ep({epochs}) {epoch} | ba({n_batches}) {b_i:3} | loss {total_loss.detach().item():.4f}")
        
        f1 = (2 * recall * prec / (recall + prec)) if (recall + prec) != 0 else 0.0
        
        metrics['recall'].append(round(recall,4))
        metrics['precision'].append(round(prec,4))
        metrics['f1'].append(round(f1,4))
        metrics['ncdg'].append(round(ncdg,4))
        
        losses['bpr_loss'].append(round(np.mean(bpr_losses), 4) if bpr_losses else np.nan)
        losses['reg_loss'].append(round(np.mean(reg_losses), 4) if reg_losses else np.nan)
        losses['total_loss'].append(round(np.mean(total_losses), 4) if total_losses else np.nan)
        
        if contrast_loss != None:
            run.log({"ncdg": ncdg, "recall@20": recall, "reg_loss": np.mean(reg_losses), "contrast_loss": np.mean(contrast_losses),  "bpr_loss": np.mean(bpr_losses), "total_loss": np.mean(total_losses),})
        else:
            run.log({"ncdg": ncdg, "recall@20": recall, "reg_loss": np.mean(reg_losses), "bpr_loss": np.mean(bpr_losses), "total_loss": np.mean(total_losses),})
            
    return (losses, metrics)

def exec_exp(orig_train_df, orig_test_df, exp_n = 1, g_seed=42, device='cpu', verbose = -1):
    
    _test_df = orig_test_df[
      (orig_test_df['user_id'].isin(orig_train_df['user_id'].unique())) & \
      (orig_test_df['item_id'].isin(orig_train_df['item_id'].unique()))
    ]
    
    _train_df, _test_df = ut.encode_ids(orig_train_df, _test_df)
        
    N_USERS = _train_df['user_id'].nunique()
    N_ITEMS = _train_df['item_id'].nunique()
    
    if verbose >= 0:
        print(f"dataset: {br}{config['dataset']} {rs}| seed: {g_seed} | exp: {exp_n} | device: {device}")
        print(f"{br}Trainset{rs} | #users: {N_USERS}, #items: {N_ITEMS}, #interactions: {len(_train_df)}")
        print(f" {br}Testset{rs} | #users: {_test_df['user_id'].nunique()}, #items: {_test_df['item_id'].nunique()}, #interactions: {len(_test_df)}")
      
    adj_list = ut.make_adj_list(_train_df) # adj_list is a user dictionary with a list of positive items (pos_items) and negative items (neg_items)
     
    if config['edge'] == 'bi': # edge from a bipartite graph
        
        u_t = torch.LongTensor(_train_df.user_id)
        i_t = torch.LongTensor(_train_df.item_id) + N_USERS
    
        bi_edge_index = torch.stack((
            torch.cat([u_t, i_t]),
            torch.cat([i_t, u_t])
        )).to(device)
        
        edge_index = bi_edge_index.to(device)
        edge_attrs = None
        
        item_sim_mat = None
         
    if config['edge'] == 'knn': # edge from a k-nearest neighbor or similarity graph
        
        if config['sim'] == 'ind':
            knn_train_adj_df = create_uuii_adjmat(_train_df, verbose)
        elif config['sim'] == 'trans':  
            knn_train_adj_df = create_uuii_adjmat_from_feature_data(_train_df, verbose)
        else:
            print(f"{br}Invalid sim mode{rs}")
            return
        
        knn_edge_index, knn_edge_attrs = get_edge_index(knn_train_adj_df)
        knn_edge_index = torch.tensor(knn_edge_index).to(device).long()
                    
        edge_index = knn_edge_index.to(device)
        edge_attrs = torch.tensor(knn_edge_attrs).to(device)
        
    if config['edge'] == 'spectral': # edge from a spectral graph
        # Execute spectral domain processing
        laplacian, adjacency = construct_laplacian(_train_df, N_USERS, N_ITEMS)
        
        # Get parameters from config
        k_eigenvectors = config['eigen_K']
        k_users = config['u_K']
        k_items = config['i_K']
        similarity_type = config.get('similarity_type', 'cosine')
        
        eigenvalues, eigenvectors = spectral_transform(laplacian, k=k_eigenvectors)
        
        # Apply anisotropic diffusion if enabled (from S-Diff approach)
        if config.get('use_anisotropic', False):
            print(f"{bg}Applying anisotropic diffusion in spectral domain{rs}")
            # Prepare eigenvalues for anisotropic diffusion
            # Lower eigenvalues correspond to lower frequencies that we want to preserve
            alpha_min = config.get('alpha_min', 0.1)  # Minimum preservation factor
            sigma_max = config.get('sigma_max', 0.5)  # Maximum noise level
            
            # Use scheduled noise based on eigenvalues (following S-Diff approach)
            alpha_t = (1 - alpha_min) * np.exp(-eigenvalues) + alpha_min
            sigma_t = np.minimum(np.sqrt(1 - alpha_t**2), sigma_max)
            
            # Apply anisotropic diffusion in spectral domain
            noise = np.random.normal(0, 1, eigenvectors.shape)
            diffused_eigenvectors = alpha_t * eigenvectors + sigma_t[:, np.newaxis] * noise
            eigenvectors = diffused_eigenvectors
        
        user_similarity, item_similarity = compute_spectral_similarities(
            eigenvectors, N_USERS, N_ITEMS, similarity_type=similarity_type
        )
        
        # Create edges from computed similarities
        spectral_edge_index, spectral_edge_attrs = create_edges_from_similarities(
            user_similarity, item_similarity, k_users, k_items, N_USERS
        )
        
        edge_index = torch.tensor(spectral_edge_index).to(device).long()
        edge_attrs = torch.tensor(spectral_edge_attrs).to(device).float()
        
        print(f"{bg}Spectral processing complete. Created {len(edge_attrs)} edges{rs}")
            
    
    cf_model = RecSysGNN(model=config['model'], 
                         emb_dim=config['emb_dim'],  
                         n_layers=config['layers'], 
                         n_users=N_USERS, 
                         n_items=N_ITEMS,
                         self_loop=config['self_loop'],
                         device = device).to(device)
    
    opt = torch.optim.Adam(cf_model.parameters(), lr=config['lr'])

    model_file_path = f"./models/params/{config['model']}_{config['dataset']}_{config['edge']}"
    
    if config['load'] and os.path.exists(model_file_path):
        cf_model.load_state_dict(torch.load(model_file_path, weights_only=True))

    losses, metrics = train_and_eval(cf_model, 
                                     opt, 
                                     _train_df,
                                     _test_df, 
                                     edge_index, 
                                     edge_attrs,
                                     adj_list,
                                     device,
                                     g_seed)
   

    # Assume 'model' is your PyTorch model
    if config['save_model']:
        torch.save(cf_model.state_dict(), model_file_path)
    # make all predictions for all users and items
    if config['save_pred']:
        predictions = get_all_predictions(cf_model, edge_index, edge_attrs, device)
        #save predictions to a file
        np.save(f"./models/preds/{config['model']}_{config['dataset']}_{config['batch_size']}__{config['layers']}_{config['edge']}", predictions)

    run.finish()
    
    return losses, metrics