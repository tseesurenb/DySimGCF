'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import os
import torch
import numpy as np
import torch.nn.functional as F
import utils as ut
import sys

from tqdm import tqdm
from model import RecSysGNN, get_all_predictions
from world import config
from data_prep import get_edge_index, create_uuii_adjmat, create_uuii_adjmat_from_feature_data
import wandb

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

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


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    margin = config['margin']
    negative_weight = config['l_weight']
    
    # Normalize embeddings
    users_emb_norm = F.normalize(users_emb, p=2, dim=1)
    pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)
    
    # Calculate positive similarity
    pos_similarity = F.cosine_similarity(users_emb_norm, pos_emb_norm, dim=1)
    
    # Prepare negative similarities
    if neg_emb.dim() == 2:  # Single negative sample case
        neg_emb_norm = F.normalize(neg_emb, p=2, dim=1)
        neg_similarity = F.cosine_similarity(users_emb_norm, neg_emb_norm, dim=1).unsqueeze(1)
    else:  # Multiple negative samples case
        neg_emb_norm = F.normalize(neg_emb, p=2, dim=2)
        users_emb_norm_expanded = users_emb_norm.unsqueeze(1)
        neg_similarity = F.cosine_similarity(users_emb_norm_expanded, neg_emb_norm, dim=2)
    
    # Combine positive and negative similarities to match y_pred format in CosineContrastiveLoss
    y_pred = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
    
    # Create dummy y_true tensor (not actually used in the loss calculation)
    y_true = torch.zeros_like(y_pred)
    
    # Follow the exact same structure as CosineContrastiveLoss.forward()
    pos_logits = y_pred[:, 0]
    pos_loss = torch.relu(1 - pos_logits)
    neg_logits = y_pred[:, 1:]
    neg_loss = torch.relu(neg_logits - margin)
    
    if negative_weight:
        loss = pos_loss + neg_loss.mean(dim=-1) * negative_weight
    else:
        loss = pos_loss + neg_loss.sum(dim=-1)
    
    ccl_loss = loss.mean()
    
    # Regularization
    user_reg_loss_sum = user_emb0.norm(2).pow(2)
    pos_reg_loss_sum = pos_emb0.norm(2).pow(2)
    
    if neg_emb0.dim() == 2:  # Single negative
        neg_reg_loss_component = neg_emb0.norm(2).pow(2)
    else:  # Multiple negatives
        sum_sq_norms = neg_emb0.norm(2, dim=2).pow(2).sum()
        if negative_weight:  # Match the logic for loss calculation
            neg_reg_loss_component = sum_sq_norms / neg_emb0.size(1)
        else:
            neg_reg_loss_component = sum_sq_norms
    
    reg_loss = (1 / 2) * (user_reg_loss_sum + pos_reg_loss_sum + neg_reg_loss_component) / float(len(users))
    
    return ccl_loss, reg_loss, None

# similar but not exactly same
def compute_bpr_loss_2(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    ccl_margin = config['margin']
    ccl_weight_w = config['l_weight']
    num_negative_samples = config['samples']
    
    # Normalize embeddings
    users_emb_norm = F.normalize(users_emb, p=2, dim=1)
    pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)
    
    # Positive pair similarity
    pos_similarity = F.cosine_similarity(users_emb_norm, pos_emb_norm, dim=1)
    
    # Directly use torch.relu for positive loss (matches original class)
    pos_loss = torch.relu(1 - pos_similarity)
    
    # Handle negative samples (similar to original class structure)
    if num_negative_samples == 1:
        neg_emb_norm = F.normalize(neg_emb, p=2, dim=1)
        neg_similarity = F.cosine_similarity(users_emb_norm, neg_emb_norm, dim=1)
        neg_loss = torch.relu(neg_similarity - ccl_margin)
        
        # Follow the original class logic
        if ccl_weight_w:
            ccl_loss_per_sample = pos_loss + neg_loss * ccl_weight_w
        else:
            ccl_loss_per_sample = pos_loss + neg_loss
            
    else:  # Multiple negative samples
        neg_emb_norm = F.normalize(neg_emb, p=2, dim=2)
        users_emb_norm_expanded = users_emb_norm.unsqueeze(1)
        neg_similarity = F.cosine_similarity(users_emb_norm_expanded, neg_emb_norm, dim=2)
        neg_loss = torch.relu(neg_similarity - ccl_margin)
        
        # Follow the original class logic
        if ccl_weight_w:
            ccl_loss_per_sample = pos_loss + neg_loss.mean(dim=1) * ccl_weight_w
        else:
            ccl_loss_per_sample = pos_loss + neg_loss.sum(dim=1)
    
    # Mean over batch
    ccl_loss = ccl_loss_per_sample.mean()
    
    # Regularization (unchanged)
    user_reg_loss_sum = user_emb0.norm(2).pow(2)
    pos_reg_loss_sum = pos_emb0.norm(2).pow(2)
    
    if num_negative_samples == 1:
        neg_reg_loss_component = neg_emb0.norm(2).pow(2)
    else:
        sum_sq_norms_all_neg_samples = neg_emb0.norm(2, dim=2).pow(2).sum()
        neg_reg_loss_component = sum_sq_norms_all_neg_samples / num_negative_samples

    reg_loss = (1 / 2) * (user_reg_loss_sum + pos_reg_loss_sum + neg_reg_loss_component) / float(len(users))

    return ccl_loss, reg_loss, None

def compute_bpr_loss_gemini(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    ccl_margin = config['margin']
    ccl_weight_w = config['l_weight']
    num_negative_samples = config['samples']
    batch_size = users_emb.shape[0]

    # Normalize embeddings
    users_emb_norm = F.normalize(users_emb, p=2, dim=1)
    pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)
    
    # Positive pair similarity
    pos_similarity = F.cosine_similarity(users_emb_norm, pos_emb_norm, dim=1)
    positive_loss_term = 1 - pos_similarity

    # Handle negative samples
    if num_negative_samples == 1:
        neg_emb_norm = F.normalize(neg_emb, p=2, dim=1)
        neg_similarity = F.cosine_similarity(users_emb_norm, neg_emb_norm, dim=1)
        individual_negative_terms = F.relu(neg_similarity - ccl_margin)
        avg_sum_negative_terms = individual_negative_terms
    else:
        neg_emb_norm = F.normalize(neg_emb, p=2, dim=2)
        users_emb_norm_expanded = users_emb_norm.unsqueeze(1)
        neg_similarity = F.cosine_similarity(users_emb_norm_expanded, neg_emb_norm, dim=2)
        individual_negative_terms = F.relu(neg_similarity - ccl_margin)
        sum_of_negative_terms = torch.sum(individual_negative_terms, dim=1)
        avg_sum_negative_terms = sum_of_negative_terms / num_negative_samples

    # Weight and combine loss terms
    weighted_avg_negative_loss_term = ccl_weight_w * avg_sum_negative_terms
    ccl_loss_per_sample = positive_loss_term + weighted_avg_negative_loss_term
    ccl_loss = torch.mean(ccl_loss_per_sample)

    # Regularization
    user_reg_loss_sum = user_emb0.norm(2).pow(2)
    pos_reg_loss_sum = pos_emb0.norm(2).pow(2)
    
    if num_negative_samples == 1:
        neg_reg_loss_component = neg_emb0.norm(2).pow(2)
    else:
        sum_sq_norms_all_neg_samples = neg_emb0.norm(2, dim=2).pow(2).sum()
        neg_reg_loss_component = sum_sq_norms_all_neg_samples / num_negative_samples

    reg_loss = (1 / 2) * (user_reg_loss_sum + pos_reg_loss_sum + neg_reg_loss_component) / float(len(users))

    return ccl_loss, reg_loss, None


# multi neg sample + margin
def compute_bpr_loss_v1(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0, margin=0.1):
    """
    Compute BPR loss with a margin parameter to focus on hard negative samples.
    
    The margin is subtracted from the negative-positive score difference before applying softplus.
    This means that negative samples need to have a higher score by at least 'margin' to contribute significantly to the loss.
    """

    margin = config['margin']
    
    # Compute regularization loss (unchanged)
    if config['samples'] == 1:
        neg_reg_loss = neg_emb0.norm(2).pow(2)
    else:
        neg_reg_loss = neg_emb0.norm(2, dim=2).pow(2).sum() / neg_emb0.shape[1]
    
    reg_loss = (1 / 2) * (
        user_emb0.norm(2).pow(2) +
        pos_emb0.norm(2).pow(2) +
        neg_reg_loss
    ) / float(len(users))
    
    # Compute scores (unchanged)
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    
    if config['samples'] == 1:
        # Single negative case
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
        
        # Add margin: only apply significant loss when neg_score > pos_score + margin
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores - margin))
    else:
        # Multiple negatives case
        users_emb_expanded = users_emb.unsqueeze(1)
        neg_scores = torch.sum(users_emb_expanded * neg_emb, dim=2)
        pos_scores_expanded = pos_scores.unsqueeze(1)
        
        # Add margin: only apply significant loss when neg_score > pos_score + margin
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores_expanded - margin))
    
    return bpr_loss, reg_loss, None

def compute_bpr_loss_orig(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    
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
            users = torch.Tensor(S[:, 0]).long().to(device)
            pos_items = torch.Tensor(S[:, 1]).long().to(device)
            neg_items = torch.Tensor(S[:, 2]).long().to(device)
        else:
            users, pos_items, neg_items_list = ut.multiple_neg_uniform_sample(train_df, adj_list, n_users)
            users = torch.Tensor(users).long().to(device)
            pos_items = torch.Tensor(pos_items).long().to(device)
            neg_items = torch.Tensor(neg_items_list).long().to(device)
                
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

            u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0 = model.encode_minibatch(b_users, b_pos, b_neg, edge_index, edge_attrs)

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