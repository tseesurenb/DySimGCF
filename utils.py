'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import torch
import random
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from world import config
from sklearn import preprocessing as pp

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def print_metrics(recalls, precs, f1s, ncdg, max_indices, stats): 
    
    print(f" Dataset: {config['dataset']}, num_users: {stats['num_users']}, num_items: {stats['num_items']}, num_interactions: {stats['num_interactions']}")
    
    if config['edge'] == 'bi':
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs} | Shuffle: {br}{config['shuffle']}{rs} | Test Ratio: {br}{config['test_ratio']}{rs} ")
    else:
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | self-{config['self_loop']}): {br}u-{config['u_sim']}(topK {config['u_K']}), i-{config['i_sim']}(topK {config['i_K']}){rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs} | Shuffle: {br}{config['shuffle']}{rs} | Test Ratio: {br}{config['test_ratio']}{rs}")

    metrics = [("Recall", recalls), 
           ("Prec", precs), 
           ("F1 score", f1s), 
           ("NDCG", ncdg)]

    for name, metric in metrics:
        values_str = ', '.join([f"{x:.4f}" for x in metric[:5]])
        mean_str = f"{round(np.mean(metric), 4):.4f}"
        std_str = f"{round(np.std(metric), 4):.4f}"
        
        # Apply formatting with bb and rs if necessary
        if name in ["F1 score", "NDCG"]:
            mean_str = f"{bb}{mean_str}{rs}"
        
        print(f"{name:>8}: {values_str} | {mean_str}, {std_str}")
    
    print(f"{35*'-'}")    
    print(f"   The max NDCG occurs at epoch {br}{(max_indices)}{rs}")
    

def encode_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    
    # Apply transformations to the training DataFrame
    train_df.loc[:, 'user_id'] = le_user.fit_transform(train_df['user_id'].values)
    train_df.loc[:, 'item_id'] = le_item.fit_transform(train_df['item_id'].values)
    
    # Apply transformations to the test DataFrame
    test_df.loc[:, 'user_id'] = le_user.transform(test_df['user_id'].values)
    test_df.loc[:, 'item_id'] = le_item.transform(test_df['item_id'].values)
        
    train_df = train_df.astype({'user_id': 'int64', 'item_id': 'int64'})
    test_df = test_df.astype({'user_id': 'int64', 'item_id': 'int64'})
    
    # Create mappings of original IDs to encoded IDs
    user_mapping = pd.DataFrame({
        'original_user_id': le_user.classes_,
        'encoded_user_id': range(len(le_user.classes_))
    })
    item_mapping = pd.DataFrame({
        'original_item_id': le_item.classes_,
        'encoded_item_id': range(len(le_item.classes_))
    })
    
    if config['diff'] == False:
        # Save mappings to CSV files
        user_mapping.to_csv(f"data/{config['dataset']}/user_id_mapping.csv", index=False)
        item_mapping.to_csv(f"data/{config['dataset']}/item_id_mapping.csv", index=False)
    else:
        # Save mappings to CSV files
        user_mapping.to_csv(f"diff_data/{config['dataset']}/user_id_mapping_diff.csv", index=False)
        item_mapping.to_csv(f"diff_data/{config['dataset']}/item_id_mapping_diff.csv", index=False)
            
    return train_df, test_df

def get_metrics(user_Embed_wts, item_Embed_wts, test_df, K, interactions_t, device, batch_size=100):
    
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)

    n_users = user_Embed_wts.shape[0]

    # Initialize metrics
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
        
    # Collect results across batches
    all_topk_relevance_indices = []
    all_user_ids = []

    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_user_indices = torch.arange(batch_start, batch_end).to(device)

        # Extract embeddings for the current batch
        user_Embed_wts_batch = user_Embed_wts[batch_user_indices]
        #relevance_score_batch = torch.matmul(user_Embed_wts_batch, item_Embed_wts.t())
        relevance_score_batch = torch.matmul(user_Embed_wts_batch, torch.transpose(item_Embed_wts,0, 1))

        # Mask out training user-item interactions from metric computation
        relevance_score_batch = relevance_score_batch * (1 - interactions_t[batch_user_indices])

        # Compute top scoring items for each user
        topk_relevance_indices = torch.topk(relevance_score_batch, K).indices
        all_topk_relevance_indices.append(topk_relevance_indices)
        all_user_ids.extend(batch_user_indices.cpu().numpy())

    # Combine results
    topk_relevance_indices = torch.cat(all_topk_relevance_indices).cpu().numpy()
    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_df.groupby('user_id')['item_id'].apply(list).reset_index()
    # Merge test interactions with top-K predicted relevance indices
    metrics_df = pd.merge(test_interacted_items, pd.DataFrame({'user_id': all_user_ids, 'top_rlvnt_itm': topk_relevance_indices.tolist()}), how='left', on='user_id')
    # Handle missing values and ensure that item_id and top_rlvnt_itm are lists
    metrics_df['item_id'] = metrics_df['item_id'].apply(lambda x: x if isinstance(x, list) else [])
    metrics_df['top_rlvnt_itm'] = metrics_df['top_rlvnt_itm'].apply(lambda x: x if isinstance(x, list) else [])

    # Calculate intersection items
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]
    
    # Calculate recall, precision, and nDCG
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']) if len(x['item_id']) > 0 else 0, axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    
    # Generate a binary relevance matrix for test interactions (same as in the first function)
    test_matrix = np.zeros((len(metrics_df), K))
    for i, row in metrics_df.iterrows():
        relevant_items = set(row['item_id'])
        predicted_items = row['top_rlvnt_itm']
        length = min(K, len(relevant_items))
        test_matrix[i, :length] = 1
   
    # Compute IDCG (Ideal DCG)
    idcg = np.sum(test_matrix * 1./np.log2(np.arange(2, K + 2)), axis=1)
    
    # Compute DCG based on predicted relevance
    dcg_matrix = np.zeros((len(metrics_df), K))
    for i, row in metrics_df.iterrows():
        relevant_items = set(row['item_id'])
        predicted_items = row['top_rlvnt_itm']
        dcg_matrix[i] = [1 if item in relevant_items else 0 for item in predicted_items]
    
    dcg = np.sum(dcg_matrix * (1. / np.log2(np.arange(2, K + 2))), axis=1)
    
    # Handle cases where idcg == 0 to avoid division by zero
    idcg[idcg == 0.] = 1.

    # Compute nDCG as DCG / IDCG
    ndcg = dcg / idcg

    # Set NaNs in nDCG to zero
    ndcg[np.isnan(ndcg)] = 0.

    # Aggregate metrics
    total_recall = metrics_df['recall'].mean()
    total_precision = metrics_df['precision'].mean()
    total_ndcg = np.mean(ndcg)
    
    #if torch.backends.mps.is_available():
    #    device = torch.device("mps")
    
    return total_recall, total_precision, total_ndcg

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, batch_size):

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def make_adj_list(train_df):
    
    all_items_set = set(train_df['item_id'].unique())

    # Group by user_id and aggregate item_ids into lists (positive items)
    pos_items = train_df.groupby('user_id')['item_id'].agg(list)
    
    # Compute neg_items by subtracting the pos_items from all_items for each user
    neg_items = pos_items.apply(lambda pos: list(all_items_set.difference(pos)))
        
    # Create a dictionary with user_id as the key and a sub-dictionary with both pos_items and neg_items
    full_adj_list_dict = {
        user_id: {'pos_items': pos_items[user_id], 'neg_items': neg_items[user_id]} for user_id in pos_items.index
    }

    # Clear unnecessary variables from memory
    del pos_items, neg_items, all_items_set
    
    return full_adj_list_dict

def neg_uniform_sample(train_df, adj_list, n_usr):

    interactions = train_df.to_numpy()
    users = interactions[:, 0].astype(int)
    pos_items = interactions[:, 1].astype(int)           

    neg_items = np.array([random.choice(adj_list[u]['neg_items']) for u in users])
        
    pos_items = [item + n_usr for item in pos_items]
    neg_items = [item + n_usr for item in neg_items]
    
    S = np.column_stack((users, pos_items, neg_items))
    
    del users, pos_items, neg_items
    
    return S

def calculate_neg_weights(train_df, adj_list, items_sim_matrix):
    # Prepare data for vectorized operations
    user_ids = train_df['user_id'].to_numpy()
    item_ids = train_df['item_id'].to_numpy()
    
    top_k = 40

    neg_weights_data = []

    for u, pos_item in tqdm(zip(user_ids, item_ids), total=len(user_ids), desc="Processing Users"):
        # Fetch negative items for the user
        neg_items = np.array(adj_list[u]['neg_items'])

        # Use vectorized computation for weights
        weights = items_sim_matrix[pos_item, neg_items]
        if hasattr(weights, "toarray"):  # Handle sparse matrix case
            weights = weights.toarray().flatten()
            
        # Get top-k negative items by weight
        if len(weights) > top_k:
            #top_k_indices = np.argsort(weights)[-top_k:][::-1]  # Indices of top-k weights in descending order
            
            top_k_indices = np.argsort(weights)[:top_k]  # 
            
            top_neg_items = neg_items[top_k_indices]
            top_neg_weights = weights[top_k_indices]
        else:
            top_neg_items = neg_items
            top_neg_weights = weights

        top_neg_items, top_neg_weights = shuffle(top_neg_items, top_neg_weights)
        
        # Normalize weights
        weights_sum = top_neg_weights.sum()
        if weights_sum > 0:
            top_neg_weights /= weights_sum
        else:
            top_neg_weights = np.full(len(top_neg_weights), 1 / len(top_neg_weights), dtype=np.float32)

        # Append results
        neg_weights_data.append((u, pos_item, top_neg_items.tolist(), top_neg_weights.tolist()))

    # Convert to DataFrame
    train_df_with_neg_list = pd.DataFrame(neg_weights_data, columns=['user_id', 'item_id', 'neg_items', 'neg_weights'])

    return train_df_with_neg_list


def multiple_neg_uniform_sample_old(train_df, full_adj_list, n_usr):
    interactions = train_df.to_numpy()
    users = interactions[:, 0].astype(int)
    pos_items = interactions[:, 1].astype(int)
        
    #For each user, generate N negative samples
    neg_items_list = np.array([
         np.random.choice(full_adj_list[u]['neg_items'], size=config["samples"], replace=True) 
         for u in users
     ])

    # Adjust positive and negative item indices by adding n_usr
    pos_items = [item + n_usr for item in pos_items]
    neg_items_list = [[item + n_usr for item in neg_list] for neg_list in neg_items_list]  # Keep the list structure
    
    # Stack the users, positive items, and the list of negative items
    S = np.column_stack((users, pos_items, neg_items_list))
    
    return S

def multiple_neg_uniform_sample(train_df, full_adj_list, n_usr):
    interactions = train_df.to_numpy()
    users = interactions[:, 0].astype(int)
    pos_items = interactions[:, 1].astype(int)
    
    # For each user, generate N negative samples
    neg_items_list = np.array([
        np.random.choice(full_adj_list[u]['neg_items'], size=config["samples"], replace=True)
        for u in users
    ])
    
    # Adjust positive and negative item indices by adding n_usr
    pos_items = pos_items + n_usr
    neg_items_list = neg_items_list + n_usr
    
    # Return components separately
    return users, pos_items, neg_items_list


import os
import pickle
import numpy as np
from tqdm import tqdm

def precompute_all_epochs_samples(train_df, full_adj_list, n_usr, num_epochs, seed=42, save_dir="precomputed"):
    os.makedirs(save_dir, exist_ok=True)
    config_signature = f"{config['dataset']}_seed{seed}_epochs{num_epochs}_samples{config['samples']}"
    filename = os.path.join(save_dir, f"precomputed_{config_signature}.pkl")

    # Try to load from file
    if os.path.exists(filename):
        print(f"Loading precomputed samples from {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)

    print(f"Precomputing samples for {num_epochs} epochs using fast indexing with seed {seed}...")
    np.random.seed(seed)

    num_samples = config["samples"]
    max_neg_len = max(len(full_adj_list[u]['neg_items']) for u in full_adj_list)

    neg_item_matrix = []
    neg_item_lengths = []
    for u in range(n_usr):
        negs = np.array(full_adj_list[u]['neg_items'])
        if len(negs) == 0:
            negs = np.array([0])
        np.random.shuffle(negs)
        neg_item_matrix.append(negs)
        neg_item_lengths.append(len(negs))

    all_epoch_data = []
    
    for epoch in tqdm(range(num_epochs), desc="Generating epoch samples"):
        shuffled_df = train_df.sample(frac=1, random_state=seed + epoch).reset_index(drop=True)
        interactions = shuffled_df.to_numpy()
        users = interactions[:, 0].astype(int)
        pos_items = interactions[:, 1].astype(int)

        neg_items_list = np.zeros((len(users), num_samples), dtype=np.int32)

        for i, u in enumerate(users):
            negs = neg_item_matrix[u]
            len_negs = neg_item_lengths[u]
            idxs = np.random.randint(0, len_negs, size=num_samples)
            neg_items_list[i] = negs[idxs]

        pos_items += n_usr
        neg_items_list += n_usr

        all_epoch_data.append((users, pos_items, neg_items_list))

    # Save to file
    with open(filename, "wb") as f:
        pickle.dump(all_epoch_data, f)

    return all_epoch_data
                 
def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
                
def plot_results(plot_name, num_exp, all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics):
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    num_test_epochs = len(all_bi_losses[0]['loss'])
    epoch_list = [(j + 1) for j in range(num_test_epochs)]
             
    for i in range(num_exp):
        
        plt.subplot(1, 3, 1)
        # BI Losses
        plt.plot(epoch_list, all_bi_losses[i]['loss'], label=f'Exp {i+1} - BI Total Training Loss', linestyle='-', color='blue')
        plt.plot(epoch_list, all_bi_losses[i]['bpr_loss'], label=f'Exp {i+1} - BI BPR Training Loss', linestyle='--', color='blue')
        plt.plot(epoch_list, all_bi_losses[i]['reg_loss'], label=f'Exp {i+1} - BI Reg Training Loss', linestyle='-.', color='blue')
        
        # KNN Losses
        plt.plot(epoch_list, all_knn_losses[i]['loss'], label=f'Exp {i+1} - KNN Total Training Loss', linestyle='-', color='orange')
        plt.plot(epoch_list, all_knn_losses[i]['bpr_loss'], label=f'Exp {i+1} - KNN BPR Training Loss', linestyle='--', color='orange')
        plt.plot(epoch_list, all_knn_losses[i]['reg_loss'], label=f'Exp {i+1} - KNN Reg Training Loss', linestyle='-.', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        #plt.legend()

        # Plot for metrics
        plt.subplot(1, 3, 2)
        # BI Metrics
        plt.plot(epoch_list, all_bi_metrics[i]['recall'], label=f'Exp {i+1} - BI Recall', linestyle='-', color='blue')
        plt.plot(epoch_list, all_bi_metrics[i]['precision'], label=f'Exp {i+1} - BI Precision', linestyle='--', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_knn_metrics[i]['recall'], label=f'Exp {i+1} - KNN Recall', linestyle='-', color='orange')
        plt.plot(epoch_list, all_knn_metrics[i]['precision'], label=f'Exp {i+1} - KNN Precision', linestyle='--', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Recall & Precision')
        plt.title('Recall & Precision')
        
        # Plot for metrics
        plt.subplot(1, 3, 3)
        # BI Metrics
        plt.plot(epoch_list, all_bi_metrics[i]['ncdg'], label=f'Exp {i+1} - BI NCDG', linestyle='-', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_knn_metrics[i]['ncdg'], label=f'Exp {i+1} - KNN NCDG', linestyle='-', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('NCDG')
        plt.title('NCDG')
        #plt.legend()

    # Custom Legend
    bi_line = mlines.Line2D([], [], color='blue', label='BI')
    knn_line = mlines.Line2D([], [], color='orange', label='KNN')
    plt.legend(handles=[bi_line, knn_line], loc='lower right')
    
    plt.tight_layout()  # Adjust spacing between subplots
    #plt.show()
    
    # Get current date and time
    now = datetime.now()

    # Format date and time as desired (e.g., "2024-08-27_14-30-00")
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(plot_name + '_' + timestamp +'.png')  # Save plot to file
    
def plot_results(plot_name, num_exp, all_losses, all_metrics):
    
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    num_test_epochs = len(all_losses[0]['total_loss'])
    epoch_list = [(j + 1) for j in range(num_test_epochs)]
             
    for i in range(num_exp-1):
        
        plt.subplot(1, 3, 1)
        # BI Losses
        plt.plot(epoch_list, all_losses[i]['total_loss'], label=f'Exp {i+1} - BI Total Training Loss', linestyle='-', color='blue')
        plt.plot(epoch_list, all_losses[i]['bpr_loss'], label=f'Exp {i+1} - BI BPR Training Loss', linestyle='--', color='blue')
        plt.plot(epoch_list, all_losses[i]['reg_loss'], label=f'Exp {i+1} - BI Reg Training Loss', linestyle='-.', color='blue')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        #plt.legend()

        # Plot for metrics
        plt.subplot(1, 3, 2)
        # BI Metrics
        plt.plot(epoch_list, all_metrics[i]['recall'], label=f'Exp {i+1} - BI Recall', linestyle='-', color='blue')
        plt.plot(epoch_list, all_metrics[i]['precision'], label=f'Exp {i+1} - BI Precision', linestyle='--', color='blue')
        
            
        plt.xlabel('Epoch')
        plt.ylabel('Recall & Precision')
        plt.title('Recall & Precision')
        
        # Plot for metrics
        plt.subplot(1, 3, 3)
        # BI Metrics
        plt.plot(epoch_list, all_metrics[i]['ncdg'], label=f'Exp {i+1} - BI NCDG', linestyle='-', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_metrics[i]['ncdg'], label=f'Exp {i+1} - KNN NCDG', linestyle='-', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('NCDG')
        plt.title('NCDG')
        #plt.legend()
    
    plt.tight_layout()  # Adjust spacing between subplots
    #plt.show()
    
    # Get current date and time
    now = datetime.now()

    if plot_name != None:
        # Format date and time as desired (e.g., "2024-08-27_14-30-00")
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(plot_name + '_' + timestamp +'.png')  # Save plot to file
    else:
        plt.show()