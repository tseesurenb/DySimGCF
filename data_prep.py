'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import pandas as pd
import numpy as np
import i_sim as sim
import f_sim as fs

import gc
import os

from world import config
from scipy.sparse import coo_matrix, vstack, hstack, load_npz, save_npz


# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def get_edge_index(sparse_matrix):    
    # Extract row, column indices and data values
    row_indices = sparse_matrix.row
    column_indices = sparse_matrix.col
    data = sparse_matrix.data.astype(np.float32)
        
    # Prepare edge index
    edge_index = np.vstack((row_indices, column_indices))
    
    del row_indices, column_indices
    
    return edge_index, data

def create_uuii_adjmat(df, verbose=-1):
    
    u_sim = config['u_sim']
    i_sim = config['i_sim']
    u_top_k = config['u_K']
    i_top_k = config['i_K']
    self_loop = config['self_loop']
    
    if config['save_sim_mat']:
        file_path=f"pre_proc/{config['dataset']}_u_{u_sim}_{u_top_k}_i_{i_sim}_{i_top_k}_self_{config['self_loop']}_uuii_adjmat.npz"
    
        # Check if the file exists
        if os.path.exists(file_path):
            if verbose > 0:
                print('Loading an adjacency matrix from file: ', file_path)
            # Load the sparse matrix from the file
            combined_adjacency = load_npz(file_path)
            return combined_adjacency

    if verbose > 0:
        print('Creating an user-item matrix ...')
    # Convert to NumPy arrays
    user_ids = df['user_id'].to_numpy()
    item_ids = df['item_id'].to_numpy()

    # Create a sparse matrix directly
    user_item_matrix_coo = coo_matrix((np.ones(len(df)), (user_ids, item_ids)))
    cos_user_user_sim_matrix = None
    jac_user_user_sim_matrix = None
    cos_item_item_sim_matrix = None
    jac_item_item_sim_matrix = None
    user_item_matrix = user_item_matrix_coo.toarray()

    if verbose > 0:
        print('The user-item coo matrix was created.')
        
    # Calculate user-user similarity matrix
    if u_sim == 'cos':
        user_user_sim_matrix = sim.cosine_sim(user_item_matrix, top_k=u_top_k, self_loop = self_loop, verbose=verbose)
    elif u_sim == 'jac':
        user_user_sim_matrix = sim.jaccard_sim(user_item_matrix, top_k=u_top_k, self_loop = self_loop, verbose=verbose)
    else:
        print(f'{br} The similarity metric {i_sim} is not supported yet !!!{rs}, available options: cos and jac')
        
    if verbose > 0:
        print('The user-user similarity matrix was created.')
    
    # Calculate item-item similarity matrix
    if i_sim == 'cos':
        item_item_sim_matrix = sim.cosine_sim(user_item_matrix.T, top_k=i_top_k, self_loop = config['self_loop'], verbose=verbose)
    elif i_sim == 'jac': # Jaccard similarity
        item_item_sim_matrix = sim.jaccard_sim(user_item_matrix.T, top_k=i_top_k, self_loop = config['self_loop'], verbose=verbose)
    else:
        print(f'{br} The similarity metric {i_sim} is not supported yet !!!{rs}, available options: cos and jac')
    
          
    if verbose > 0:
        print('The item-item similarity matrix was created.')
    
    # Stack user-user and item-item matrices vertically and horizontally
    num_users = user_user_sim_matrix.shape[0]
    num_items = item_item_sim_matrix.shape[0]

    # Initialize combined sparse matrix
    combined_adjacency = vstack([
        hstack([user_user_sim_matrix, coo_matrix((num_users, num_items))]),
        hstack([coo_matrix((num_items, num_users)), item_item_sim_matrix])
    ])
    
    if config['save_sim_mat']:
        # Save the sparse matrix to a file
        save_npz(file_path, combined_adjacency)
            
    del user_item_matrix_coo, user_item_matrix, user_user_sim_matrix, item_item_sim_matrix, cos_user_user_sim_matrix, jac_user_user_sim_matrix, cos_item_item_sim_matrix, jac_item_item_sim_matrix

    return combined_adjacency


def create_uuii_adjmat_from_feature_data(df, verbose=-1):

    # Configuration parameters
    u_sim = config['u_sim']
    i_sim = config['i_sim']
    u_top_k = config['u_K']
    i_top_k = config['i_K']
    self_loop = config['self_loop']
    
    if verbose > 0:
        print('Creating a user-item matrix...')
    
    # Create a sparse user-item interaction matrix
    #user_item_matrix_coo = coo_matrix((np.ones(len(df)), (user_ids, item_ids)))
    #user_item_matrix = user_item_matrix_coo.toarray()

    if verbose > 0:
        print('User-item matrix created.')

    users_path = 'data/ml-100k/u.user'
    movies_path = 'data/ml-100k/u.item'
    # Generate user-user similarity matrix using external `create_user_sim`
    if u_sim == 'cos':
        user_user_sim_matrix = fs.create_user_sim(users_path, top_k=u_top_k)
    else:
        print(f'{u_sim} similarity metric for users is not implemented in this function.')

    # Generate item-item similarity matrix using external `create_item_sim`
    if i_sim == 'cos':
        item_item_sim_matrix = fs.create_item_sim(movies_path, top_k=i_top_k)
    else:
        print(f'{i_sim} similarity metric for items is not implemented in this function.')

    if verbose > 0:
        print('User-user and item-item similarity matrices created.')

    # Stack user-user and item-item matrices vertically and horizontally
    num_users = user_user_sim_matrix.shape[0]
    num_items = item_item_sim_matrix.shape[0]

    # Combine similarity matrices into an adjacency matrix
    combined_adjacency = vstack([
        hstack([coo_matrix(user_user_sim_matrix), coo_matrix((num_users, num_items))]),
        hstack([coo_matrix((num_items, num_users)), coo_matrix(item_item_sim_matrix)])
    ])

    #del user_item_matrix_coo, user_item_matrix, user_user_sim_matrix, item_item_sim_matrix
    del user_user_sim_matrix, item_item_sim_matrix

    return combined_adjacency

def sum_common_entries(c1, c2):
    # Assume c1 and c2 are your input sparse matrices
    # c1 and c2 are in COO format
    c1_coo = c1.tocoo()
    c2_coo = c2.tocoo()

    # Create sets of row, column indices for both matrices
    indices_c1 = set(zip(c1_coo.row, c1_coo.col))
    indices_c2 = set(zip(c2_coo.row, c2_coo.col))

    # Find common indices
    common_indices = indices_c1.intersection(indices_c2)

    # Create dictionaries to quickly access values
    c1_dict = {(row, col): val for row, col, val in zip(c1_coo.row, c1_coo.col, c1_coo.data)}
    c2_dict = {(row, col): val for row, col, val in zip(c2_coo.row, c2_coo.col, c2_coo.data)}

    # Compute the sum for common indices
    rows, cols, data = [], [], []
    for row, col in common_indices:
        rows.append(row)
        cols.append(col)
        data.append(c1_dict[(row, col)] + c2_dict[(row, col)])

    # Create the resulting sparse matrix
    c = coo_matrix((data, (rows, cols)), shape=c1.shape)
    
    return c

def create_item_similarity_dict(item_item_sim_matrix, verbose=0):
    # Convert sparse matrix to COO format to easily access non-zero elements
    if not isinstance(item_item_sim_matrix, coo_matrix):
        item_item_sim_matrix = item_item_sim_matrix.tocoo()

    item_similarity_dict = {}

    # Iterate through the non-zero entries of the item-item similarity matrix
    for row, col, sim_value in zip(item_item_sim_matrix.row, item_item_sim_matrix.col, item_item_sim_matrix.data):
        if row != col:  # Optional: exclude self-similarities if necessary
            if row not in item_similarity_dict:
                item_similarity_dict[row] = []
            item_similarity_dict[row].append(col)
    
    if verbose > 0:
        print(f"Created item similarity dictionary with {len(item_similarity_dict)} items.")
    
    return item_similarity_dict

def create_item_similarity_dict_20(item_item_sim_matrix, top_n=10, verbose=0):
    # Convert sparse matrix to COO format to easily access non-zero elements
    if not isinstance(item_item_sim_matrix, coo_matrix):
        item_item_sim_matrix = item_item_sim_matrix.tocoo()

    item_similarity_dict = {}

    # Dictionary to hold similarities before selecting top N
    similarity_scores = {}

    # Iterate through the non-zero entries of the item-item similarity matrix
    for row, col, sim_value in zip(item_item_sim_matrix.row, item_item_sim_matrix.col, item_item_sim_matrix.data):
        if row != col:  # Exclude self-similarities
            if row not in similarity_scores:
                similarity_scores[row] = []
            similarity_scores[row].append((col, sim_value))  # Store both item and similarity score

    # For each item, sort by similarity score and retain only the top N
    for item, sims in similarity_scores.items():
        top_similar_items = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]  # Sort by score, keep top N
        item_similarity_dict[item] = [x[0] for x in top_similar_items]  # Keep only the item IDs

    if verbose > 0:
        print(f"Created item similarity dictionary with {len(item_similarity_dict)} items.")
    
    return item_similarity_dict


def load_data_from_adj_list(dataset = "lastfm", verbose = 0):
    
    train_df = None
    test_df = None
    df = None
    
    datasets = ['amazon_book', 'yelp2018', 'lastfm', 'gowalla', 'itstore', 'ml-1m', 'ml-100k', 'ml-100k_2']
    
    if dataset not in datasets:
        print(f'{br} The dataset {dataset} is not supported yet !!!{rs}')
        return
                      
    # Paths for ML-1M data files
    train_path = f'data/{dataset}/train_coo.txt'
    test_path = f'data/{dataset}/test_coo.txt'
    
    # Load the entire ratings dataframe into memory
    df = pd.read_csv(train_path, header=0, sep=' ')
    # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
    train_df = df[['user_id', 'item_id']]
                            
    # Load the entire ratings dataframe into memory
    df = pd.read_csv(test_path, header=0, sep=' ')
    # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
    test_df = df[['user_id', 'item_id']]
    
    if verbose > 0:
        print(f'{bg}Data loaded for dataset: {dataset} !!!{rs}')
        print(f'{b}Train data shape: {train_df.shape}, Test data shape: {test_df.shape}{rs}')
               
    del df
    gc.collect()
        
    return train_df, test_df

