'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import csr_matrix

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def jaccard_sim(matrix, top_k=20, self_loop=False, verbose=-1):
    if verbose > 0:
        print('Computing Jaccard similarity by top-k...')
    
    # Ensure the matrix is binary and of type int
    binary_matrix = csr_matrix((matrix > 0).astype(int))

    # Compute the intersection using dot product
    intersection = binary_matrix.dot(binary_matrix.T).toarray()  # Convert to dense format to avoid dtype issues

    # Compute the row sums (number of interactions)
    row_sums = np.array(binary_matrix.sum(axis=1)).flatten()
    
    # Compute the union
    union = row_sums[:, None] + row_sums[None, :] - intersection
    
    # Ensure intersection and union are of type float to avoid dtype issues
    intersection = intersection.astype(np.float32)
    union = union.astype(np.float32)

    # Compute Jaccard similarity
    similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union != 0)

    #full_similarity_matrix = similarity_matrix.copy()  # Keep the full similarity matrix
    
    # If self_loop is False, set the diagonal to zero
    if self_loop:
        np.fill_diagonal(similarity_matrix, 1)
    else:
        np.fill_diagonal(similarity_matrix, 0)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing {br}jaccard{rs} similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix[i]
        if np.count_nonzero(row) == 0:
            continue
        
        # Sort indices based on similarity values (in descending order) and select top K
        top_k_idx = np.argsort(-row)[:top_k]
        
        # Store the top K similarities
        filtered_data.extend(row[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(top_k_idx)

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del binary_matrix, intersection, row_sums, union, similarity_matrix, filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr() #, csr_matrix(full_similarity_matrix)

def cosine_sim(matrix, top_k=20, self_loop=False, verbose=-1):
    
    if verbose > 0:
        print('Computing cosine similarity by top-k...')
    
    # Convert the sparse matrix to a binary sparse matrix
    sparse_matrix = csr_matrix(matrix)
    sparse_matrix.data = (sparse_matrix.data > 0).astype(int)

    # Compute sparse cosine similarity (output will be sparse)
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    # If self_sim is False, set the diagonal to zero
    if self_loop:
        similarity_matrix.setdiag(1)
    else:
        similarity_matrix.setdiag(0)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f"Preparing {br} cosine {rs} similarity matrix | Top-K: {top_k}")
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col

        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del sparse_matrix, similarity_matrix
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr()


def pearson_sim(matrix, top_k=20, threshold = 0.0, self_loop=False, verbose=-1):
       
    if verbose > 0:
        print('Computing Pearson similarity by top-k...')
    
    # Convert the input matrix to a sparse format (CSR)
    sparse_matrix = csr_matrix(matrix)
    
    # Row-wise mean centering: subtract the mean from non-zero entries
    row_means = np.array(sparse_matrix.mean(axis=1)).flatten()
    sparse_matrix.data -= row_means[sparse_matrix.nonzero()[0]]
    
    if verbose > 0:
        print('Data mean-centered for Pearson similarity.')

    # Compute cosine similarity on the mean-centered data
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    if verbose > 0:
        print('Pearson similarity computed.')
    
    # If self_loop is True, set the diagonal to 1; otherwise, set it to 0
    if self_loop:
        similarity_matrix.setdiag(1)
    else:
        similarity_matrix.setdiag(0)
    
    #full_similarity_matrix = similarity_matrix.copy()  # Keep the full similarity matrix
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing {br}pearson{rs} similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Apply the threshold filter
        valid_idx = row_data > threshold
        row_data = row_data[valid_idx]
        row_indices = row_indices[valid_idx]

        # Sort indices based on similarity values (in descending order) and select top K
        if row_data.size > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del sparse_matrix, similarity_matrix
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr() #, full_similarity_matrix