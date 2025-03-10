import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
        
# Construct Laplacian matrix
def construct_laplacian(_train_df, N_USERS, N_ITEMS):
    # Create user-item interaction matrix as sparse matrix
    rows = _train_df['user_id'].values
    cols = _train_df['item_id'].values
    
    # If there's a rating column, use it as values, otherwise use 1s
    if 'rating' in _train_df.columns:
        data = _train_df['rating'].values
    else:
        data = np.ones(len(rows))
    
    # Create sparse interaction matrix
    interact_matrix = sp.csr_matrix((data, (rows, cols)), shape=(N_USERS, N_ITEMS))
    
    # Create bipartite adjacency matrix
    zero_uu = sp.csr_matrix((N_USERS, N_USERS))
    zero_ii = sp.csr_matrix((N_ITEMS, N_ITEMS))
    
    # Combine to form full adjacency matrix
    adjacency = sp.vstack([
        sp.hstack([zero_uu, interact_matrix]),
        sp.hstack([interact_matrix.T, zero_ii])
    ]).tocsr()
    
    # Compute degree matrix
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    
    # Add small epsilon to avoid division by zero
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degrees + 1e-8))
    
    # Compute normalized Laplacian: I - D^(-1/2) A D^(-1/2)
    I = sp.eye(adjacency.shape[0])
    L = I - D_sqrt_inv @ adjacency @ D_sqrt_inv
    
    return L, adjacency

# Apply spectral transformation
def spectral_transform(laplacian, k=200):
    print(f"Computing {k} smallest eigenvectors of the Laplacian matrix...")
    
    # Compute k smallest eigenvectors (approximation for large graphs)
    k = min(k, laplacian.shape[0] - 2)  # Make sure k is valid
    eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
    
    print(f"Spectral decomposition complete. Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    
    return eigenvalues, eigenvectors

# Compute similarity in spectral domain
def compute_spectral_similarities(eigenvectors, N_USERS, N_ITEMS, similarity_type='cosine'):
    # Extract user and item spectral representations
    user_spectral = eigenvectors[:N_USERS, :]
    item_spectral = eigenvectors[N_USERS:, :]
    
    if similarity_type == 'cosine':
        # Compute cosine similarity for users
        user_norms = np.linalg.norm(user_spectral, axis=1, keepdims=True)
        user_normalized = user_spectral / (user_norms + 1e-8)
        user_similarity = user_normalized @ user_normalized.T
        
        # Compute cosine similarity for items
        item_norms = np.linalg.norm(item_spectral, axis=1, keepdims=True)
        item_normalized = item_spectral / (item_norms + 1e-8)
        item_similarity = item_normalized @ item_normalized.T
    
    elif similarity_type == 'jaccard':
        # For jaccard, we need binary vectors
        # This is an approximation for spectral embeddings
        user_binary = (user_spectral > 0).astype(float)
        item_binary = (item_spectral > 0).astype(float)
        
        # Compute jaccard similarity for users
        user_similarity = np.zeros((N_USERS, N_USERS))
        for i in range(N_USERS):
            for j in range(i, N_USERS):
                intersection = np.sum(np.logical_and(user_binary[i] > 0, user_binary[j] > 0))
                union = np.sum(np.logical_or(user_binary[i] > 0, user_binary[j] > 0))
                user_similarity[i, j] = intersection / (union + 1e-8)
                user_similarity[j, i] = user_similarity[i, j]  # Symmetry
        
        # Compute jaccard similarity for items
        item_similarity = np.zeros((N_ITEMS, N_ITEMS))
        for i in range(N_ITEMS):
            for j in range(i, N_ITEMS):
                intersection = np.sum(np.logical_and(item_binary[i] > 0, item_binary[j] > 0))
                union = np.sum(np.logical_or(item_binary[i] > 0, item_binary[j] > 0))
                item_similarity[i, j] = intersection / (union + 1e-8)
                item_similarity[j, i] = item_similarity[i, j]  # Symmetry
    
    else:
        raise ValueError(f"Similarity type '{similarity_type}' not supported. Use 'cosine' or 'jaccard'.")
    
    return user_similarity, item_similarity

# Create edges from similarities
def create_edges_from_similarities(user_similarity, item_similarity, k_users, k_items, N_USERS):
    # Get top-k neighbors for each user
    user_edges = []
    user_attrs = []
    for i in range(user_similarity.shape[0]):
        # Get top-k indices for this user (excluding self)
        sim_row = user_similarity[i, :].copy()
        sim_row[i] = -np.inf  # Exclude self
        top_indices = np.argsort(sim_row)[-k_users:][::-1]
        
        for j in top_indices:
            user_edges.append([i, j])
            user_attrs.append(user_similarity[i, j])
    
    # Get top-k neighbors for each item
    item_edges = []
    item_attrs = []
    for i in range(item_similarity.shape[0]):
        # Get top-k indices for this item (excluding self)
        sim_row = item_similarity[i, :].copy()
        sim_row[i] = -np.inf  # Exclude self
        top_indices = np.argsort(sim_row)[-k_items:][::-1]
        
        for j in top_indices:
            # Add N_USERS to item indices to maintain separation
            item_edges.append([i + N_USERS, j + N_USERS])
            item_attrs.append(item_similarity[i, j])
    
    # Combine user and item edges
    all_edges = np.array(user_edges + item_edges).T
    all_attrs = np.array(user_attrs + item_attrs)
    
    return all_edges, all_attrs