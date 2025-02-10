'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def normalize_features(features):
    # Convert to NumPy and normalize
    features_np = features.numpy()
    normalized_features = normalize(features_np, axis=1, norm='l2')  # L2 normalization
    return normalized_features

def calculate_similarity(features):
    print("Calculating similarity...")
    print("Features shape:", features.shape)
     # Normalize features for meaningful similarity
    normalized_features = normalize_features(features)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(normalized_features)

    return similarity_matrix

def encode_item_features(movies_df, title_model_name='all-MiniLM-L6-v2'):
    # One-hot encode genres
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    genres = movies_df[genre_columns].values
    genres = torch.from_numpy(genres).to(torch.float)

    # Load pre-trained sentence transformer model
    model = SentenceTransformer(title_model_name)

    # Encode movie titles
    with torch.no_grad():
        titles = model.encode(movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        titles = titles.cpu()
        
    # Normalize genres
    genres = torch.nn.functional.normalize(genres, p=2, dim=1)

    # Normalize titles
    titles = torch.nn.functional.normalize(titles, p=2, dim=1)
    
    movie_features = torch.cat([genres, titles], dim=-1)

    return movie_features

def encode_user_features(users_df, occupation_model_name='all-MiniLM-L6-v2'):
    # One-hot encode gender ('M' -> [1, 0], 'F' -> [0, 1])
    gender_mapping = {'M': [1, 0], 'F': [0, 1]}
    genders = users_df['gender'].map(gender_mapping).tolist()
    genders = torch.tensor(genders, dtype=torch.float)

    # Normalize age
    ages = torch.tensor(users_df['age'].values, dtype=torch.float).view(-1, 1)
    ages = (ages - ages.mean()) / ages.std()
    ages = torch.nn.functional.normalize(ages, p=2, dim=0)

    # Encode occupation using a SentenceTransformer model
    model = SentenceTransformer(occupation_model_name)
    with torch.no_grad():
        occupations = model.encode(users_df['occupation'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        occupations = occupations.cpu()

    # Normalize occupations
    occupations = torch.nn.functional.normalize(occupations, p=2, dim=1)

    # Concatenate features
    user_features = torch.cat([genders, ages, occupations], dim=-1)

    return user_features

def load_item_file(file_path, item_mapping_path):

    column_names = [
        'movie_id', 'title', 'release_date', 'video_release_date', 'url',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Load the movies file
    movies_df = pd.read_csv(
        file_path,
        sep='|',
        names=column_names,
        encoding='latin-1',
        engine='python'
    )

    # Drop unnecessary columns
    movies_df = movies_df.drop(columns=['video_release_date', 'url'])

    # Load item ID mapping
    item_mapping = pd.read_csv(item_mapping_path)
    
    # Map original movie IDs to encoded item IDs
    movies_df = movies_df.merge(
        item_mapping,
        how='inner',
        left_on='movie_id',
        right_on='original_item_id'
    ).drop(columns=['movie_id', 'original_item_id']).rename(columns={'encoded_item_id': 'movie_id'})
    
    print("Movies DataFrame:")
    print(movies_df.shape)
    # print item_mapping length
    print("Item Mapping:")
    print(item_mapping.shape)

    return movies_df

def load_user_file(file_path, user_mapping_path):
    # Define column names based on the u.user file format
    column_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

    # Load the users file
    users_df = pd.read_csv(
        file_path,
        sep='|',
        names=column_names,
        encoding='latin-1',
        engine='python'
    )

    # Load user ID mapping
    user_mapping = pd.read_csv(user_mapping_path)
    
    # Map original user IDs to encoded user IDs
    users_df = users_df.merge(
        user_mapping,
        how='inner',
        left_on='user_id',
        right_on='original_user_id'
    ).drop(columns=['user_id', 'original_user_id']).rename(columns={'encoded_user_id': 'user_id'})
    
    print("Users DataFrame:")
    print(users_df.shape)

    return users_df


def create_item_sim(movies_path, top_k=10, verbose=False):
    # Load the entire movie dataframe into memory:
    movies_df = load_item_file(movies_path, 'data/ml-100k/item_id_mapping.csv')
    if verbose:
        print("Loaded Movies DataFrame:")
        print(movies_df.head())
    
    # Encode item features
    item_features = encode_item_features(movies_df)
    if verbose:
        print("Encoded Item Features Shape:", item_features.shape)
    
    # Calculate item-item similarity matrix
    item_similarity_matrix = calculate_similarity(item_features)
    if verbose:
        print("Initial Item-Item Similarity Matrix Shape:", item_similarity_matrix.shape)

    # Filter to retain only the top_k most similar items for each item
    num_items = item_similarity_matrix.shape[0]
    filtered_similarity_matrix = np.zeros_like(item_similarity_matrix)

    for i in range(num_items):
        # Get the indices of the top_k most similar items for the current item
        top_k_indices = np.argsort(item_similarity_matrix[i])[-top_k:]
        
        # Retain only the top_k similarities
        for idx in top_k_indices:
            filtered_similarity_matrix[i, idx] = item_similarity_matrix[i, idx]

    if verbose:
        print("Filtered Item-Item Similarity Matrix Shape:", filtered_similarity_matrix.shape)

    return filtered_similarity_matrix


def create_user_sim(user_path, top_k=10, verbose=False):
    # Load the user data
    user_df = load_user_file(user_path, 'data/ml-100k/user_id_mapping.csv')
    
    if verbose:
        print("Loaded User DataFrame:")
        print(user_df.head())
    
    # Encode user features
    user_features = encode_user_features(user_df)
    
    if verbose:
        print("Encoded User Features Shape:", user_features.shape)
    
    # Calculate user-user similarity matrix
    user_similarity_matrix = calculate_similarity(user_features)
    
    if verbose:
        print("Initial User-User Similarity Matrix Shape:", user_similarity_matrix.shape)

    # Filter to retain only the top_k most similar users for each user
    num_users = user_similarity_matrix.shape[0]
    filtered_similarity_matrix = np.zeros_like(user_similarity_matrix)

    for i in range(num_users):
        # Get the indices of the top_k most similar users for the current user
        top_k_indices = np.argsort(user_similarity_matrix[i])[-top_k:]
        
        # Retain only the top_k similarities
        for idx in top_k_indices:
            filtered_similarity_matrix[i, idx] = user_similarity_matrix[i, idx]

    if verbose:
        print("Filtered User-User Similarity Matrix Shape:", filtered_similarity_matrix.shape)

    return filtered_similarity_matrix


def count_sim(sim_matrix):
    count = 0
    count_0 = 0
    count_1 = 0
    count_all = 0
    count_minus_1 = 0
    count_plus_1 = 0
    count_minus_one = 0
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            count_all += 1
            if sim_matrix[i][j] > 0 and sim_matrix[i][j] < 1:
                count += 1
            if sim_matrix[i][j] == 0:
                count_0 += 1
            if sim_matrix[i][j] == 1:
                count_1 += 1
            if sim_matrix[i][j] == -1:
                count_minus_one += 1
            if sim_matrix[i][j] > -1 and sim_matrix[i][j] < 0:
                count_minus_1 += 1
            if sim_matrix[i][j] > 1:
                count_plus_1 += 1

    print("Count of values between 0 and 1 in similarity matrix:", count)
    print("Count of values equal to 0 in similarity matrix:", count_0)
    print("Count of values equal to 1 in similarity matrix:", count_1)
    print("Count of all values in similarity matrix:", count_all)
    print("Count of values between -1 and 0 in similarity matrix:", count_minus_1)
    print("Count of values greater than 1 in similarity matrix:", count_plus_1)
    print("Count of values equal to -1 in similarity matrix:", count_minus_one)

    return count, count_0, count_1, count_all, count_minus_1, count_plus_1, count_minus_one


def show_sim(sim_matrix):
    plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Similarity Matrix")
    plt.show()
    
def show_sim2(sim_matrix, u_id, max_k=10):
    # Ensure we slice correctly to get a square matrix of size (max_k, max_k)
    sub_matrix = sim_matrix[u_id:u_id + max_k, u_id:u_id + max_k]  # Properly select a square submatrix
    
    # Check the shape to avoid ValueError
    if sub_matrix.shape != (max_k, max_k):
        raise ValueError(f"The selected submatrix is not of shape ({max_k}, {max_k}). Actual shape: {sub_matrix.shape}")
    
    plt.imshow(sub_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Similarity Matrix")
    plt.show()



