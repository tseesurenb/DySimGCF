'''
Created on April 9, 2025
Implementation of robustness and cold-start evaluation for DySimGCF
'''

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import get_metrics, minibatch
from world import config

# ANSI escape codes for formatting
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def create_noisy_data(train_df, noise_level=0.1, noise_type='random_drop'):
    """
    Create a noisy version of the training data to test robustness.
    
    Args:
        train_df: Original training dataframe
        noise_level: Proportion of interactions to affect (0.0-1.0)
        noise_type: Type of noise to add ('random_drop', 'random_add', 'adversarial')
    
    Returns:
        Dataframe with noise applied
    """
    noisy_df = train_df.copy()
    
    if noise_type == 'random_drop':
        # Randomly drop interactions
        drop_mask = np.random.random(len(noisy_df)) < noise_level
        noisy_df = noisy_df[~drop_mask].reset_index(drop=True)
        print(f"Randomly dropped {sum(drop_mask)} interactions ({noise_level*100:.1f}%)")
        
    elif noise_type == 'random_add':
        # Randomly add fake interactions
        n_users = noisy_df['user_id'].nunique()
        n_items = noisy_df['item_id'].nunique()
        n_add = int(len(noisy_df) * noise_level)
        
        # Generate random user-item pairs
        random_users = np.random.randint(0, n_users, n_add)
        random_items = np.random.randint(0, n_items, n_add)
        
        # Create new dataframe with random interactions
        random_df = pd.DataFrame({
            'user_id': random_users,
            'item_id': random_items
        })
        
        # Concatenate with original, drop duplicates to avoid conflicts
        noisy_df = pd.concat([noisy_df, random_df]).drop_duplicates(['user_id', 'item_id']).reset_index(drop=True)
        print(f"Added {len(noisy_df) - len(train_df)} random interactions")
        
    elif noise_type == 'adversarial':
        # Add interactions between users and their least similar items
        # This requires the similarity matrix which we'll extract from the model
        # We'll implement the logic in a separate function
        pass
    
    return noisy_df

def create_cold_start_scenarios(train_df, test_df, scenario_type='new_users', ratio=0.2):
    """
    Create evaluation scenarios for cold-start problems.
    
    Args:
        train_df: Original training dataframe
        test_df: Original test dataframe
        scenario_type: 'new_users', 'new_items', or 'sparse_users'
        ratio: Proportion of users/items to consider as cold-start (0.0-1.0)
    
    Returns:
        Modified train_df and test_df for cold-start evaluation
    """
    if scenario_type == 'new_users':
        # Identify a subset of users to treat as "new" users
        all_users = train_df['user_id'].unique()
        n_cold_users = int(len(all_users) * ratio)
        cold_start_users = np.random.choice(all_users, n_cold_users, replace=False)
        
        # Move most interactions from these users to test set
        cold_mask = train_df['user_id'].isin(cold_start_users)
        cold_interactions = train_df[cold_mask].copy()
        
        # Keep only 1-3 interactions per cold-start user in training
        kept_interactions = pd.DataFrame()
        for user in cold_start_users:
            user_data = cold_interactions[cold_interactions['user_id'] == user]
            # Keep 1-3 random interactions for training
            n_keep = np.random.randint(1, min(4, len(user_data)))
            keep_idx = np.random.choice(user_data.index, n_keep, replace=False)
            kept_interactions = pd.concat([kept_interactions, train_df.loc[keep_idx]])
        
        # Remove cold users from training and add the few kept interactions back
        new_train_df = train_df[~cold_mask].copy().reset_index(drop=True)
        new_train_df = pd.concat([new_train_df, kept_interactions]).reset_index(drop=True)
        
        # Add remaining cold user interactions to test
        moved_to_test = cold_interactions[~cold_interactions.index.isin(kept_interactions.index)]
        new_test_df = pd.concat([test_df, moved_to_test]).reset_index(drop=True)
        
        print(f"Cold-start users evaluation: {n_cold_users} users with limited training data")
        return new_train_df, new_test_df
        
    elif scenario_type == 'new_items':
        # Similar logic for new items
        all_items = train_df['item_id'].unique()
        n_cold_items = int(len(all_items) * ratio)
        cold_start_items = np.random.choice(all_items, n_cold_items, replace=False)
        
        # Move most interactions from these items to test set
        cold_mask = train_df['item_id'].isin(cold_start_items)
        cold_interactions = train_df[cold_mask].copy()
        
        # Keep only a few interactions per cold-start item in training
        kept_interactions = pd.DataFrame()
        for item in cold_start_items:
            item_data = cold_interactions[cold_interactions['item_id'] == item]
            # Keep 1-3 random interactions for training
            n_keep = np.random.randint(1, min(4, len(item_data)))
            keep_idx = np.random.choice(item_data.index, n_keep, replace=False)
            kept_interactions = pd.concat([kept_interactions, train_df.loc[keep_idx]])
        
        # Remove cold items from training and add the few kept interactions back
        new_train_df = train_df[~cold_mask].copy().reset_index(drop=True)
        new_train_df = pd.concat([new_train_df, kept_interactions]).reset_index(drop=True)
        
        # Add remaining cold item interactions to test
        moved_to_test = cold_interactions[~cold_interactions.index.isin(kept_interactions.index)]
        new_test_df = pd.concat([test_df, moved_to_test]).reset_index(drop=True)
        
        print(f"Cold-start items evaluation: {n_cold_items} items with limited training data")
        return new_train_df, new_test_df
        
    elif scenario_type == 'sparse_users':
        # Create evaluation for users with very few interactions
        interaction_counts = train_df['user_id'].value_counts()
        sparse_users = interaction_counts[interaction_counts <= 5].index.tolist()
        
        if len(sparse_users) < 10:  # If not enough naturally sparse users
            return create_cold_start_scenarios(train_df, test_df, 'new_users', ratio)
            
        print(f"Evaluating {len(sparse_users)} naturally sparse users")
        
        # Create a test set focused on sparse users
        sparse_test_df = test_df[test_df['user_id'].isin(sparse_users)].copy()
        if len(sparse_test_df) < 50:  # If not enough test data for sparse users
            # Move some training data to test for evaluation
            sparse_train = train_df[train_df['user_id'].isin(sparse_users)].copy()
            move_ratio = 0.3  # Move 30% to test
            move_mask = np.random.random(len(sparse_train)) < move_ratio
            
            new_test_items = sparse_train[move_mask].copy()
            new_train_df = pd.concat([train_df, sparse_train[~move_mask]]).reset_index(drop=True)
            sparse_test_df = pd.concat([sparse_test_df, new_test_items]).reset_index(drop=True)
        
        return train_df, sparse_test_df
    
    return train_df, test_df

def evaluate_robustness(model, train_df, test_df, noise_levels, edge_index_fn, device):
    """
    Evaluate model robustness under different noise conditions
    
    Args:
        model: Trained model
        train_df: Original training dataframe
        test_df: Test dataframe
        noise_levels: List of noise levels to test
        edge_index_fn: Function to create edge indices from dataframe
        device: Computing device
    
    Returns:
        Dictionary with performance metrics at each noise level
    """
    results = {
        'noise_levels': noise_levels,
        'recall': [],
        'precision': [],
        'ndcg': []
    }
    
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    
    # Create interactions tensor for evaluation
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    
    v = torch.ones(len(train_df), dtype=torch.float32).to(device)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()
    
    for noise_level in noise_levels:
        # Create noisy version of training data
        noisy_df = create_noisy_data(train_df, noise_level=noise_level)
        
        # Create new edge index and attributes based on noisy data
        edge_index, edge_attrs = edge_index_fn(noisy_df)
        edge_index = torch.tensor(edge_index).to(device).long()
        if edge_attrs is not None:
            edge_attrs = torch.tensor(edge_attrs).to(device)
        
        # Evaluate model on noisy data
        model.eval()
        with torch.no_grad():
            _, out = model(edge_index, edge_attrs)
            final_u_emb, final_i_emb = torch.split(out, (n_users, n_items))
            recall, precision, ndcg = get_metrics(final_u_emb, final_i_emb, test_df, config['top_K'], interactions_t, device)
        
        # Store results
        results['recall'].append(recall)
        results['precision'].append(precision)
        results['ndcg'].append(ndcg)
        
        print(f"Noise level {noise_level:.2f}: Recall@{config['top_K']}={recall:.4f}, NDCG={ndcg:.4f}")
    
    return results

def evaluate_cold_start(model, train_df, test_df, edge_index_fn, device, scenario_types=None):
    """
    Evaluate model performance in cold-start scenarios
    
    Args:
        model: Trained model
        train_df: Original training dataframe
        test_df: Test dataframe
        edge_index_fn: Function to create edge indices from dataframe
        device: Computing device
        scenario_types: List of cold-start scenarios to evaluate
    
    Returns:
        Dictionary with performance metrics for each scenario
    """
    if scenario_types is None:
        scenario_types = ['new_users', 'new_items', 'sparse_users']
    
    results = {
        'scenario': scenario_types,
        'recall': [],
        'precision': [],
        'ndcg': []
    }
    
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    
    for scenario in scenario_types:
        # Create cold-start scenario
        cold_train_df, cold_test_df = create_cold_start_scenarios(
            train_df, test_df, scenario_type=scenario)
        
        # Create interactions tensor for evaluation
        i = torch.stack((
            torch.LongTensor(cold_train_df['user_id'].values),
            torch.LongTensor(cold_train_df['item_id'].values)
        )).to(device)
        
        v = torch.ones(len(cold_train_df), dtype=torch.float32).to(device)
        interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()
        
        # Create edge index for cold-start scenario
        edge_index, edge_attrs = edge_index_fn(cold_train_df)
        edge_index = torch.tensor(edge_index).to(device).long()
        if edge_attrs is not None:
            edge_attrs = torch.tensor(edge_attrs).to(device)
        
        # Evaluate model on cold-start scenario
        model.eval()
        with torch.no_grad():
            _, out = model(edge_index, edge_attrs)
            final_u_emb, final_i_emb = torch.split(out, (n_users, n_items))
            recall, precision, ndcg = get_metrics(final_u_emb, final_i_emb, cold_test_df, config['top_K'], interactions_t, device)
        
        # Store results
        results['recall'].append(recall)
        results['precision'].append(precision)
        results['ndcg'].append(ndcg)
        
        print(f"Cold-start scenario '{scenario}': Recall@{config['top_K']}={recall:.4f}, NDCG={ndcg:.4f}")
    
    return results

def visualize_robustness_results(results, baseline_results=None, save_path=None):
    """Visualize robustness evaluation results"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot NDCG vs noise level
    plt.subplot(1, 2, 1)
    plt.plot(results['noise_levels'], results['ndcg'], 'o-', label='DySimGCF', linewidth=2)
    
    if baseline_results:
        plt.plot(baseline_results['noise_levels'], baseline_results['ndcg'], 's--', label='LightGCN', linewidth=2)
    
    plt.xlabel('Noise Level')
    plt.ylabel('NDCG@K')
    plt.title('Robustness to Noise')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Recall vs noise level
    plt.subplot(1, 2, 2)
    plt.plot(results['noise_levels'], results['recall'], 'o-', label='DySimGCF', linewidth=2)
    
    if baseline_results:
        plt.plot(baseline_results['noise_levels'], baseline_results['recall'], 's--', label='LightGCN', linewidth=2)
    
    plt.xlabel('Noise Level')
    plt.ylabel(f'Recall@{config["top_K"]}')
    plt.title('Robustness to Noise')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_cold_start_results(results, baseline_results=None, save_path=None):
    """Visualize cold-start evaluation results"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(14, 6))
    
    # Set up bar positions
    scenarios = results['scenario']
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Plot NDCG
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, results['ndcg'], width, label='DySimGCF')
    
    if baseline_results:
        plt.bar(x + width/2, baseline_results['ndcg'], width, label='LightGCN')
    
    plt.xlabel('Scenario')
    plt.ylabel('NDCG@K')
    plt.title('Cold-Start Performance (NDCG)')
    plt.xticks(x, scenarios)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Recall
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, results['recall'], width, label='DySimGCF')
    
    if baseline_results:
        plt.bar(x + width/2, baseline_results['recall'], width, label='LightGCN')
    
    plt.xlabel('Scenario')
    plt.ylabel(f'Recall@{config["top_K"]}')
    plt.title('Cold-Start Performance (Recall)')
    plt.xticks(x, scenarios)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()