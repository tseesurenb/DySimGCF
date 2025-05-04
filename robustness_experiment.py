'''
Created on April 9, 2025
Implementation of robustness and cold-start experiments for DySimGCF
'''

import os
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import data_prep as dp
from utils import set_seed, encode_ids
from model import RecSysGNN
from world import config
from data_prep import get_edge_index, create_uuii_adjmat, create_uuii_adjmat_from_feature_data
from robust_eval import (
    evaluate_robustness, 
    evaluate_cold_start, 
    visualize_robustness_results,
    visualize_cold_start_results
)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run robustness and cold-start experiments')
    parser.add_argument('--model_path', type=str, default= './models/params/DySimGCF_ml_100k_knn', 
                        help='Path to the saved model')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='Path to baseline model for comparison')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['robustness', 'cold_start', 'all'],
                        help='Which experiment to run')
    parser.add_argument('--noise_levels', type=str, default='0.1,0.2,0.3,0.4,0.5',
                        help='Comma-separated list of noise levels for robustness test')
    parser.add_argument('--output_dir', type=str, default='./models/robustness_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device (cuda or cpu)')
    return parser.parse_args()

# Load a trained model
def load_model(model_path, device):
    # Extract model configuration from filename
    model_name = os.path.basename(model_path)
    
    # Load dataset based on config
    train_df, test_df = dp.load_data_from_adj_list(dataset=config['dataset'], diff=config['diff'])
    train_df, test_df = encode_ids(train_df, test_df)
    
    N_USERS = train_df['user_id'].nunique()
    N_ITEMS = train_df['item_id'].nunique()
    
    # Initialize model with same configuration
    model = RecSysGNN(
        model=config['model'],
        emb_dim=config['emb_dim'],
        n_layers=config['layers'],
        n_users=N_USERS,
        n_items=N_ITEMS,
        self_loop=config['self_loop'],
        device=device
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, train_df, test_df

def edge_index_generator(train_df):
    """Function to create edge indices based on config settings"""
    N_USERS = train_df['user_id'].nunique()
    N_ITEMS = train_df['item_id'].nunique()
    
    if config['edge'] == 'bi':  # Bipartite graph
        u_t = torch.LongTensor(train_df.user_id)
        i_t = torch.LongTensor(train_df.item_id) + N_USERS
        
        edge_index = torch.stack((
            torch.cat([u_t, i_t]),
            torch.cat([i_t, u_t])
        ))
        
        return edge_index, None
        
    elif config['edge'] == 'knn':  # K-nearest neighbor graph
        if config['sim'] == 'ind':
            knn_train_adj_df = create_uuii_adjmat(train_df, verbose=-1)
        elif config['sim'] == 'trans':
            knn_train_adj_df = create_uuii_adjmat_from_feature_data(train_df, verbose=-1)
        else:
            raise ValueError(f"Invalid sim mode: {config['sim']}")
            
        return get_edge_index(knn_train_adj_df)
    
    else:
        raise ValueError(f"Unsupported edge type: {config['edge']}")

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    print(f"Loading model from {args.model_path}")
    model, train_df, test_df = load_model(args.model_path, device)
    
    # Set up baseline comparison if requested
    baseline_model = None
    if args.compare_with:
        print(f"Loading baseline model from {args.compare_with}")
        if config['model'] != 'DySimGCF':
            # Save original model setting
            original_model = config['model']
            
            # Temporarily change model type for loading baseline
            config['model'] = 'lightGCN' if 'light' in args.compare_with.lower() else 'NGCF'
            baseline_model, _, _ = load_model(args.compare_with, device)
            
            # Restore original model setting
            config['model'] = original_model
        else:
            baseline_model, _, _ = load_model(args.compare_with, device)
    
    # Run robustness experiment
    if args.experiment in ['robustness', 'all']:
        print("===== Running Robustness Experiment =====")
        noise_levels = [float(x) for x in args.noise_levels.split(',')]
        
        # Evaluate main model
        main_results = evaluate_robustness(
            model, train_df, test_df, noise_levels, edge_index_generator, device)
        
        # Evaluate baseline model if provided
        baseline_results = None
        if baseline_model:
            baseline_results = evaluate_robustness(
                baseline_model, train_df, test_df, noise_levels, edge_index_generator, device)
        
        # Visualize and save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.output_dir, f"robustness_results_{timestamp}.png")
        visualize_robustness_results(main_results, baseline_results, save_path)
        
        # Save numerical results
        results_df = pd.DataFrame({
            'noise_level': main_results['noise_levels'],
            'dysimgcf_ndcg': main_results['ndcg'],
            'dysimgcf_recall': main_results['recall'],
            'dysimgcf_precision': main_results['precision'],
        })
        
        if baseline_results:
            results_df['baseline_ndcg'] = baseline_results['ndcg']
            results_df['baseline_recall'] = baseline_results['recall']
            results_df['baseline_precision'] = baseline_results['precision']
        
        csv_path = os.path.join(args.output_dir, f"robustness_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Robustness results saved to {save_path} and {csv_path}")
    
    # Run cold-start experiment
    if args.experiment in ['cold_start', 'all']:
        print("===== Running Cold-Start Experiment =====")
        scenario_types = ['new_users', 'new_items', 'sparse_users']
        
        # Evaluate main model
        main_results = evaluate_cold_start(
            model, train_df, test_df, edge_index_generator, device, scenario_types)
        
        # Evaluate baseline model if provided
        baseline_results = None
        if baseline_model:
            baseline_results = evaluate_cold_start(
                baseline_model, train_df, test_df, edge_index_generator, device, scenario_types)
        
        # Visualize and save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.output_dir, f"cold_start_results_{timestamp}.png")
        visualize_cold_start_results(main_results, baseline_results, save_path)
        
        # Save numerical results
        results_df = pd.DataFrame({
            'scenario': main_results['scenario'],
            'dysimgcf_ndcg': main_results['ndcg'],
            'dysimgcf_recall': main_results['recall'],
            'dysimgcf_precision': main_results['precision'],
        })
        
        if baseline_results:
            results_df['baseline_ndcg'] = baseline_results['ndcg']
            results_df['baseline_recall'] = baseline_results['recall']
            results_df['baseline_precision'] = baseline_results['precision']
        
        csv_path = os.path.join(args.output_dir, f"cold_start_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Cold-start results saved to {save_path} and {csv_path}")

if __name__ == "__main__":
    main()