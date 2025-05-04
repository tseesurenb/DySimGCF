'''
Created on Apr 7, 2025
Script to plot the distribution of user and item interactions for multiple datasets
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from collections import Counter

# ANSI escape codes for formatted output
br = "\033[1;31m"  # bold red
b = "\033[1m"      # bold
bg = "\033[1;32m"  # bold green
bb = "\033[1;34m"  # bold blue
rs = "\033[0m"     # reset style

def load_data(dataset, verbose=1):
    """
    Load training data for a given dataset
    
    Parameters:
    -----------
    dataset : str
        Name of the dataset to load
    verbose : int
        Level of verbosity
    
    Returns:
    --------
    train_df : pandas.DataFrame
        DataFrame containing user-item interactions
    """
    # Paths for data files
    train_path = f'data/{dataset}/train_coo.txt'
    
    if not os.path.exists(train_path):
        print(f"{br}Error: File not found - {train_path}{rs}")
        return None
        
    try:
        # Load the training dataframe
        df = pd.read_csv(train_path, header=0, sep=' ')
        # Select the relevant columns
        train_df = df[['user_id', 'item_id']]
        
        if verbose > 0:
            print(f"{bg}Data loaded for dataset: {dataset}{rs}")
            print(f"{b}Train data shape: {train_df.shape}{rs}")
        
        return train_df
    except Exception as e:
        print(f"{br}Error loading dataset {dataset}: {str(e)}{rs}")
        return None

def compute_interaction_stats(df):
    """
    Compute interaction statistics for users and items
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing user-item interactions
    
    Returns:
    --------
    user_counts : Counter
        Dictionary with user_id as key and interaction count as value
    item_counts : Counter
        Dictionary with item_id as key and interaction count as value
    """
    # Count interactions per user
    user_counts = Counter(df['user_id'])
    
    # Count interactions per item
    item_counts = Counter(df['item_id'])
    
    return user_counts, item_counts

def plot_interaction_distribution(datasets, log_scale=True, save_path='plots'):
    """
    Plot the distribution of user and item interactions for multiple datasets
    
    Parameters:
    -----------
    datasets : list
        List of dataset names to analyze
    log_scale : bool
        Whether to use log scale for the y-axis
    save_path : str
        Directory to save the plots
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # Distribution plots with density curves
    fig, axes = plt.subplots(2, len(datasets), figsize=(18, 10))
    
    for i, dataset in enumerate(datasets):
        print(f"{bb}Processing dataset: {dataset}{rs}")
        
        # Load data
        train_df = load_data(dataset)
        if train_df is None:
            continue
            
        # Compute interaction statistics
        user_counts, item_counts = compute_interaction_stats(train_df)
        
        # Convert to DataFrames for easier plotting
        user_df = pd.DataFrame.from_dict(user_counts, orient='index', columns=['count']).reset_index()
        user_df.columns = ['user_id', 'count']
        
        item_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['count']).reset_index()
        item_df.columns = ['item_id', 'count']
        
        # User interaction distribution with density estimation
        sns.histplot(user_df['count'], ax=axes[0, i], bins=30, stat="density", kde=True, 
                     color="steelblue", edgecolor="darkblue", alpha=0.7, line_kws={"linewidth": 2})
        
        # Attempt to fit normal distribution for comparison
        from scipy import stats
        x = np.linspace(user_df['count'].min(), user_df['count'].max(), 100)
        user_mean, user_std = user_df['count'].mean(), user_df['count'].std()
        axes[0, i].plot(x, stats.norm.pdf(x, user_mean, user_std), 'r-', lw=2, 
                      label=f'Normal: μ={user_mean:.1f}, σ={user_std:.1f}')
        axes[0, i].legend()
        
        axes[0, i].set_title(f'{dataset.upper()} - User Interactions Distribution')
        axes[0, i].set_xlabel('Number of Interactions')
        axes[0, i].set_ylabel('Density')
        
        # Item interaction distribution with density estimation
        sns.histplot(item_df['count'], ax=axes[1, i], bins=30, stat="density", kde=True,
                     color="lightgreen", edgecolor="darkgreen", alpha=0.7, line_kws={"linewidth": 2})
        
        # Attempt to fit normal distribution for comparison
        item_mean, item_std = item_df['count'].mean(), item_df['count'].std()
        x = np.linspace(item_df['count'].min(), item_df['count'].max(), 100)
        axes[1, i].plot(x, stats.norm.pdf(x, item_mean, item_std), 'r-', lw=2, 
                      label=f'Normal: μ={item_mean:.1f}, σ={item_std:.1f}')
        axes[1, i].legend()
        
        axes[1, i].set_title(f'{dataset.upper()} - Item Interactions Distribution')
        axes[1, i].set_xlabel('Number of Interactions')
        axes[1, i].set_ylabel('Density')
        
        # Add dataset statistics as text
        axes[0, i].text(0.65, 0.95, 
                       f'Users: {len(user_counts)}\nTotal: {sum(user_counts.values())}\nAvg: {sum(user_counts.values())/len(user_counts):.1f}\nMedian: {user_df["count"].median():.1f}',
                       transform=axes[0, i].transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[1, i].text(0.65, 0.95, 
                       f'Items: {len(item_counts)}\nTotal: {sum(item_counts.values())}\nAvg: {sum(item_counts.values())/len(item_counts):.1f}\nMedian: {item_df["count"].median():.1f}',
                       transform=axes[1, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Clean up
        del train_df, user_counts, item_counts
        gc.collect()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/interaction_distributions_density.png", dpi=300, bbox_inches='tight')
    print(f"{bg}Density plot saved to {save_path}/interaction_distributions_density.png{rs}")
    plt.close()
    
    # Create plots to compare with standard distribution types
    fig, axes = plt.subplots(2, len(datasets), figsize=(18, 10))
    
    for i, dataset in enumerate(datasets):
        print(f"{bb}Creating distribution comparison for: {dataset}{rs}")
        
        # Load data
        train_df = load_data(dataset, verbose=0)
        if train_df is None:
            continue
            
        # Compute interaction statistics
        user_counts, item_counts = compute_interaction_stats(train_df)
        
        # Convert to DataFrames
        user_df = pd.DataFrame.from_dict(user_counts, orient='index', columns=['count']).reset_index()
        user_df.columns = ['user_id', 'count']
        
        item_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['count']).reset_index()
        item_df.columns = ['item_id', 'count']
        
        # User Q-Q plot (to check for normality)
        from scipy import stats
        stats.probplot(user_df['count'], dist="norm", plot=axes[0, i])
        axes[0, i].set_title(f'{dataset.upper()} - User Interactions Q-Q Plot')
        
        # Item Q-Q plot
        stats.probplot(item_df['count'], dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'{dataset.upper()} - Item Interactions Q-Q Plot')
        
        # Clean up
        del train_df, user_counts, item_counts, user_df, item_df
        gc.collect()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/distribution_qq_plots.png", dpi=300, bbox_inches='tight')
    print(f"{bg}Q-Q plots saved to {save_path}/distribution_qq_plots.png{rs}")
    plt.close()
    
    # Create distribution comparison plots
    fig, axes = plt.subplots(2, len(datasets), figsize=(18, 10))
    
    for i, dataset in enumerate(datasets):
        print(f"{bb}Creating distribution fits for: {dataset}{rs}")
        
        # Load data
        train_df = load_data(dataset, verbose=0)
        if train_df is None:
            continue
            
        # Compute interaction statistics
        user_counts, item_counts = compute_interaction_stats(train_df)
        
        # Convert to DataFrames
        user_df = pd.DataFrame.from_dict(user_counts, orient='index', columns=['count']).reset_index()
        user_df.columns = ['user_id', 'count']
        
        item_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['count']).reset_index()
        item_df.columns = ['item_id', 'count']
        
        # Fit distribution for users
        from scipy import stats
        x = np.linspace(user_df['count'].min(), user_df['count'].max(), 100)
        
        # Get user data
        user_data = user_df['count'].values
        
        # Plot histogram
        axes[0, i].hist(user_data, bins=30, density=True, alpha=0.6, color='steelblue')
        
        # Fit normal distribution
        norm_params = stats.norm.fit(user_data)
        axes[0, i].plot(x, stats.norm.pdf(x, *norm_params), 'r-', 
                      label=f'Normal', linewidth=2)
        
        # Fit log-normal distribution (often better for count data)
        lognorm_params = stats.lognorm.fit(user_data)
        axes[0, i].plot(x, stats.lognorm.pdf(x, *lognorm_params), 'g-', 
                      label=f'Log-normal', linewidth=2)
        
        # Fit power-law distribution (common in recommendation systems)
        # Using exponential as approximation
        exp_params = stats.expon.fit(user_data)
        axes[0, i].plot(x, stats.expon.pdf(x, *exp_params), 'b-', 
                      label=f'Exponential', linewidth=2)
        
        axes[0, i].set_title(f'{dataset.upper()} - User Interactions Distribution Fit')
        axes[0, i].set_xlabel('Number of Interactions')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        
        # Repeat for items
        x = np.linspace(item_df['count'].min(), item_df['count'].max(), 100)
        
        # Get item data
        item_data = item_df['count'].values
        
        # Plot histogram
        axes[1, i].hist(item_data, bins=30, density=True, alpha=0.6, color='lightgreen')
        
        # Fit normal distribution
        norm_params = stats.norm.fit(item_data)
        axes[1, i].plot(x, stats.norm.pdf(x, *norm_params), 'r-', 
                      label=f'Normal', linewidth=2)
        
        # Fit log-normal distribution
        lognorm_params = stats.lognorm.fit(item_data)
        axes[1, i].plot(x, stats.lognorm.pdf(x, *lognorm_params), 'g-', 
                      label=f'Log-normal', linewidth=2)
        
        # Fit power-law distribution
        exp_params = stats.expon.fit(item_data)
        axes[1, i].plot(x, stats.expon.pdf(x, *exp_params), 'b-', 
                      label=f'Exponential', linewidth=2)
        
        axes[1, i].set_title(f'{dataset.upper()} - Item Interactions Distribution Fit')
        axes[1, i].set_xlabel('Number of Interactions')
        axes[1, i].set_ylabel('Density')
        axes[1, i].legend()
        
        # Clean up
        del train_df, user_counts, item_counts, user_df, item_df
        gc.collect()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/distribution_fits.png", dpi=300, bbox_inches='tight')
    print(f"{bg}Distribution fits saved to {save_path}/distribution_fits.png{rs}")
    plt.close()

def plot_comparative_boxplots(datasets, save_path='plots'):
    """
    Create comparative boxplots for user and item interactions across datasets
    
    Parameters:
    -----------
    datasets : list
        List of dataset names to analyze
    save_path : str
        Directory to save the plots
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Containers for data
    user_data = []
    item_data = []
    user_labels = []
    item_labels = []
    
    for dataset in datasets:
        print(f"{bb}Preparing comparative data for: {dataset}{rs}")
        
        # Load data
        train_df = load_data(dataset, verbose=0)
        if train_df is None:
            continue
            
        # Compute interaction statistics
        user_counts, item_counts = compute_interaction_stats(train_df)
        
        # Add to containers
        user_data.append(list(user_counts.values()))
        item_data.append(list(item_counts.values()))
        user_labels.extend([dataset.upper()] * len(user_counts))
        item_labels.extend([dataset.upper()] * len(item_counts))
        
        # Clean up
        del train_df
        gc.collect()
    
    # Create DataFrames for boxplots
    user_df = pd.DataFrame({'Dataset': user_labels, 'Interactions': np.concatenate(user_data)})
    item_df = pd.DataFrame({'Dataset': item_labels, 'Interactions': np.concatenate(item_data)})
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # User boxplot
    sns.boxplot(x='Dataset', y='Interactions', data=user_df, ax=ax1)
    ax1.set_title('User Interactions by Dataset')
    ax1.set_yscale('log')
    
    # Item boxplot
    sns.boxplot(x='Dataset', y='Interactions', data=item_df, ax=ax2)
    ax2.set_title('Item Interactions by Dataset')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/comparative_boxplots.png", dpi=300, bbox_inches='tight')
    print(f"{bg}Comparative boxplots saved to {save_path}/comparative_boxplots.png{rs}")
    plt.close()

def generate_dataset_summary_table(datasets, save_path='plots'):
    """
    Generate a summary table of dataset statistics
    
    Parameters:
    -----------
    datasets : list
        List of dataset names to analyze
    save_path : str
        Directory to save the table
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Data for the table
    data = []
    
    for dataset in datasets:
        print(f"{bb}Generating summary for: {dataset}{rs}")
        
        # Load data
        train_df = load_data(dataset, verbose=0)
        if train_df is None:
            continue
            
        # Compute interaction statistics
        user_counts, item_counts = compute_interaction_stats(train_df)
        
        # Calculate statistics
        n_users = len(user_counts)
        n_items = len(item_counts)
        n_interactions = sum(user_counts.values())
        sparsity = 1 - (n_interactions / (n_users * n_items))
        avg_user_interactions = n_interactions / n_users
        avg_item_interactions = n_interactions / n_items
        
        # Add to data list
        data.append({
            'Dataset': dataset.upper(),
            'Users': n_users,
            'Items': n_items,
            'Interactions': n_interactions,
            'Sparsity': f"{sparsity:.4%}",
            'Avg User': f"{avg_user_interactions:.2f}",
            'Avg Item': f"{avg_item_interactions:.2f}"
        })
        
        # Clean up
        del train_df
        gc.collect()
    
    # Create DataFrame for summary table
    summary_df = pd.DataFrame(data)
    
    # Save to CSV
    summary_df.to_csv(f"{save_path}/dataset_summary.csv", index=False)
    print(f"{bg}Summary table saved to {save_path}/dataset_summary.csv{rs}")
    
    # Create a pretty table with matplotlib
    fig, ax = plt.subplots(figsize=(12, len(datasets) + 1))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Save figure
    plt.savefig(f"{save_path}/dataset_summary_table.png", dpi=300, bbox_inches='tight')
    print(f"{bg}Summary table image saved to {save_path}/dataset_summary_table.png{rs}")
    plt.close()

if __name__ == "__main__":
    # Datasets to analyze
    datasets = ['ml_100k', 'yelp2018', 'amazon_book']
    
    print(f"{bg}Starting interaction distribution analysis...{rs}")
    
    # Plot interaction distributions
    plot_interaction_distribution(datasets)
    
    # Plot comparative boxplots
    plot_comparative_boxplots(datasets)
    
    # Generate summary table
    generate_dataset_summary_table(datasets)
    
    print(f"{bg}Analysis completed!{rs}")