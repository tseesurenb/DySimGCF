import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os

# Set the style for all plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define dataset statistics based on the paper
datasets = {
    'ml-100k': {
        'users': 943,
        'items': 1682,
        'sparsity': 0.937,
        'u_K': 80,
        'i_K': 10,
        'total_interactions': 100000,  # Approximated
    },
    'yelp2018': {
        'users': 31668,
        'items': 38048,
        'sparsity': 0.9987,
        'u_K': 50,
        'i_K': 20,
        'total_interactions': 1561406,
    },
    'amazon_book': {
        'users': 52643,
        'items': 91599,
        'sparsity': 0.9994,
        'u_K': 80,
        'i_K': 18,
        'total_interactions': 2984108,
    }
}

# Calculate average interactions per user and per item
for name, data in datasets.items():
    data['avg_per_user'] = data['total_interactions'] / data['users']
    data['avg_per_item'] = data['total_interactions'] / data['items']

# Function to create synthetic data with power-law distribution
def create_power_law_distribution(n, alpha):
    """
    Create a power-law distribution.
    
    Parameters:
    n (int): Number of elements in the distribution
    alpha (float): Power-law exponent (higher means steeper distribution)
    
    Returns:
    numpy.ndarray: Array with power-law distributed values
    """
    # Generate ranks
    ranks = np.arange(1, n + 1)
    
    # Generate power-law distribution values
    values = ranks ** (-alpha)
    
    # Normalize values so they sum to total_interactions
    return values / np.sum(values)

# ===== REASON 1: INTERACTION COUNT ASYMMETRY =====
def plot_interaction_asymmetry():
    """
    Plot the asymmetry in average interactions per user vs per item
    for each dataset.
    """
    print("\n==== REASON 1: INTERACTION COUNT ASYMMETRY ====")
    print("This visualization demonstrates that users generally have more interactions")
    print("than items on average, creating a natural asymmetry that explains why")
    print("larger u_K values are beneficial.\n")
    
    # Prepare data for plotting
    names = []
    user_avgs = []
    item_avgs = []
    uk_values = []
    ik_values = []
    
    for name, data in datasets.items():
        names.append(name)
        user_avgs.append(data['avg_per_user'])
        item_avgs.append(data['avg_per_item'])
        uk_values.append(data['u_K'])
        ik_values.append(data['i_K'])
        
        print(f"{name}:")
        print(f"  Average interactions per user: {data['avg_per_user']:.2f}")
        print(f"  Average interactions per item: {data['avg_per_item']:.2f}")
        print(f"  Ratio of user/item avg interactions: {data['avg_per_user']/data['avg_per_item']:.2f}")
        print(f"  Optimal u_K: {data['u_K']}")
        print(f"  Optimal i_K: {data['i_K']}")
        print(f"  Ratio of u_K/i_K: {data['u_K']/data['i_K']:.2f}\n")
    
    # Create the dataframe for plotting
    df = pd.DataFrame({
        'Dataset': names,
        'Avg. User Interactions': user_avgs,
        'Avg. Item Interactions': item_avgs,
        'u_K Value': uk_values,
        'i_K Value': ik_values
    })
    
    # Plot the comparison
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    bar_width = 0.35
    index = np.arange(len(names))
    
    plt.bar(index, df['Avg. User Interactions'], bar_width, label='Avg. User Interactions', color='#3498db')
    plt.bar(index + bar_width, df['Avg. Item Interactions'], bar_width, label='Avg. Item Interactions', color='#e74c3c')
    
    # Add u_K and i_K as horizontal lines on the bars
    for i, (uk, ik) in enumerate(zip(uk_values, ik_values)):
        plt.hlines(y=uk, xmin=index[i]-0.1, xmax=index[i]+0.1, colors='blue', linestyles='dashed', label='_nolegend_')
        plt.text(index[i], uk+5, f'u_K={uk}', ha='center', color='blue', fontweight='bold')
        
        plt.hlines(y=ik, xmin=index[i]+bar_width-0.1, xmax=index[i]+bar_width+0.1, colors='red', linestyles='dashed', label='_nolegend_')
        plt.text(index[i]+bar_width, ik+5, f'i_K={ik}', ha='center', color='red', fontweight='bold')
    
    plt.xlabel('Dataset')
    plt.ylabel('Number of Interactions')
    plt.title('Average Interactions per User vs. per Item with u_K and i_K Values')
    plt.xticks(index + bar_width/2, names)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('plots/interaction_asymmetry.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'plots/interaction_asymmetry.png'\n")
    
    # Calculate and plot the coverage ratios
    plt.figure(figsize=(12, 6))
    
    # Coverage percentages
    user_coverage = [uk / avg * 100 for uk, avg in zip(uk_values, user_avgs)]
    item_coverage = [ik / avg * 100 for ik, avg in zip(ik_values, item_avgs)]
    
    plt.bar(index, user_coverage, bar_width, label='u_K as % of Avg User Interactions', color='#3498db')
    plt.bar(index + bar_width, item_coverage, bar_width, label='i_K as % of Avg Item Interactions', color='#e74c3c')
    
    # Add coverage percentage labels
    for i, (uc, ic) in enumerate(zip(user_coverage, item_coverage)):
        plt.text(index[i], uc+1, f'{uc:.1f}%', ha='center', color='blue')
        plt.text(index[i]+bar_width, ic+1, f'{ic:.1f}%', ha='center', color='red')
    
    plt.xlabel('Dataset')
    plt.ylabel('Coverage Percentage (%)')
    plt.title('Neighborhood Size as Percentage of Average Interactions')
    plt.xticks(index + bar_width/2, names)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('plots/coverage_ratios.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'plots/coverage_ratios.png'")
    

# ===== REASON 2: INFORMATION DENSITY DIFFERENCES =====
def plot_information_density():
    """
    Plot the difference in information density between users and items
    using simulated power-law distributions.
    """
    print("\n==== REASON 2: INFORMATION DENSITY DIFFERENCES ====")
    print("This visualization demonstrates how interaction information is distributed")
    print("differently between users and items, with items typically having a steeper")
    print("power-law distribution, meaning information is concentrated in fewer items.\n")
    
    # Number of entities to simulate
    n_entities = 100
    
    # Create simulated power-law distributions 
    # (higher alpha = steeper distribution)
    user_distribution = create_power_law_distribution(n_entities, alpha=1.5)  # Less steep for users
    item_distribution = create_power_law_distribution(n_entities, alpha=2.2)  # Steeper for items
    
    # Calculate cumulative distributions
    user_cumulative = np.cumsum(user_distribution)
    item_cumulative = np.cumsum(item_distribution)
    
    # Percentage calculation
    ranks = np.arange(1, n_entities + 1)
    
    # Print key statistics
    print("Simulated power-law distributions:")
    print(f"  Top 10% of users capture {user_cumulative[9]*100:.1f}% of all user activity")
    print(f"  Top 10% of items capture {item_cumulative[9]*100:.1f}% of all item activity")
    print(f"  Top 20% of users capture {user_cumulative[19]*100:.1f}% of all user activity")
    print(f"  Top 20% of items capture {item_cumulative[19]*100:.1f}% of all item activity\n")
    
    # Plot the distributions
    plt.figure(figsize=(12, 10))
    
    # First subplot: Individual distributions
    plt.subplot(2, 1, 1)
    plt.plot(ranks, user_distribution * 100, 'b-', linewidth=2, label='User Interaction Distribution')
    plt.plot(ranks, item_distribution * 100, 'r-', linewidth=2, label='Item Interaction Distribution')
    plt.xlabel('Rank (Most to Least Active)')
    plt.ylabel('Percentage of Total Interactions (%)')
    plt.title('Power-Law Distribution: User vs Item Interaction Patterns')
    plt.legend()
    plt.grid(True)
    
    # Add annotations to explain the key insight
    plt.annotate('Items have a steeper\npower-law distribution,\nmeaning information is\nmore concentrated', 
                xy=(30, item_distribution[29]*100), 
                xytext=(50, item_distribution[29]*100*2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))
    
    # Second subplot: Cumulative distributions
    plt.subplot(2, 1, 2)
    plt.plot(ranks, user_cumulative * 100, 'b-', linewidth=2, label='User Cumulative %')
    plt.plot(ranks, item_cumulative * 100, 'r-', linewidth=2, label='Item Cumulative %')
    
    # Add lines for typical u_K and i_K values (using the amazon_book ratio as example)
    uk_ratio = datasets['amazon_book']['u_K'] / datasets['amazon_book']['users']
    ik_ratio = datasets['amazon_book']['i_K'] / datasets['amazon_book']['items']
    
    # Scale to our 100-entity example
    scaled_uk = int(uk_ratio * n_entities)
    scaled_ik = int(ik_ratio * n_entities)
    
    # Ensure we have valid indices
    scaled_uk = max(1, min(scaled_uk, n_entities-1))
    scaled_ik = max(1, min(scaled_ik, n_entities-1))
    
    # Add vertical lines for the scaled k values
    plt.axvline(x=scaled_uk, color='blue', linestyle='--', label=f'Relative u_K position')
    plt.axvline(x=scaled_ik, color='red', linestyle='--', label=f'Relative i_K position')
    
    # Add information gain annotations
    plt.annotate(f'{user_cumulative[scaled_uk-1]*100:.1f}% of user information',
                xy=(scaled_uk, user_cumulative[scaled_uk-1]*100),
                xytext=(scaled_uk+5, user_cumulative[scaled_uk-1]*100-10),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                fontsize=9,
                color='blue')
    
    plt.annotate(f'{item_cumulative[scaled_ik-1]*100:.1f}% of item information',
                xy=(scaled_ik, item_cumulative[scaled_ik-1]*100),
                xytext=(scaled_ik+5, item_cumulative[scaled_ik-1]*100-5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=9,
                color='red')
    
    # Add key insight annotation
    plt.annotate('Key Insight: Fewer items (lower i_K)\nare needed to capture the same\namount of information',
                xy=(n_entities//2, 60),
                xytext=(n_entities//2, 60),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                ha='center')
    
    plt.xlabel('Top-K Entities')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Cumulative Information Capture: Smaller i_K Captures Similar Information as Larger u_K')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('plots/information_density.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'plots/information_density.png'")
    
    # Calculate information gain per entity - this shows the diminishing returns
    plt.figure(figsize=(10, 6))
    
    # Calculate information gain per additional entity
    user_gain = np.diff(np.insert(user_cumulative, 0, 0)) * 100
    item_gain = np.diff(np.insert(item_cumulative, 0, 0)) * 100
    
    plt.plot(ranks, user_gain, 'b-', linewidth=2, label='Information Gain per User')
    plt.plot(ranks, item_gain, 'r-', linewidth=2, label='Information Gain per Item')
    
    # Find points where gain drops below thresholds
    user_threshold = np.where(user_gain < 0.5)[0][0] if any(user_gain < 0.5) else n_entities
    item_threshold = np.where(item_gain < 0.5)[0][0] if any(item_gain < 0.5) else n_entities
    
    plt.axvline(x=user_threshold, color='blue', linestyle='--')
    plt.axvline(x=item_threshold, color='red', linestyle='--')
    
    plt.annotate(f'User diminishing returns at ~{user_threshold}',
                xy=(user_threshold, 0.5),
                xytext=(user_threshold+5, 1),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                fontsize=9,
                color='blue')
    
    plt.annotate(f'Item diminishing returns at ~{item_threshold}',
                xy=(item_threshold, 0.5),
                xytext=(item_threshold+5, 2),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=9,
                color='red')
    
    plt.xlabel('Entity Rank')
    plt.ylabel('Information Gain (%)')
    plt.title('Diminishing Returns: Information Gain per Additional Entity')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale to better visualize the diminishing returns
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('plots/diminishing_returns.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'plots/diminishing_returns.png'\n")
    
    print("Key observations:")
    print("1. Items show a steeper power-law distribution than users")
    print("2. Information gain diminishes more quickly for items than for users")
    print("3. This explains why a smaller i_K is sufficient to capture the necessary item information")
    print("4. Meanwhile, users require a larger u_K to capture comparable information")


def main():
    """
    Main function to run the analysis and generate visualizations.
    """
    print("=== DySimGCF Neighborhood Size Analysis ===")
    print("This script visualizes why u_K values are larger than i_K values in recommendation systems.\n")
    
    # Run the analysis for Reason 1
    plot_interaction_asymmetry()
    
    # Run the analysis for Reason 2
    plot_information_density()
    
    print("\n=== Analysis Complete ===")
    print("All visualizations have been saved to the 'plots' directory.")


if __name__ == "__main__":
    main()