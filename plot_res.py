'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_results(plot_name, file_names):
    plt.figure(figsize=(14, 6))  # Increased figure height for better readability
    
    # Iterate over the list of files
    for file_idx, file_name in enumerate(file_names):
        # Load saved results with allow_pickle=True
        all_losses = np.load(file_name[0], allow_pickle=True)
        all_metrics = np.load(file_name[1], allow_pickle=True)

        num_exp = len(all_losses)
        num_test_epochs = len(all_losses[0]['total_loss'])
        epoch_list = [(j + 1) for j in range(num_test_epochs)]

        for i in range(num_exp):
            
            if file_idx == 0:
                label_suffix = f"LightGCN"
            elif file_idx == 1:
                label_suffix = f"DySimGCF"
            else:
                label_suffix = r"DySimGCF$^{i}$"

            # Plot losses
            plt.subplot(2, 3, 1)
            plt.plot(epoch_list, all_losses[i]['total_loss'], label=f'{label_suffix}', linestyle='-', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.ylim(ymax=0.5)  # Set minimum y-axis value to 0.15
            plt.legend()

            # Plot Recall and Precision
            plt.subplot(2, 3, 2)
            plt.plot(epoch_list, all_metrics[i]['recall'], label=f'{label_suffix} - Recall', linestyle='-', alpha=0.7)
            plt.plot(epoch_list, all_metrics[i]['precision'], label=f'{label_suffix} - Precision', linestyle='--', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Recall & Precision')
            plt.title('Recall & Precision')
            plt.ylim(ymin=0.15)  # Set minimum y-axis value to 0.15
            plt.legend()

            # Plot NDCG
            plt.subplot(2, 3, 3)
            plt.plot(epoch_list, all_metrics[i]['ncdg'], label=f'{label_suffix}', linestyle='-', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('NCDG')
            plt.title('NDCG')
            plt.ylim(ymin=0.2)  # Set minimum y-axis value to 0.15
            plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots

    # Save or display the plot
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    if plot_name is not None:
        plt.savefig(f"{plot_name}_{timestamp}.png")  # Save plot to file
    else:
        plt.show()

res_dir = "models/results"

exp_n = 2

file_name1 = "lightGCN_cuda_2020_ml-100k_1024__3_1001_bi"
file_name2 = "hyperGAT_cuda_2020_ml-100k_1024__3_1001_knn"
file_name3 = "hyperGAT_cuda_2020_ml-100k_1024__3_1001_knn_f"

# Construct file paths for each experiment
file_paths = [
    (f"{res_dir}/{file_name1}_all_losses.npy", f"{res_dir}/{file_name1}_all_metrics.npy"),
    (f"{res_dir}/{file_name2}_all_losses.npy", f"{res_dir}/{file_name2}_all_metrics.npy"),
    (f"{res_dir}/{file_name3}_all_losses.npy", f"{res_dir}/{file_name3}_all_metrics.npy")
]

# Plot and save the results
plot_save_dir = "models/plots_2/"
plot_results(plot_save_dir, file_paths)
