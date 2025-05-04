'''
Created on April 9, 2025
Generate publication-quality figures for DySimGCF paper from experiment results
'''

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Use a publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.figsize': (9, 6),
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def parse_args():
    parser = argparse.ArgumentParser(description='Generate paper figures from experiment results')
    parser.add_argument('--results_dir', type=str, default='./models/robustness_results',
                        help='Directory containing experiment result files')
    parser.add_argument('--robustness_file', type=str, default=None,
                        help='Specific robustness results CSV file (if not specified, latest is used)')
    parser.add_argument('--cold_start_file', type=str, default=None,
                        help='Specific cold-start results CSV file (if not specified, latest is used)')
    parser.add_argument('--output_dir', type=str, default='./paper_figures',
                        help='Directory to save generated figures')
    parser.add_argument('--model_names', type=str, default='DySimGCF,LightGCN',
                        help='Names to use for models in figure legends')
    return parser.parse_args()

def find_latest_file(directory, prefix):
    """Find the most recent file with the given prefix in the directory"""
    files = [f for f in os.listdir(directory) if