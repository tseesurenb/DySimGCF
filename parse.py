'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog="DySimGCF", description="Dynamic GCN-based CF recommender")
    parser.add_argument('--model', type=str, default='hyperGCN', help='rec-model, support [LightGCN, NGCF, DySimGCF]')
    parser.add_argument('--dataset', type=str, default='ml_100k', help="available datasets: [ml-100k, yelp2018, amazon-book]")
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--emb_dim', type=int, default=64, help="the embedding size for learning parameters")
    parser.add_argument('--layers', type=int, default=3, help="the layer num of GCN")
    parser.add_argument('--batch_size', type=int, default= 1024, help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int,default=51)
    parser.add_argument('--epochs_per_eval', type=int,default=10)
    parser.add_argument('--verbose', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-04, help="the weight decay for l2 normalizaton")
    parser.add_argument('--s_temp', type=float, default=1.0, help="temperature for softmax normalization")
    parser.add_argument('--top_K', type=int, default=20, help="@k test list")
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--u_sim', type=str, default='cos', help='options are cos (cosine) and jac (jaccard)')
    parser.add_argument('--i_sim', type=str, default='cos', help='options are cos (cosine) and jac (jaccard)')
    parser.add_argument('--sim', type=str, default='ind', help='options are ind (inductive mode) and trans (transductive mode)')
    parser.add_argument('--edge', type=str, default='knn', help='options are knn (similarity graph) and bi (bi-partite graph)')
    parser.add_argument('--u_K', type=int, default=20)
    parser.add_argument('--i_K', type=int, default=20)
    parser.add_argument('--eigen_K', type=int, default=300)
    parser.add_argument('--abl_study', type=int, default=0)
    parser.add_argument('--self_loop', type=bool, default=False)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--weighted_neg_sampling', type=bool, default=False)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--diff', type=bool, default=False)
    parser.add_argument('--save_res', type=bool, default=True)
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_sim_mat', type=bool, default=False)
    parser.add_argument('--margin', type=float, default=0.0, help="the margin in BPR loss")

     # Diffusion/ODE settings
    parser.add_argument('--diff', action='store_true', help='Use diffusion (ODE-based) modeling')
    parser.add_argument('--K', type=float, default=10.0, help="Maximum integration time for ODE")
    parser.add_argument('--solver', type=str, default='dopri5', help="ODE solver [euler, rk4, dopri5]")
    parser.add_argument('--time_split', type=int, default=3, help="Number of time splits")
    parser.add_argument('--learnable_time', action='store_true', help="Use learnable time embeddings")
    parser.add_argument('--dual_res', action='store_true', help="Enable dual residual connections")

    return parser.parse_args()