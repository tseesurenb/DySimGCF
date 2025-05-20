'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

from parse import parse_args

args = parse_args()

config = {
    'batch_size': args.batch_size,
    'lr': args.lr,
    'dataset': args.dataset,
    'layers': args.layers,
    'emb_dim': args.emb_dim,
    'model': args.model,
    'decay': args.decay,
    'epochs': args.epochs,
    'top_K': args.top_K,
    'verbose': args.verbose,
    'epochs_per_eval': args.epochs_per_eval,
    'seed': args.seed,
    'test_ratio': args.test_ratio,
    'u_sim': args.u_sim,
    'i_sim': args.i_sim,
    'sim': args.sim,
    'edge': args.edge,
    'i_K': args.i_K,
    'u_K': args.u_K,
    'eigen_K': args.eigen_K,
    'abl_study': args.abl_study,
    'self_loop': bool(args.self_loop),
    'shuffle': args.shuffle,
    'weighted_neg_sampling': args.weighted_neg_sampling,
    'samples': args.samples,
    'save_sim_mat': args.save_sim_mat,
    'save_res': args.save_res,
    'save_pred': args.save_pred,
    'save_model': args.save_model,
    'margin': args.margin,
    'load': args.load,
    'diff': args.diff,
    's_temp': args.s_temp,
    'l_temp': args.l_temp,
    'l_weight': args.l_weight
}
