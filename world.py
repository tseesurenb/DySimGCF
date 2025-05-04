'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

from parse import parse_args

args = parse_args()

config = {}
config['batch_size'] = args.batch_size
config['lr'] = args.lr
config['dataset'] = args.dataset
config['layers'] = args.layers
config['emb_dim'] = args.emb_dim
config['model'] = args.model
config['decay'] = args.decay
config['epochs'] = args.epochs
config['top_K'] = args.top_K
config['verbose'] = args.verbose
config['epochs_per_eval'] = args.epochs_per_eval
config['seed'] = args.seed
config['test_ratio'] = args.test_ratio
config['u_sim'] = args.u_sim
config['i_sim'] = args.i_sim
config['sim'] = args.sim
config['edge'] = args.edge
config['i_K'] = args.i_K
config['u_K'] = args.u_K
config['eigen_K'] = args.eigen_K
config['abl_study'] = args.abl_study
config['self_loop'] = bool(args.self_loop)
config['shuffle'] = args.shuffle
config['weighted_neg_sampling'] = args.weighted_neg_sampling
config['samples'] = args.samples
config['save_sim_mat'] = args.save_sim_mat
config['save_res'] = args.save_res
config['save_pred'] = args.save_pred
config['save_model'] = args.save_model
config['margin'] = args.margin
config['load'] = args.load
config['diff'] = args.diff
config['s_temp'] = args.s_temp

# Add these lines to your config setup
config['K'] = args.K if hasattr(args, 'K') else 10.0  # Default to 10.0 if not provided
config['solver'] = args.solver if hasattr(args, 'solver') else 'dopri5'
config['time_split'] = args.time_split if hasattr(args, 'time_split') else 3
config['learnable_time'] = args.learnable_time if hasattr(args, 'learnable_time') else True
config['dual_res'] = args.dual_res if hasattr(args, 'dual_res') else False
config['max_time'] = args.K if hasattr(args, 'K') else 10.0  # Use same value as K