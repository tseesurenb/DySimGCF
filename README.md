
'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

   python main.py --layers=1 --decay=1e-03 --model=DySimGCF  --epochs=201 --verbose=1

 -- if to run an experiment with only one negative sample, run the following command:
    python main.py --layers=1 --u_K=15 --decay=1e-03 --i_K=50  --model=DySimGCF  --epochs=201

 -- if to run an experiment with multiple negative samples, run the following command:

    python main.py --layers=1 --u_K=15 --decay=1e-03 --i_K=50  --model=DySimGCF  --epochs=201 --neg_samples=10