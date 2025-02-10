
'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

 -- To run an experiment with ml-100k dataset, use the following command:
   python main.py --layers=3 --decay=1e-03 --model=DySimGCF  --epochs=1001 --verbose=1 --u_K=80 --i_K=10

 -- To run an experiment with yelp2018 dataset, use the following command:
   python main.py --layers=4 --decay=1e-04  --model=DySimGCF  --epochs=1001 --u_K=50 --i_K=20  --verbose=1 

 -- To run an experiment with amazon-book dataset, use the following command:
   python main.py --layers=1 --decay=1e-05 --u_K=80 --i_K=18  --model=DySimGCF  --epochs=1001 --verbose=1

 -- To run an experiment with ml-100k for LightGCN, use the following command:
   python main.py --layers=3 --decay=1e-03 --model=LightGCN --edge=bi  --epochs=1001 --verbose=1


-- To run an experiment with ml-100k for DySimGCF in transductive mode (creating similarity matrices using user and movie feature data), use the following command:
   python main.py --layers=3 --decay=1e-03 --model=DySimGCF  --epochs=1001 --verbose=1 --u_K=900 --i_K=900 --sim=trans # here we only have one option of using ml-100k dataset.

