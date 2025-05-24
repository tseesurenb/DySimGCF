
'''
Created on May 1, 2025
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

 -- To run an experiment with ml-100k dataset, use the following command:
   python main.py --layers=3 --model=DySimGCF  --epochs=1001 --verbose=1 --u_K=80 --i_K=10 --samples=50 --margin=0.03 --decay=1e-03 --e_drop=0.5 --s_temp=0.1 

 -- To run an experiment with yelp2018 dataset, use the following command:
   python main.py --layers=4 --model=DySimGCF  --epochs=501 --verbose=1 --dataset=yelp2018  --u_K=50 --i_K=20 --samples=100 --margin=0.1 --decay=1e-04 --e_drop=0.5 --s_temp=0.1

 -- To run an experiment with amazon-book dataset, use the following command:
   python main.py --layers=3 --model=DySimGCF  --epochs=401 --verbose=1 --dataset=amazon_book  --u_K=40 --i_K=5 --samples=40 --margin=0.1 --decay=1e-05 --e_drop=0.7 --s_temp=0.2


-- To run an experiment with ml-100k for DySimGCF in transductive mode (creating similarity matrices using user and movie feature data), use the following command:
   python main.py --layers=3 --decay=1e-03 --model=DySimGCF  --epochs=1001 --verbose=1 --u_K=900 --i_K=900 --sim=trans # here we only have one option of using ml-100k dataset.

