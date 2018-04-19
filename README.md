# BCA
Extension of hierarchical attention network for text classification. 

Tensorflow implementation of basic hierarchical attention network (HAN) from  "Hierarchical Attention Networks for Document Classification", Zichao Yang et al. (http://www.aclweb.org/anthology/N16-1174).
There is also an extension to include bi-directional context across sentences (BCA). 

The models have been set up to train and validate on yelp data (https://www.yelp.com/dataset/challenge). 
The data (review.json) can be preprocessed using yelp_preprocess.py. 

The scripts han_train.py and bca_train.py can be used to train the models. There are two notebooks set up to run the trained models on the test set and visualize both word and sentence level attention. 

Requirements: 

Tensorflow 1.0 or greater. 

Codes are just for testing, there has been no hyper parameter tuning. 
