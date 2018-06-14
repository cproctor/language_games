# When using snapshot embeddings, we want to pre-compute comment vectors 
# for bag-of-words representations of ia user's first n 
# comments. The result is a |U| * (n * d) numpy array, where
# |U| is the number of users, n is the number of comments to include, and d is 
# the dimension of the vectors
# June 13: Complete

import numpy as np
import pandas as pd
from settings import *

def collect_comment_bow_features(users, comments, bowEmbedding, outFile, nComments=20):
    """
    Given a list of users (in userFile), gets the first n comments for each user, 

    Params:
        users: df containing a username column
        comments: df containing a username column. Should be sorted (by date).
        bowEmbedding: |C| * d ndarray, where |C| is the number
            of comments in CommentsFile and d is the dimension of word vectors. 
            Each row is a bag-of-words representation of the corresponding comment.
        outFile: location for a np file containing a |U| * (d*n) array. For each 
            user, this will be a concatenation of comments.
    """
    comments = comments[comments.username.isin(users.username)]
    commentWVs = comments.groupby('username', sort=False).apply(
            lambda g: bowEmbedding[g.index[:nComments]].ravel())
    np.save(outFile, np.vstack(commentWVs))
    

comments = pd.read_csv(HN_SCORED_COMMENTS, usecols=['username'])
train = pd.read_csv(TRAIN_EXAMPLES)
dev = pd.read_csv(DEV_EXAMPLES)
test = pd.read_csv(TEST_EXAMPLES)

if False:
    emb = np.load(HN_SCORED_COMMENT_BOW_WV, 'r')
    print("loaded emb")
    collect_comment_bow_features(train, comments, emb, TRAIN_NN_FEATURES)
    print("1")
    collect_comment_bow_features(dev,   comments, emb, DEV_NN_FEATURES)
    print("2")
    collect_comment_bow_features(test,  comments, emb, TEST_NN_FEATURES)
    print("3")

if False:
    emb = np.load(HN_SCORED_COMMENT_BOW_WV_BASELINE, 'r')
    print("loaded emb")
    collect_comment_bow_features(train, comments, emb, BASELINE_TRAIN_NN_FEATURES)
    print("4")
    collect_comment_bow_features(dev,   comments, emb, BASELINE_DEV_NN_FEATURES)
    print("5")
    collect_comment_bow_features(test,  comments, emb, BASELINE_TEST_NN_FEATURES)
    print("6")
        
if True:
    emb = np.load(HN_SCORED_COMMENT_GLOVE_BOW_WV, 'r')
    print("loaded emb")
    collect_comment_bow_features(train, comments, emb, GLOVE_TRAIN_NN_FEATURES)
    print("4")
    collect_comment_bow_features(dev,   comments, emb, GLOVE_DEV_NN_FEATURES)
    print("5")
    collect_comment_bow_features(test,  comments, emb, GLOVE_TEST_NN_FEATURES)
    print("6")
        
