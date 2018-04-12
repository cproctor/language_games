# Here's another attempt to predict user futures, this time with a three-layer neural network based on an embedding. 
# First, I'll go simple and use the initial embedding. Next I'll try to make a TF model which uses the proper monthly
# word vectors. (Or I'll pre-embed the comments. That's probably easier.

# The input will be a 20 * 300 matrix. 


# Load all comments. 
# Group by username.
# Save a npy with username -> comment indices.

# Open train, test, dev

import numpy as np
import pandas as pd
import tensorflow as tf
from settings import *


def feature_class(name, use_max=True, use_min=False):
    "Generates a list of strings for four buckets of features, optionally including min and max"
    suffixes = ['0', '1', '2', '3']
    if use_min: suffixes += ['min']
    if use_max: suffixes += ['max']
    return ["{}_{}".format(name, suffix) for suffix in suffixes]

# Create numpy arrays of features and labels. Features include 10 activity features 
# and 20 * 300 = 6000 BoW representations of the first 20 comments.

if False:
    activity_features = feature_class('freq') + feature_class('month')
    comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=['username'])
    comment_bow_wvs = np.load(HN_SCORED_COMMENT_BOW_WV, 'r')

    for inf, outf in [
        (TRAIN_EXAMPLES, TRAIN_NN_FEATURES), 
        (DEV_EXAMPLES, DEV_NN_FEATURES),
        (TEST_EXAMPLES, TEST_NN_FEATURES)
    ]:
        users = pd.read_csv(inf, usecols=['username', 'label'] + activity_features)
        user_comments = users.merge(comments, left_on='username', right_on='username', how='left')
        wvs = user_comments.groupby('username').apply(lambda g: comment_bow_wvs[g.index[:20]].ravel())
        features = np.concatenate((users[activity_features], np.vstack(wvs)), axis=1)
        labels = users.label.values
        np.savez(outf, features=features, labels=labels)

# Create the neural net and run it.
if True:
    def train_input_fn():
        with np.load(TRAIN_NN_FEATURES) as train:
            train_features = train['features']
            train_labels = train['labels'].reshape((-1,1))
        train_dataset = tf.data.Dataset.from_tensor_slices(({'all_features': train_features}, train_labels))
        train_iterator = train_dataset.make_one_shot_iterator()
        #return train_iterator.get_next()
        return train_dataset

    def dev_input_fn():
        with np.load(DEV_NN_FEATURES) as train:
            dev_features = train['features']
            dev_labels = train['labels'].reshape((-1,1))
        dev_dataset = tf.data.Dataset.from_tensor_slices(({'all_features': dev_features}, dev_labels))
        dev_iterator = dev_dataset.make_one_shot_iterator()
        #return dev_iterator.get_next()
        return dev_dataset

    with np.load(TRAIN_NN_FEATURES) as train:
        train_features = train['features']
        train_labels = train['labels'].reshape((-1,1))
    new_train_input_fn = tf.estimator.inputs.numpy_input_fn({'all_features': train_features}, train_labels, batch_size=1, shuffle=True, num_epochs=1)

    all_features = tf.feature_column.numeric_column('all_features', shape=(6010,1), dtype=tf.float64)

    estimator = tf.estimator.DNNClassifier(
        feature_columns=[all_features],
        hidden_units=[300]
    )

    estimator.train(input_fn=new_train_input_fn)
    #metrics = estimator.evaluate(input_fn=dev_input_fn)
