# Instead of using a distance-from-community metric (log-likelihood)
# leverage the distributed representation quality of word vectors

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from pathlib import Path
from helpers import 
from settings import *

TRAIN_MODELS = True # As opposed to using pretrained models

results = []

def evaluate_tf_preds(preds, labels, dataset, desc):
    """
    Returns precision, recall, and F1, suitable for usual results reporting
    Expects preds to be the result of estimator.predict
    """
    classes = [x['class_ids'][0] for x in preds]
    p, r, f1, s = precision_recall_fscore_support(classes, labels.ravel())
    return [dataset, desc, p[1], r[1], f1[1], s[1]]

# Straight-up logistic regression (effectively 0-hidden-layer neural network) as a baseline. 
if True:
    # With monthly embeddings
    with np.load(TRAIN_NN_FEATURES) as train:
        train_features = train['features']
        train_labels = train['labels']
    with np.load(DEV_NN_FEATURES) as dev:
        dev_features = dev['features']
        dev_labels = dev['labels']
    model = LogisticRegression()
    model.fit(train_features, train_labels)
    yHat = model.predict(dev_features)
    p, r, f1, s = precision_recall_fscore_support(dev_labels, yHat)
    results.append(["Dev", "BoW WVs Logistic", p[1], r[1], f1[1], s[1]])

     # With weighted monthly embeddings
    with np.load(WEIGHTED_TRAIN_NN_FEATURES) as w_train:
        w_train_features = w_train['features']
        w_train_labels = w_train['labels']
    with np.load(WEIGHTED_DEV_NN_FEATURES) as w_dev:
        w_dev_features = w_dev['features']
        w_dev_labels = w_dev['labels']
    model = LogisticRegression()
    model.fit(w_train_features, w_train_labels)
    w_yHat = model.predict(w_dev_features)
    w_p, w_r, w_f1, w_s = precision_recall_fscore_support(w_dev_labels, w_yHat)
    results.append(["Dev", "Weighted BoW WVs Logistic", w_p[1], w_r[1], w_f1[1], w_s[1]])
      
    if True: 
        # With original embedding
        with np.load(BASELINE_TRAIN_NN_FEATURES) as train:
            b_train_features = train['features']
            b_train_labels = train['labels']
        with np.load(BASELINE_DEV_NN_FEATURES) as dev:
            b_dev_features = dev['features']
            b_dev_labels = dev['labels']
        b_model = LogisticRegression()
        b_model.fit(b_train_features, b_train_labels)
        b_yHat = b_model.predict(b_dev_features)
        b_p, b_r, b_f1, b_s = precision_recall_fscore_support(b_dev_labels, b_yHat)
        results.append(["Dev", "Initial BoW WVs Logistic", b_p[1], b_r[1], b_f1[1], b_s[1]])
    
        # With diff embedding
        diff_model = LogisticRegression()
        diff_model.fit(train_features - b_train_features, train_labels)
        diff_yhat = diff_model.predict(dev_features - b_dev_features)
        p, r, f1, s = precision_recall_fscore_support(dev_labels, diff_yhat)
        results.append(["Dev", "Diff BoW WVs Logistic", p[1], r[1], f1[1], s[1]])
    
        # With both embeddings
        plus_model = LogisticRegression()
        train_plus_features = np.concatenate([train_features, b_train_features[:, 10:]], axis=1)
        dev_plus_features = np.concatenate([dev_features, b_dev_features[:, 10:]], axis=1)
    
        plus_model.fit(train_plus_features, train_labels)
        plus_yhat = plus_model.predict(dev_plus_features)
        p, r, f1, s = precision_recall_fscore_support(dev_labels, plus_yhat)
        results.append(["Dev", "Plus BoW WVs Logistic", p[1], r[1], f1[1], s[1]])

# =======================================================================================
# Try this with a neural net.
if True:
    use_baseline = False
    # Built input functions.
    # Let's be more responsible and do our experiments on slices of the training data.
    # We'll do simple cross-validation
    with np.load(TRAIN_NN_FEATURES) as train:
        train_features = train['features']
        train_labels = train['labels'].reshape((-1,1))
    with np.load(DEV_NN_FEATURES) as dev:
        dev_features = dev['features']
        dev_labels = dev['labels'].reshape((-1,1))
    with np.load(WEIGHTED_TRAIN_NN_FEATURES) as w_train:
        w_train_features = w_train['features']
        w_train_labels = w_train['labels']
    with np.load(WEIGHTED_DEV_NN_FEATURES) as w_dev:
        w_dev_features = w_dev['features']
        w_dev_labels = w_dev['labels']

    # Skipping CV for now
    CV_RATIO = 0.2
    cv_train_size = round(train_features.shape[0] * (1-CV_RATIO))
    cv_train_features, cv_test_features = train_features[:cv_train_size], train_features[cv_train_size:]
    cv_w_train_features, cv_w_test_features = w_train_features[:cv_train_size], w_train_features[cv_train_size:]
    cv_train_labels, cv_test_labels = train_labels[:cv_train_size], train_labels[cv_train_size:]

    # Input functions
    BATCH_SIZE = 96
    NUM_EPOCHS = 40

    def create_input_fn(activity, monthly, labels, train=True):
        return tf.estimator.inputs.numpy_input_fn(
            {'activity': activity, 'monthly_wv': monthly}, 
            labels, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_epochs=NUM_EPOCHS if train else 1
        )

    # Creating input functions for CV, which we're not using now.
    cv_train_input_fn = create_input_fn(cv_train_features[:, :10], cv_train_features[:, 10:], cv_train_labels)
    cv_test_input_fn = create_input_fn(cv_test_features[:, :10], cv_test_features[:, 10:], cv_test_labels, 
        train=False)

    # Train and dev input functions for unweighted word vectors
    train_input_fn = create_input_fn(train_features[:, :10], train_features[:, 10:], train_labels)
    dev_input_fn = create_input_fn(dev_features[:, :10], dev_features[:, 10:], dev_labels, train=False)

    # Train and dev input functions for weighted word vectors
    w_train_input_fn = create_input_fn(w_train_features[:, :10], w_train_features[:, 10:], w_train_labels)
    w_dev_input_fn = create_input_fn(w_dev_features[:, :10], w_dev_features[:, 10:], w_dev_labels, train=False)

    # Features
    activity_features = tf.feature_column.numeric_column('activity', 
            shape=(10, ), dtype=tf.float64)
    monthly_wv_features = tf.feature_column.numeric_column('monthly_wv', 
            shape=(6000, ), dtype=tf.float64)

    def evaluate_model(train_fn, predict_fn, labels, hyperparams, dataset, description, train=True):
        estimator = tf.estimator.DNNClassifier(model_dir=Path(DNN_MODEL_DIR) / Path(description), **hyperparams)
        if train:
            estimator.train(input_fn=train_fn)
        estimator.evaluate(train_fn, steps=10000, name=description + " TRAIN")
        estimator.evaluate(predict_fn, steps=10000, name=description + " TEST")
        preds = estimator.predict(input_fn=predict_fn)
        results.append(evaluate_tf_preds(preds, labels, dataset, description))

    def grid_search(base, param, values):
        grid = []
        for value in values:
            case = dict(base)
            case[param] = value
            grid.append(case)
        return grid

    # Models
    nn_model_params = {
        'feature_columns': [activity_features, monthly_wv_features],
        'hidden_units': [1024, 512, 256],
        'dropout': 0.5,
        'optimizer': tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.01
        )
    }

    # BASE CASE FOR WEIGHTED AND UNWEIGHTED
    if True:
        evaluate_model(train_input_fn, dev_input_fn, dev_labels, nn_model_params, "DEV", 
                "unweighted embedding", train=TRAIN_MODELS)
        evaluate_model(w_train_input_fn, w_dev_input_fn, w_dev_labels, nn_model_params, "DEV", 
                "weighted embedding", train=TRAIN_MODELS)

    # TEST NUMBER OF LAYERS
    if False:
        depths = [[1024, 512, 256], [512, 256], [256]]
        names = ["Monthly {} layer".format(i) for i in [3,2,1]]
        for name, model in zip(names, grid_search(cv_3layer_monthly, 'hidden_units', depths)):
            evaluate_model(cv_train_input_fn, cv_test_input_fn, cv_test_labels, model, "CV", name, 
                    train=TRAIN_MODELS)
    
    # TEST AMOUNT OF DROPOUT
    if False:
        drops = np.linspace(0.1, 0.6, 6)
        names = ["2 Monthly drop {:.1f}".format(i) for i in drops]
        for name, model in zip(names, grid_search(cv_3layer_monthly, 'dropout', drops)):
            evaluate_model(cv_train_input_fn, cv_test_input_fn, cv_test_labels, model, "CV", name, 
                    train=TRAIN_MODELS)

    # TEST ADDITION OF INITIAL
    if False:
        evaluate_model(train_input_fn, dev_input_fn, dev_labels, cv_3layer_monthly, "DEV", "Initial")
        evaluate_model(train_input_fn, dev_input_fn, dev_labels, cv_3layer_monthly, "DEV", "Initlal+Monthly")


print(tabulate(results, headers=['dataset', 'model', 'precision', 'recall', 'f1', 'support']))
