# Once labeled examples have been generated for train, dev, and test, 
# we can create a logistic model predicting whether users are likely
# to remain or depart.

from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from settings import *
from tabulate import tabulate
import numpy as np

def add_min(data, feature_class_name):
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_min': np.argmin(data[features].values, axis=1)})

def add_max(data, feature_class_name):
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_max': np.argmax(data[features].values, axis=1)})

def feature_class(name, use_max=True, use_min=False):
    suffixes = ['0', '1', '2', '3']
    if use_min: suffixes += ['min']
    if use_max: suffixes += ['max']
    return ["{}_{}".format(name, suffix) for suffix in suffixes]

train = pd.read_csv(TRAIN_EXAMPLES)
train_y = train.label.values
dev = pd.read_csv(DEV_EXAMPLES)
dev_y = dev.label.values
train = add_min(train, 'bigram')
dev = add_min(dev, 'bigram')
train = add_min(train, 'wv')
dev = add_min(dev, 'wv')

initial_model_features = pd.read_csv(WV_INITIAL_MODEL_FEATURES)
train = train.merge(initial_model_features, left_on='username', right_on='username')
dev = dev.merge(initial_model_features, left_on='username', right_on='username')

for diff_wv, wv, initial_wv in zip(feature_class('diff_wv', use_max=False), 
        feature_class('wv', use_max=False), feature_class('initial_wv', use_max=False)):
    train = train.assign(**{diff_wv: train[wv] - train[initial_wv]})
    dev = dev.assign(**{diff_wv: dev[wv] - dev[initial_wv]})

train = add_max(train, 'diff_wv')
train = add_min(train, 'diff_wv')
dev = add_max(dev, 'diff_wv')
dev = add_min(dev, 'diff_wv')

results = []

def evaluate_model(description, trainX, devX, trainY, devY):
    model = LogisticRegression()
    model.fit(trainX, trainY)
    yHat = model.predict(trainX)
    results = []
    p, r, f1, s = precision_recall_fscore_support(trainY, yHat)
    results.append(["Train", description, p[1], r[1], f1[1], s[1]])
    yHat = model.predict(devX)
    p, r, f1, s = precision_recall_fscore_support(devY, yHat)
    results.append(["Dev", description, p[1], r[1], f1[1], s[1]])
    return results

results += evaluate_model("Activity",
    train[feature_class('freq') + feature_class('month')],
    dev[feature_class('freq') + feature_class('month')],
    train_y, dev_y
)
results += evaluate_model("Activity + Bigram",
    train[feature_class('freq') + feature_class('month') + feature_class('bigram', use_max=False, use_min=True)],
    dev[feature_class('freq') + feature_class('month') + feature_class('bigram', use_max=False, use_min=True)],
    train_y, dev_y
)
results += evaluate_model("Activity + WV",
    train[feature_class('freq') + feature_class('month') + feature_class('wv', use_max=False, use_min=True)],
    dev[feature_class('freq') + feature_class('month') + feature_class('wv', use_max=False, use_min=True)],
    train_y, dev_y
)
results += evaluate_model("Activity + Initial WV",
    train[feature_class('freq') + feature_class('month') + feature_class('initial_wv', use_max=False, use_min=True)],
    dev[feature_class('freq') + feature_class('month') + feature_class('initial_wv', use_max=False, use_min=True)],
    train_y, dev_y
)

features = (
    feature_class('freq') + 
    feature_class('month') + 
    feature_class('wv', use_max=False, use_min=True) + 
    feature_class('initial_wv', use_max=False, use_min=True)
)
results += evaluate_model("Activity + Initial WV + WV", train[features], dev[features], train_y, dev_y)

features = (
    feature_class('freq') + 
    feature_class('month') + 
    feature_class('diff_wv', use_max=True, use_min=True)
)
results += evaluate_model("Activity + Diff WV", train[features], dev[features], train_y, dev_y)

features = (
    feature_class('freq') + 
    feature_class('month') + 
    feature_class('wv', use_max=False, use_min=True) + 
    feature_class('diff_wv', use_max=False, use_min=True)
)
results += evaluate_model("Activity + WV + Diff WV", train[features], dev[features], train_y, dev_y)



print(tabulate(results, headers=['dataset', 'model', 'precision', 'recall', 'f1', 'support'], tablefmt='html'))








