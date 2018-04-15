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
    "Given a feature name, adds a new feature indicating the index of the minimum value for that feature"
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_min': np.argmin(data[features].values, axis=1)})

def add_max(data, feature_class_name):
    "Given a feature name, adds a new feature indicating the index of the maximum value for that feature"
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_max': np.argmax(data[features].values, axis=1)})

def feature_class(name, use_max=True, use_min=False):
    "Generates a list of strings for four buckets of features, optionally including min and max"
    suffixes = ['0', '1', '2', '3']
    if use_min: suffixes += ['min']
    if use_max: suffixes += ['max']
    return ["{}_{}".format(name, suffix) for suffix in suffixes]

## Load the pre-generated examples from csv. They are stored in a number of different
## files, so we need to load them separately and merge them.

train = pd.read_csv(TRAIN_EXAMPLES)
train_y = train.label.values
dev = pd.read_csv(DEV_EXAMPLES)
dev_y = dev.label.values
test = pd.read_csv(TEST_EXAMPLES)
test_y = test.label.values

train = add_min(train, 'bigram')
dev = add_min(dev, 'bigram')
test = add_min(test, 'bigram')

train = add_min(train, 'wv')
dev = add_min(dev, 'wv')
test = add_min(test, 'wv')

initial_model_features = pd.read_csv(WV_INITIAL_MODEL_FEATURES)
train = train.merge(initial_model_features, left_on='username', right_on='username')
dev = dev.merge(initial_model_features, left_on='username', right_on='username')
test = test.merge(initial_model_features, left_on='username', right_on='username')

for diff_wv, wv, initial_wv in zip(feature_class('diff_wv', use_max=False), 
        feature_class('wv', use_max=False), feature_class('initial_wv', use_max=False)):
    train = train.assign(**{diff_wv: train[wv] - train[initial_wv]})
    dev = dev.assign(**{diff_wv: dev[wv] - dev[initial_wv]})
    test = test.assign(**{diff_wv: test[wv] - test[initial_wv]})

train = add_max(train, 'diff_wv')
train = add_min(train, 'diff_wv')
dev = add_max(dev, 'diff_wv')
dev = add_min(dev, 'diff_wv')
test = add_max(test, 'diff_wv')
test = add_min(test, 'diff_wv')

results = []
USEMAX = True
USEMIN = True

def evaluate_model(description, trainX, devX, trainY, devY, with_train=False):
    print(description)
    model = LogisticRegression()
    model.fit(trainX, trainY)
    results = []
    if with_train:
        yHat = model.predict(trainX)
        p, r, f1, s = precision_recall_fscore_support(trainY, yHat)
        results.append(["Train", description, p[1], r[1], f1[1], s[1]])
    print(devX.columns)
    yHat = model.predict(devX)
    p, r, f1, s = precision_recall_fscore_support(devY, yHat)
    results.append(["Dev", description, p[1], r[1], f1[1], s[1]])
    return results

results += evaluate_model("Activity",
    train[feature_class('freq') + feature_class('month')],
    test[feature_class('freq') + feature_class('month')],
    train_y, test_y
)
results += evaluate_model("Activity + Bigram",
    train[feature_class('freq') + feature_class('month') + feature_class('bigram', use_max=USEMAX, use_min=USEMIN)],
    test[feature_class('freq') + feature_class('month') + feature_class('bigram', use_max=USEMAX, use_min=USEMIN)],
    train_y, test_y
)
results += evaluate_model("Activity + WV",
    train[feature_class('freq') + feature_class('month') + feature_class('wv', use_max=USEMAX, use_min=USEMIN)],
    test[feature_class('freq') + feature_class('month') + feature_class('wv', use_max=USEMAX, use_min=USEMIN)],
    train_y, test_y
)

features = (
    feature_class('freq') + 
    feature_class('month') + 
    feature_class('diff_wv', use_max=USEMAX, use_min=USEMIN)
)
results += evaluate_model("Activity + Diff WV", train[features], test[features], train_y, test_y)

features = (
    feature_class('freq') + 
    feature_class('month') + 
    feature_class('wv', use_max=USEMAX, use_min=USEMIN) + 
    feature_class('diff_wv', use_max=USEMAX, use_min=USEMIN)
)
#results += evaluate_model("Activity + WV + Diff WV", train[features], test[features], train_y, test_y)



print(tabulate(results, headers=['dataset', 'model', 'precision', 'recall', 'f1', 'support']))








