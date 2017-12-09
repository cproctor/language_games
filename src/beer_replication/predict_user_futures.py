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

def feature_class(name):
    return ["{}_{}".format(name, suffix) for suffix in ['0', '1', '2', '3', 'max']]

train = pd.read_csv(TRAIN_EXAMPLES)
train_y = train.label.values
dev = pd.read_csv(DEV_EXAMPLES)
dev_y = dev.label.values

print("ACTIVITY")
train_X = train[feature_class('freq') + feature_class('month')]
dev_X = dev[feature_class('freq') + feature_class('month')]
model = LogisticRegression()
model.fit(train_X, train_y)
yHat = model.predict(dev_X)
p, r, f1, s = precision_recall_fscore_support(dev_y, yHat)
print(tabulate(np.array([p,r,f1,s]).T, headers=['precision', 'recall', 'f1', 'support']))

print("ACTIVITY + BIGRAM_LINGUISTIC")
train_X = train[feature_class('freq') + feature_class('month') + feature_class('bigram')]
dev_X = dev[feature_class('freq') + feature_class('month') + feature_class('bigram')]
model = LogisticRegression()
model.fit(train_X, train_y)
yHat = model.predict(dev_X)
p, r, f1, s = precision_recall_fscore_support(dev_y, yHat)
print(tabulate(np.array([p,r,f1,s]).T, headers=['precision', 'recall', 'f1', 'support']))

print("ACTIVITY + WV_LINGUISTIC")
train_X = train[feature_class('freq') + feature_class('month') + feature_class('wv')]
model = LogisticRegression()
model.fit(train_X, train_y)
yHat = model.predict(dev_X)
p, r, f1, s = precision_recall_fscore_support(dev_y, yHat)
print(tabulate(np.array([p,r,f1,s]).T, headers=['precision', 'recall', 'f1', 'support']))
