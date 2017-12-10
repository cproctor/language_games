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

results = []

# ======= ACTIVITY ==============
train_X = train[feature_class('freq') + feature_class('month')]
dev_X = dev[feature_class('freq') + feature_class('month')]
model = LogisticRegression()
model.fit(train_X, train_y)

yHat = model.predict(train_X)
p, r, f1, s = precision_recall_fscore_support(train_y, yHat)
results.append(["Train", "Activity", p[1], r[1], f1[1], s[1]])

yHat = model.predict(dev_X)
p, r, f1, s = precision_recall_fscore_support(dev_y, yHat)
results.append(["Dev", "Activity", p[1], r[1], f1[1], s[1]])

# ======= ACTIVITY + LINGUISTIC BIGRAM ==============
train_X = train[feature_class('freq') + feature_class('month') + feature_class('bigram')]
dev_X = dev[feature_class('freq') + feature_class('month') + feature_class('bigram')]
model = LogisticRegression()
model.fit(train_X, train_y)

yHat = model.predict(train_X)
p, r, f1, s = precision_recall_fscore_support(train_y, yHat)
results.append(["Train", "Activity + bigram CE", p[1], r[1], f1[1], s[1]])

yHat = model.predict(dev_X)
p, r, f1, s = precision_recall_fscore_support(dev_y, yHat)
results.append(["Dev", "Activity + bigram CE", p[1], r[1], f1[1], s[1]])

# ======= ACTIVITY + WORD VECTOR BIGRAM ==============
train_X = train[feature_class('freq') + feature_class('month') + feature_class('wv')]
model = LogisticRegression()
model.fit(train_X, train_y)

yHat = model.predict(train_X)
p, r, f1, s = precision_recall_fscore_support(train_y, yHat)
results.append(["Train", "Activity + Word Vector CE", p[1], r[1], f1[1], s[1]])

yHat = model.predict(dev_X)
p, r, f1, s = precision_recall_fscore_support(dev_y, yHat)
results.append(["Dev", "Activity + Word Vector CE", p[1], r[1], f1[1], s[1]])

print(tabulate(results, headers=['dataset', 'model', 'precision', 'recall', 'f1', 'support'], tablefmt='html'))
