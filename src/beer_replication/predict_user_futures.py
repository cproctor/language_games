# Once labeled examples have been generated for train, dev, and test, 
# we can create a logistic model predicting whether users are likely
# to remain or depart.
# June 12: DONE

from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
import pandas as pd
from settings import *
from tabulate import tabulate
from helpers import add_min, add_max, feature_class, evaluate_model
import numpy as np

# Prepare data

TRAIN_MODE = 'test'
USEMAX = True
USEMIN = True

train = pd.read_csv(TRAIN_EXAMPLES)
train_y = train.label.values

train = train.replace(-np.inf, 10.497) # Pesky -inf

dev = pd.read_csv(DEV_EXAMPLES)
dev_y = dev.label.values
test = pd.read_csv(TEST_EXAMPLES)
test_y = test.label.values

if USEMIN:
    train = add_min(train, 'bigram')
    dev = add_min(dev, 'bigram')
    test = add_min(test, 'bigram')
    train = add_min(train, 'wv')
    dev = add_min(dev, 'wv')
    test = add_min(test, 'wv')

if TRAIN_MODE == 'dev':
    _train = train
    _test = dev
    _train_y = train_y
    _test_y = dev_y
    trainLabel = "train"
    testLabel = "dev"
elif TRAIN_MODE == 'test':
    _train = pd.concat([train, dev])
    _test = test
    _train_y = np.concatenate([train_y, dev_y])
    _test_y = test_y
    trainLabel = "train+dev"
    testLabel = "test"
else:
    raise ValueError("TRAIN_MODE invalid")

# Specify model and features

model = LogisticRegression()
activityFeatures = feature_class('freq') + feature_class('month')
popularityFeatures = feature_class('pop5')
bigramFeatures = feature_class('bigram', use_max=USEMAX, use_min=USEMIN)
word2vecFeatures = feature_class('wv', use_max=USEMAX, use_min=USEMIN)

# Test cases

results = []

results += evaluate_model("Activity",
    LogisticRegression(),
    _train[activityFeatures],
    _test[activityFeatures],
    _train_y, _test_y
)

results += evaluate_model("Activity + bigram",
    LogisticRegression(),
    _train[activityFeatures + bigramFeatures],
    _test[activityFeatures + bigramFeatures],
    _train_y, _test_y
)

results += evaluate_model("Activity + word2vec",
    LogisticRegression(),
    _train[activityFeatures + word2vecFeatures],
    _test[activityFeatures + word2vecFeatures],
    _train_y, _test_y
)

results += evaluate_model("Activity + pop1",
    LogisticRegression(),
    _train[activityFeatures + feature_class('pop1')],
    _test[activityFeatures + feature_class('pop1')],
    _train_y, _test_y
)
results += evaluate_model("Activity + pop3",
    LogisticRegression(),
    _train[activityFeatures + feature_class('pop3')],
    _test[activityFeatures + feature_class('pop3')],
    _train_y, _test_y
)
results += evaluate_model("Activity + pop5",
    LogisticRegression(),
    _train[activityFeatures + feature_class('pop5')],
    _test[activityFeatures + feature_class('pop5')],
    _train_y, _test_y
)
results += evaluate_model("Activity + pop10",
    LogisticRegression(),
    _train[activityFeatures + feature_class('pop10')],
    _test[activityFeatures + feature_class('pop10')],
    _train_y, _test_y
)
results += evaluate_model("Activity + pop20",
    LogisticRegression(),
    _train[activityFeatures + feature_class('pop20')],
    _test[activityFeatures + feature_class('pop20')],
    _train_y, _test_y
)

print(tabulate(results, headers=['train', 'test', 'model', 'precision', 'recall', 'f1', 'support']))
print("\nLATEX\n")
print(tabulate(results, headers=['train', 'test', 'model', 'precision', 'recall', 'f1', 'support'],
        tablefmt='latex'))

