
import numpy as np
import pandas as pd
from settings import *
from helpers import feature_class, evaluate_model
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression

comments = pd.read_csv(HN_SCORED_COMMENTS)
train = pd.read_csv(TRAIN_EXAMPLES)
dev = pd.read_csv(DEV_EXAMPLES)
test = pd.read_csv(TEST_EXAMPLES)

trainComments = comments[comments.username.isin(train.username)]
devComments = comments[comments.username.isin(dev.username)]

results = []

results += evaluate_model("pop1~bigram", LogisticRegression(), 
        trainComments.bigram_score, devComments.bigram_score, 
        trainComments.pop1,         devComments.pop1, 
        "train",                    "dev")
results += evaluate_model("pop1~wv", LogisticRegression(), 
        trainComments.wv_score,     devComments.wv_score, 
        trainComments.pop1,         devComments.pop1, 
        "train",                    "dev")
results += evaluate_model("pop3~bigram", LogisticRegression(), 
        trainComments.bigram_score, devComments.bigram_score, 
        trainComments.pop3,         devComments.pop3, 
        "train",                    "dev")
results += evaluate_model("pop3~wv", LogisticRegression(), 
        trainComments.wv_score,     devComments.wv_score, 
        trainComments.pop3,         devComments.pop3, 
        "train",                    "dev")
results += evaluate_model("pop5~bigram", LogisticRegression(), 
        trainComments.bigram_score, devComments.bigram_score, 
        trainComments.pop5,         devComments.pop5, 
        "train",                    "dev")
results += evaluate_model("pop5~wv", LogisticRegression(), 
        trainComments.wv_score,     devComments.wv_score, 
        trainComments.pop5,         devComments.pop5, 
        "train",                    "dev")
results += evaluate_model("pop10~bigram", LogisticRegression(), 
        trainComments.bigram_score, devComments.bigram_score, 
        trainComments.pop10,        devComments.pop10, 
        "train",                    "dev")
results += evaluate_model("pop10~wv", LogisticRegression(), 
        trainComments.wv_score,     devComments.wv_score, 
        trainComments.pop10,        devComments.pop10, 
        "train",                    "dev")
results += evaluate_model("pop20~bigram", LogisticRegression(), 
        trainComments.bigram_score, devComments.bigram_score, 
        trainComments.pop20,        devComments.pop20, 
        "train",                    "dev")
results += evaluate_model("pop20~wv", LogisticRegression(), 
        trainComments.wv_score, devComments.wv_score, 
        trainComments.pop20,    devComments.pop20, 
        "train",                "dev")

print(tabulate(results, headers=['train', 'test', 'model', 'precision', 'recall', 'f1', 'support']))
print("\n")
print(tabulate(results, headers=['train', 'test', 'model', 'precision', 'recall', 'f1', 'support'],
        tablefmt='latex'))

