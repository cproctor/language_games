# Compute user lifecycles.
from __future__ import division
from settings import *
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from collections import Counter
import matplotlib.pyplot as plt

SAMPLES = 10000

def get_lifecycle_milestones(lifespan):
    return np.linspace(10, lifespan-10, 11).round().astype(int)

def text_jaccard(old_comments, comment):
    prior = set(sum([c.split() for c in old_comments], []))
    current = set(comment.split())
    return len(prior.intersection(current)) / len(prior.union(current)) 

users = pd.read_csv(USERS, usecols=['username'])
comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=['comment_text', 'username', 'wv_score', 'bigram_score'])
counts = Counter(comments.username)
user_counts = pd.DataFrame({'username': counts.keys(), 'comment_count': counts.values()})
users = user_counts.merge(users, left_on='username', right_on='username') 
users = users[users.comment_count >= 50]
user_comment_groups = comments.groupby('username')
jaccard_scores = []
bigram_dist = []
wv_dist = []
for i, u in users.sample(SAMPLES).iterrows():
    user_comments = user_comment_groups.get_group(u.username)
    milestones = get_lifecycle_milestones(len(user_comments))
    jaccard_scores.append([text_jaccard(
        user_comments.comment_text[ms-10:ms].values, 
        user_comments.comment_text.iloc[ms]
    ) for ms in milestones])
    bigram_dist.append(user_comments.bigram_score.iloc[milestones])
    wv_dist.append(user_comments.wv_score.iloc[milestones])

jaccard_scores = np.array(jaccard_scores)
jaccard_averages = jaccard_scores.mean(axis=0)
jaccard_std = jaccard_scores.std(axis=0) / np.sqrt(SAMPLES)

lifespan = np.linspace(0, 100, 11)
plt.errorbar(lifespan, jaccard_averages, yerr=jaccard_std, color="red")
plt.plot(lifespan, jaccard_averages, color="blue")
plt.xlim([0, 100])
plt.title("Language flexibility")
plt.xlabel("life stage (percentage)")
plt.ylabel("Self-similarity (Jaccard)")
plt.savefig(LIFE_JACCARD_CHART)
plt.clf()

bigram_dist = np.array(bigram_dist)
bigram_avg = bigram_dist.mean(axis=0)
bigram_std = bigram_dist.std(axis=0) / np.sqrt(SAMPLES)

plt.errorbar(lifespan, bigram_avg, yerr=bigram_std, color="red")
plt.plot(lifespan, bigram_avg, color="blue")
plt.xlim([0, 100])
plt.title("Distance from community: Bigram Cross-entropy")
plt.xlabel("life stage (percentage)")
plt.ylabel("Distance from community")
plt.savefig(LIFE_DIST_BIGRAM_CHART)
plt.clf()

wv_dist = np.array(wv_dist)
wv_avg = wv_dist.mean(axis=0)
wv_std = wv_dist.std(axis=0) / np.sqrt(SAMPLES)

plt.errorbar(lifespan, wv_avg, yerr=wv_std, color="red")
plt.plot(lifespan, wv_avg, color="blue")
plt.xlim([0, 100])
plt.title("Distance from community: Word vector log lik.")
plt.xlabel("life stage (percentage)")
plt.ylabel("Distance from community")
plt.savefig(LIFE_DIST_WV_CHART)
plt.clf()



