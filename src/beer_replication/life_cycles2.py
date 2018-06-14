
# Compute user lifecycles.
# This version does not consider jaccard scores, mostly because 
# it's slow and time is short.

from __future__ import division
from settings import *
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from webcolors import cp_web_colors

mpl.rcParams['lines.linewidth'] = 3.0

SAMPLES = 10000

def get_lifecycle_milestones(lifespan):
    return np.linspace(10, lifespan-10, 11).round().astype(int)

users = pd.read_csv(USERS, usecols=['username'])
comments = pd.read_csv(HN_SCORED_COMMENTS, usecols=['username', 'wv_score', 'bigram_score'])
counts = Counter(comments.username)
user_counts = pd.DataFrame({'username': list(counts.keys()), 'comment_count': list(counts.values())})
users = user_counts.merge(users, left_on='username', right_on='username') 
users = users[users.comment_count >= 50]
user_comment_groups = comments.groupby('username')
bigram_dist = []
wv_dist = []
for i, u in users.sample(SAMPLES).iterrows():
    user_comments = user_comment_groups.get_group(u.username)
    milestones = get_lifecycle_milestones(len(user_comments))
    bigram_dist.append(user_comments.bigram_score.iloc[milestones])
    wv_dist.append(user_comments.wv_score.iloc[milestones])

lifespan = np.linspace(0, 100, 11)

bigram_dist = np.array(bigram_dist)
bigram_avg = bigram_dist.mean(axis=0)
bigram_std = bigram_dist.std(axis=0) / np.sqrt(SAMPLES)

plt.errorbar(lifespan, bigram_avg, yerr=bigram_std, color=cp_web_colors[8], ecolor=cp_web_colors[7])
plt.xlim([0, 100])
plt.title("Distance from community: Bigram Cross-entropy")
plt.xlabel("life stage (percentage)")
plt.ylabel("Distance from community")
plt.savefig(LIFE_DIST_BIGRAM_CHART)
plt.clf()

wv_dist = np.array(wv_dist)
wv_avg = wv_dist.mean(axis=0)
wv_std = wv_dist.std(axis=0) / np.sqrt(SAMPLES)

plt.errorbar(lifespan, wv_avg, yerr=wv_std, color=cp_web_colors[8], ecolor=cp_web_colors[7])
plt.xlim([0, 100])
plt.title("Distance from community: Word vector log lik.")
plt.xlabel("life stage (percentage)")
plt.ylabel("Distance from community")
plt.savefig(LIFE_DIST_WV_CHART)
plt.clf()



