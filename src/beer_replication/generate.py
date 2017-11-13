# Partially replicating a baseline analytical strategy

# Danescu-Niculescu-Mizil, C., West, R., Jurafsky, D., Leskovec, J., & 
# Potts, C. (2013, May). No country for old members: User lifecycle and 
# linguistic change in online communities. In Proceedings of the 22nd 
# international conference on World Wide Web (pp. 307-318). ACM.

# author cp

import snap
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from helpers import *
from os.path import join

HN_DATA = "../../data/hn_comments_clean.csv"
HN_DB = "../../data/hn_comments.sqlite3"
HN_MONTHLY_DIR = "../../data/hn_monthly"
HN_MONTHLY_TEMPLATE = "hn_comments_{}_{}.csv"
HN_MONTHLY_COUNTS = "../../data/hn_monthly_count.csv"
HN_MONTHLY_COUNT_CHART = "../../results/hn_monthly_comments.png"

HN_MONTHLY_CORPUS_DIR = "../../data/hn_corpus_monthly"
HN_MONTHLY_CORPUS_TEMPLATE = "hn_corpus_{}_{}.txt"

def get_month_filepath(year, month):
    return join(HN_MONTHLY_DIR, HN_MONTHLY_TEMPLATE.format(year, month))

def get_month_corpus_filepath(year, month):
    return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

if False: # Split comments into monthly files
    counts = split_comments_by_month(HN_DATA, get_month_filepath)
    pd.DataFrame(counts.items(), columns=["month", "comments"]).to_csv(HN_MONTHLY_COUNTS, date_format='%Y-%m', 
            index=False)

if True: # Plot the monthly comments chart
    counts = pd.read_csv(HN_MONTHLY_COUNTS)
    counts.plot.bar(x="month", y="comments")
    plt.savefig(HN_MONTHLY_COUNT_CHART)
    
if True: # Generate monthly 
    pass

# Table 1 (p. 3)

