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

# Prepare the dataset by loading it into a database 
#create_db_if_missing(HN_DATA, HN_DB)

def get_month_filepath(year, month):
    return join(HN_MONTHLY_DIR, HN_MONTHLY_TEMPLATE.format(year, month))

counts = split_comments_by_month(HN_DATA, get_month_filepath)
counts.to_csv(HN_MONTHLY_COUNTS)

# Table 1 (p. 3)

