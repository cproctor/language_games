# Splits the comments out into separate files for each month. 
# Then generates tokens for each month's comments. 

# author cp

import snap
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from helpers import *
from os.path import join
from os import listdir
import arrow
import subprocess

KENLM_LMPLZ = "../../lib/kenlm/build/bin/lmplz"
KENLM_BUILD_BINARY = "../../lib/kenlm/build/bin/build_binary"

HN_CLEAN_DATA = "../../data/hn_comments_utf8_text.csv"
HN_DB = "../../data/hn_comments.sqlite3"
HN_MONTHLY_DIR = "../../data/hn_monthly"
HN_MONTHLY_TEMPLATE = "hn_comments_{}_{}.csv"
HN_MONTHLY_COUNTS = "../../data/hn_monthly_count.csv"
HN_MONTHLY_COUNT_CHART = "../../results/hn_monthly_comments.png"

HN_MONTHLY_CORPUS_DIR = "../../data/hn_corpus_monthly"
HN_MONTHLY_CORPUS_TEMPLATE = "hn_corpus_{}_{}.txt"
HN_MONTHLY_LM_DIR = "../../data/hn_monthly_lm"
HN_MONTHLY_LM_TEMPLATE = "lm_{}_{}.{}"

START_MONTH = "2007-02"
END_MONTH = "2017-09"

def get_month_filepath(year, month):
    return join(HN_MONTHLY_DIR, HN_MONTHLY_TEMPLATE.format(year, month))

def get_month_corpus_filepath(year, month):
    return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

def get_month_lm_filepath(year, month, ext="arpa"):
    return join(HN_MONTHLY_LM_DIR, HN_MONTHLY_LM_TEMPLATE.format(year, month, ext))

if False: # Split comments into monthly files
    counts = split_comments_by_month(HN_CLEAN_DATA, get_month_filepath, 
            start_month=START_MONTH, end_month=END_MONTH)
    print(counts)
    counts.to_csv(HN_MONTHLY_COUNTS, index=False)

if False: # Plot the monthly comments chart
    counts = pd.read_csv(HN_MONTHLY_COUNTS, parse_dates=['month'])
    ax = counts.sort_values(by='month').plot(x="month", y="comments")
    plt.savefig(HN_MONTHLY_COUNT_CHART)
    
if False: # Generate monthly tokens
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print(begin.format("YYYY-MM"))
        tokens = tokenize(get_month_filepath(begin.year, begin.month))
        with open(get_month_corpus_filepath(begin.year, begin.month), 'w') as tokensfile:
            tokensfile.write("\n".join([(" ".join(s)).lower() for s in tokens]).encode('ascii', 'ignore'))

if True: # Build langauge models
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print(begin.format("YYYY-MM"))
        estimateCmd = "{} -o {} -S {} -T {} <{} >{}".format(
            KENLM_LMPLZ, 
            '2', # order 2 (bigrams)
            '80%', # use 80% system memory
            '~/temp', # dir for temp files
            get_month_corpus_filepath(begin.year, begin.month),
            get_month_lm_filepath(begin.year, begin.month)
        )
        print("CMD: " + estimateCmd)
        subprocess.check_call(estimateCmd, shell=True)

        buildBinaryCmd = " ".join([KENLM_BUILD_BINARY, 
            get_month_lm_filepath(begin.year, begin.month),
            get_month_lm_filepath(begin.year, begin.month, 'binary')
        ])
        print("CMD: " + buildBinaryCmd)
        subprocess.check_call(buildBinaryCmd, shell=True)
        
    
