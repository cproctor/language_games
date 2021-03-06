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
from settings import *

if False: # Split comments into monthly files
    counts = split_comments_by_month(HN_CLEAN_DATA, get_month_filepath, 
            start_month=START_MONTH, end_month=END_MONTH)
    print(counts)
    counts.to_csv(HN_MONTHLY_COUNTS, index=False)

if True: # Plot the monthly comments chart
    counts = pd.read_csv(HN_MONTHLY_COUNTS, parse_dates=['month'])
    ax = counts.sort_values(by='month').plot(x="month", y="comments", legend=None,
        color="#715884", linewidth=2)
    plt.xlabel("month")
    plt.ylabel("comments per month")
    plt.savefig(HN_MONTHLY_COUNT_CHART)
    
if False: # Generate monthly tokens
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print(begin.format("YYYY-MM"))
        tokens = tokenize(get_month_filepath(begin.year, begin.month))
        with open(get_month_corpus_filepath(begin.year, begin.month), 'w') as tokensfile:
            tokensfile.write("\n".join([(" ".join(s)).lower() for s in tokens]).encode('ascii', 'ignore'))

if False: # Build langauge models
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

# NOW DO IT AGAIN, BUT WEIGHTED
if False: # Generate monthly tokens
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print(begin.format("YYYY-MM"))
        tokens = tokenize(get_month_filepath(begin.year, begin.month), weighted=True)
        with open(get_month_corpus_filepath(begin.year, begin.month, weighted=True), 'w') as tokensfile:
            tokensfile.write("\n".join([(" ".join(s)).lower() for s in tokens]).encode('ascii', 'ignore'))
        
    
