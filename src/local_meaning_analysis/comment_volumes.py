import os
from settings import *
import pandas as pd
import matplotlib.pyplot as plt
import arrow
from collections import defaultdict
from characteristic_examples import get_examples
from word_projections import default_colors

def plot_comment_volume(terms, cache_file=None, force=False):
    if not (cache_file and os.path.exists(cache_file) and not force):
        date_span = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
        timeseries = defaultdict(list)
        for begin, end in date_span:
            timeseries['year'].append(begin.year)
            timeseries['month'].append(begin.month)
            for term in terms:
                timeseries[term].append(len(get_examples(begin.year, begin.month, [term])))
        data = pd.DataFrame(timeseries).set_index(['year', 'month'])
        data.to_csv(cache_file)

    if (cache_file and os.path.exists(cache_file)):
        data = pd.read_csv(cache_file, index_col=['year', 'month'])

    fig, ax = plt.subplots()
    data.plot(color=default_colors(len(data.columns)), ax=ax, linewidth=1.5)
    xticks = [i for i, (y, m) in enumerate(data.index) if m == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(data.index.get_level_values('year').unique())
    ax.set_xlabel("Year")
    ax.set_ylabel("Comments containing term per month")
    plt.title("Comments per month")
    
