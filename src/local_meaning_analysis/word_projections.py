
# I have a separate embedding for each month over a decade, and each is several 
# gigabytes. There's no way I can load them all into memory. So these functions
# perform batch processing on requests. The interface is straightforward: 
# a pandas DataFrame serves as a database for requests and responses. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import settings as s
from itertools import permutations

def request_word_projections(words, axis_endpoints, verbose=False,
        get_embedding=s.get_month_embedding_filepath):
    """
    Looks up clusters for words along axes in various models.
    Arguments: 
        `words`: a df with columns word, month, year, axis0, axis1
                 specifying lookup requests.
        `axis_endpoints`: a dict of words -> lists of words. All values of 
                axis0 and axis1 must be present in keys.
        `get_embedding`: (year, month) -> filepath for embedding file
    """
    results = []
    requests_by_model = words.groupby(("year", "month"))
    for model_name, model_requests in requests_by_model:
        year, month = model_name
        print("Processing {} requests for {} {}".format(
                model_requests.shape[0], year, month))
        model = Word2Vec.load(get_embedding(year, month)) 
        for endpoints, requests in model_requests.groupby(("cluster0", "cluster1")):
            # Each axis endpoint is the mean of word vectors for cluster words
            def get_endpoint(label):
                cluster = axis_endpoints[label]
                return np.mean([model.wv.word_vec(w) for w in cluster], axis=0) 

            start = get_endpoint(endpoints[0])
            end = get_endpoint(endpoints[1])
            axis = end - start
            axis_length = np.dot(axis, axis)
            for word in requests['word']:
                try: 
                    wv = model.wv.word_vec(word)
                except KeyError:
                    if verbose: print(" - '{}' is not in vocabulary".format(word))
                    continue
                projection = np.dot(wv - start, axis)/axis_length
                results.append({
                    "word": word, 
                    "year": year,
                    "month": month,
                    "cluster0": endpoints[0],
                    "cluster1": endpoints[1],
                    "projection": projection
                })
    return pd.DataFrame(results)

def default_colors(length, opacity=1):
    colors = list(permutations((0.2, 0.4, 0.6)))
    return [(r,g,b,opacity) for r,b,g in colors][:length]

def plot_words_on_relational_axis(p, words, endpoints, mean_colors=None, std_colors=None,
        title="Change in word meanings over time", endpoint_labels=None, 
        word_labels=None, std=False, window=24, window_type='hamming'):
    """
    The implicit association test was reported with two groups of words.
    Create plots illustrating how these groups move over time.
    
    Arguments: 
        `projections`: a DataFrame with 'word', 'projection', 'year', 'month', 
            'custer0', 'cluster1' columns
        `words`: a list of strings or lists of words to project onto the axis
        `endpoints`: a list of two strings naming the axis endpoints (should match cluster0 and cluster1
            for relevant data in `projections`)
        `title`: Or default will be used
        `mean_colors`: a list of the same length as `words` containing rgba tuples
        `std_colors`: a list of the same length as `words` containing rgba tuples
        `endpoint_labels`: a list of two names to display for endpoints
        `word_labels`: a list of same length as `words` containing names for words or word groups
        `std`: show word standard deviation as shaded area instead of individual
            words
        `window`: window size. Default 24.
        `window_type`: type of window to use. Default 'hamming'
    """
    if mean_colors is None: mean_colors = default_colors(len(words))
    if std_colors is None: std_colors = default_colors(len(words), 0.2)
    if not (mean_colors and std_colors):
        raise NotImplementedError("Colors must be provided. No default colors available")
    if word_labels and len(words) != len(word_labels):
        raise ValueError("words and word_labels must have the same length")

    fig, ax = plt.subplots()
    matches = p[(p.cluster0 == endpoints[0]) & (p.cluster1 == endpoints[1])]
    invert = matches.empty
    if matches.empty:
        matches = p[(p.cluster0 == endpoints[1]) & (p.cluster1 == endpoints[0])]
    if matches.empty:
        raise ValueError("Projections does not contain data for endpoints")
    word_mean_handles = []
    time_series = matches.pivot_table(index=["year", "month"], columns="word", values="projection")
    if invert:
        time_series = 1 - time_series

    # Individual words in each group
    for word_group, mean_color, std_color in zip(words, mean_colors, std_colors):
        if isinstance(word_group, str): word_group = [word_group]
        cols = time_series[word_group].rolling(window=window, win_type=window_type, center=True).mean()
        col_mean = cols.mean(axis=1)
        if std:
            col_std = cols.std(axis=1)
            ax.fill_between(range(len(cols.index)), col_mean-col_std, col_mean+col_std, color=std_color)
        else:
            cols.plot(color=mean_color, ax=ax, legend=False, linewidth=1.5)
        col_mean.plot(color=mean_color, ax=ax, legend=False, linewidth=1.5)
        handles, labels = ax.get_legend_handles_labels()
        word_mean_handles.append(handles[-1])

    # Legend
    legend = plt.legend(word_mean_handles, word_labels or words, loc='lower right')
    frame = legend.get_frame()
    frame.set_linewidth(0)
    frame.set_alpha(1)

    # X Axis ticks and label
    xticks = [i for i, (y, m) in enumerate(time_series.index) if m == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(p.year.unique() + 1) # This offset is a terrible thing to do.
    ax.set_xlabel("Year")

    # Y Axis ticks, label
    ax.text(min(xticks), 0.01, (endpoint_labels or endpoints)[0], ha='left', va="bottom")
    ax.text(min(xticks), 1.01, (endpoint_labels or endpoints)[1], ha='left', va="bottom")
    plt.plot([0, len(cols.index)-1], [0,0], color="k")
    plt.plot([0, len(cols.index)-1], [1,1], color="k")
    ax.set_ylabel("Words projected onto {}-{} axis".format(*(endpoint_labels or endpoints)))
    ax.set_ylim([-0.1, 1.1])

    plt.title(title)
