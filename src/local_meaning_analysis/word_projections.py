
# I have a separate embedding for each month over a decade, and each is several 
# gigabytes. There's no way I can load them all into memory. So these functions
# perform batch processing on requests. The interface is straightforward: 
# a pandas DataFrame serves as a database for requests and responses. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import settings as s

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

def plot_iat(p, clusters, words, endpoints, colors=None, title=None,
        endpointLabels=None, wordLabels=None, std=False):
    """
    The implicit association test was reported with two groups of words.
    Create plots illustrating how these groups move over time.
    
    Arguments: 
        `projections`: a DataFrame with 'word', 'projection', 'year', 'month', 
            'custer0', 'cluster1' columns
        `clusters`: dict of label -> list of words
        `words`: two labels of word lists to plot
        `endpoints`: two labels specifying word list endpoints
        `title`: Or default will be used
        `colors`: see just below
        `endpointLabels`: alternative names to display
        `wordLabels`: alternative names to display
        `std`: show word standard deviation as shaded area instead of individual
            words
    """
    if colors is None:
        colors = {
            "words": [[(0.2, 0.4, 0.6, 0.2)], [(0.6, 0.4, 0.2, 0.2)]],
            "means": [[(0.2, 0.4, 0.6, 1)], [(0.6, 0.4, 0.2, 1)]],
        }

    window = 24

    fig, ax = plt.subplots()
    matches = p[(p.cluster0 == endpoints[0]) & (p.cluster1 == endpoints[1])]
    word_mean_handles = []
    ts = matches.pivot_table(index=["year", "month"], columns="word", values="projection")

    # Individual words in each group
    for c_word, c_mean, label in zip(colors['words'], colors['means'], words):
        cols = ts[clusters[label]].rolling(window=window, win_type='hamming', center=True).mean()
        col_mean = cols.mean(axis=1)
        if std:
            col_std = cols.std(axis=1)
            ax.fill_between(range(len(cols.index)), col_mean-col_std, col_mean+col_std, color=c_word)
        else:
            cols.plot(color=c_word, ax=ax, legend=False, linewidth=1.5)
        col_mean.plot(color=c_mean, ax=ax, legend=False, linewidth=3)
        handles, labels = ax.get_legend_handles_labels()
        word_mean_handles.append(handles[-1])

    # Legend
    legend = plt.legend(word_mean_handles, wordLabels or words, loc='lower right')
    frame = legend.get_frame()
    frame.set_linewidth(0)
    frame.set_alpha(1)

    # X Axis ticks and label
    xticks = [i for i, (y, m) in enumerate(ts.index) if m == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(p.year.unique() + 1) # This offset is a terrible thing to do.
    ax.set_xlabel("Year")

    # Y Axis ticks, label
    ax.text(min(xticks), 0.01, (endpointLabels or endpoints)[0], ha='left', va="bottom")
    ax.text(min(xticks), 1.01, (endpointLabels or endpoints)[1], ha='left', va="bottom")
    plt.plot([0, len(cols.index)-1], [0,0], color="k")
    plt.plot([0, len(cols.index)-1], [1,1], color="k")
    ax.set_ylabel("Words projected onto {}-{} axis".format(*(endpointLabels or endpoints)))
    ax.set_ylim([-0.1, 1.1])

    plt.title(title or "Change in HN word meanings over time".format(*(wordLabels or words)))
