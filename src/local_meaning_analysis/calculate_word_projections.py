from word_projections import request_word_projections, plot_iat
import weat
import pandas as pd
import matplotlib.pyplot as plt
from settings import *
import arrow
from pathlib import Path

# TODO rename this file.
# Loading monthly models is slow. So we for the association analysis, we'll rely on 
# batch processing.

# Batch-process the data we'll need. 
if False:
    requests = []
    date_span = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for exp, [[cluster0, cluster1], [stim0, stim1]] in weat.experiments.items():
        for word in weat.clusters[stim0] + weat.clusters[stim1]:
            for begin, end in date_span:
                requests.append({
                    "word": word.lower(), 
                    "year": begin.year,
                    "month": begin.month,
                    "cluster0": cluster0,
                    "cluster1": cluster1
                })
    r = pd.DataFrame(requests)
    result = request_word_projections(r, weat.clusters)
    result.to_csv(ASSOCIATION_PROJECTIONS)

# Generate the time series and the charts we need to show changes in implicit association
if True:
    projections = p = pd.read_csv(ASSOCIATION_PROJECTIONS)

    clean = lambda name: ''.join(i for i in name if not i.isdigit()).replace('_', ' ')

    for name, (endpoints, words) in weat.experiments.items():
        plot_iat(p, weat.clusters, words, endpoints, std=True, 
                title="HN word meanings ({})".format(name), 
                endpointLabels=list(map(clean, endpoints)),
                wordLabels=list(map(clean, words)))
        plt.savefig(str(Path(WEAT_RESULTS) / Path(name + '.png')))

