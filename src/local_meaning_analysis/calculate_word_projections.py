from word_projections import request_word_projections, plot_words_on_relational_axis, default_colors
from characteristic_examples import get_characteristic_examples, show_characteristic_examples
from comment_volumes import plot_comment_volume
from language_models import VectorSpaceModel
import weat
import pandas as pd
import matplotlib.pyplot as plt
from settings import *
import arrow
from pathlib import Path
from os.path import join

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
if False:
    projections = p = pd.read_csv(ASSOCIATION_PROJECTIONS)

    clean = lambda name: ''.join(i for i in name if not i.isdigit()).replace('_', ' ')

    for name, (endpoints, words) in weat.experiments.items():
        plot_words_on_relational_axis(p, [weat.clusters[words[0]], weat.clusters[words[1]]],
                endpoints, title="HN word meanings ({})".format(name), 
                endpoint_labels=list(map(clean, endpoints)),
                word_labels=list(map(clean, words)), std=True)
        plt.savefig(str(Path(WEAT_RESULTS) / Path(name + '.png')))

def project_tech_companies_pleasant_unpleasant(generate_projections=True):
    tech_companies = [
        "tesla", 
        "bitcoin"
    ]
    if generate_projections:
        requests = []
        date_span = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
        for tc in tech_companies:
            for begin, end in date_span:
                requests.append({
                    "word": tc,
                    "year": begin.year,
                    "month": begin.month,
                    "cluster0": 'pleasant1',
                    "cluster1": 'unpleasant1'
                })
        r = pd.DataFrame(requests)
        result = request_word_projections(r, weat.clusters)
        result.to_csv(TECH_COMPANY_WEAT_PLEASANT_UNPLEASANT_PROJECTIONS)

    projections = p = pd.read_csv(TECH_COMPANY_WEAT_PLEASANT_UNPLEASANT_PROJECTIONS)
    plot_words_on_relational_axis(projections, ['bitcoin', 'tesla'], 
            ['unpleasant1', 'pleasant1'],
            title="Hacker News discussion of tech companies over time",
            endpoint_labels=['unpleasant', 'pleasant'], window=4)
    plt.savefig(TECH_COMPANY_WEAT_PLEASANT_UNPLEASANT_PLOT)

def tech_company_comment_volume(force_generate=False):
    companies = ['bitcoin', 'tesla']
    plot_comment_volume(companies, TECH_COMPANY_COMMENT_VOLUMES, force=force_generate)
    plt.savefig(TECH_COMPANY_COMMENT_VOLUMES_PLOT)

def tech_company_examples():
    baseline = VectorSpaceModel(INITIAL_MODEL) 
    get_characteristic_examples(2012, 9,  ['tesla'], baseline_model=baseline) # high Tesla
    get_characteristic_examples(2013, 11, ['tesla'], baseline_model=baseline) # low Tesla
    get_characteristic_examples(2011, 11, ['bitcoin'], baseline_model=baseline) # low bitcoin
    get_characteristic_examples(2011, 3,  ['bitcoin'], baseline_model=baseline) # high Facebook

def show_examples():
    print("\n HIGH TESLA")
    show_characteristic_examples(2012, 9, ['tesla'], sort_by='score')
    print("\n LOW TESLA")
    show_characteristic_examples(2013, 11, ['tesla'], sort_by='score')
    print("\n HIGH BITCOIN")
    show_characteristic_examples(2011, 11, ['bitcoin'], sort_by='score')
    print("\n LOW BITCOIN")
    show_characteristic_examples(2011, 3, ['bitcoin'], sort_by='score')

#project_tech_companies_pleasant_unpleasant(generate_projections=True)
#tech_company_comment_volume(force_generate=True)
#tech_company_examples()
show_examples()        






