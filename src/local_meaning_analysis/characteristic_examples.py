# To get a feel for how particular terms are being used differently over time, 
# I want to pull characteristic examples of term usages for a particular month. 
# These will be the most likely comments (on some metric) containing the target
# term(s) for the month.
from settings import *
import pandas as pd
from scipy import stats
from language_models import VectorSpaceModel
import shutil

def get_examples(year, month, terms):
    with open(get_month_corpus_filepath(year, month)) as corpus:
        return [c for c in corpus if any(term in c for term in terms)]

def get_characteristic_examples(year, month, terms, model=None, baseline_model=None, n=20, print_examples=True):
    model = model or VectorSpaceModel.monthly_word2vec(year, month)
    baseline_model = baseline_model or VectorSpaceModel(INITIAL_MODEL)
    comments = get_examples(year, month, terms)
    examples = pd.DataFrame({'comment': comments, 'score': model.score(comments)})
    examples = examples.assign(baseline=baseline_model.score(comments))
    examples = examples.assign(delta=examples.score - examples.baseline)
    examples = examples.sort_values('score', ascending=False)
    examples.to_csv(get_characteristic_examples_filepath(year, month, terms))
    if print_examples:
        show_characteristic_examples(year, month, terms, n)
    return examples

def show_characteristic_examples(year, month, terms, n=20, sort_by='delta', comment_length=None, 
        min_length=30, omit_comments_with_links=True):
    comment_length = comment_length or shutil.get_terminal_size()[0] - 20
    examples = pd.read_csv(get_characteristic_examples_filepath(year, month, terms))
    if omit_comments_with_links: 
        examples = examples[~examples.comment.str.contains("://")]
    if min_length:
        examples = examples[examples.comment.str.split().apply(lambda x: len(x)) >= min_length]
    if n:
        examples = examples[:n]
    print("\nMost characteristic comments from {}-{} containing {} (sorted by {})".format(
            year, month, ', '.join(terms), sort_by))
    for i, ex in examples.sort_values(sort_by, ascending=False).iterrows():
        print("  - [{}] {}".format(ex[sort_by], ex.comment.strip()[:comment_length]))
