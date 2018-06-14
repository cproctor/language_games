# Creates monthly embeddings using GloVe retrofitting, via Mittens implementation
# First, computes word cooccurrence matrices for each month. Then
# uses mittens to update the initial matrix. 

# Factors to consider in tuning cooccurrence matrix construction:
    # Scaling method
    # Minimum count threshold

import csv
import numpy as np
import arrow
from settings import *
from count_matrix import CooccurrenceCounter, word_reader
from mittens import Mittens

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {}
        vocab = []
        for line in reader:
            embed[line[0]] = np.array(list(map(float, line[1:])))
            vocab.append(line[0])
    return vocab, embed

def create_cooccurrence_matrices():
    print("loading initial glove embedding")
    vocab, embedding = glove2dict(TRUNCATED_GLOVE_EMBEDDING)
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print("Creating cooccurrence matrix for {}".format(begin.format("YYYY-MM")))
        counter = CooccurrenceCounter(vocab)
        corpus = word_reader(get_month_corpus_filepath(begin.year, begin.month))
        counter.count(corpus)
        np.save(get_month_cooccurrence_matrix_filepath(begin.year, begin.month), counter.matrix)

# Changes: 
# I want to skip skippable months. 
# Make this startable at any point.
# Fix the bug.
def create_monthly_glove_models(begin_month=None):
    model = Mittens(n=300, max_iter=1000)
    vocab, embedding = glove2dict(TRUNCATED_GLOVE_EMBEDDING)
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print("Training mittens model for {}".format(begin.format("YYYY-MM")))
        print("  loading cooccurrence matrix")
        coo_matrix = np.load(get_month_cooccurrence_matrix_filepath(begin.year, begin.month))
        print("  training")
        embedding = model.fit(coo_matrix, vocab=vocab, initial_embedding_dict=embedding)
        print("  saving")
        np.save(get_month_glove_embedding_filepath(begin.year, begin.month), embedding)

if __name__ == '__main__':
    #create_cooccurrence_matrices()
    create_monthly_glove_models()
