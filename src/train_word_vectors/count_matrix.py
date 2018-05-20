from collections import defaultdict
import numpy as np
import pandas as pd

def word_reader(filename, verbose=False):
    "Reads lines from a file as lists of words"
    with open(filename) as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print("  {}".format(i))
            yield line.split()
        
# Note: In my use case, this takes a lot of storage: 2gig for a 50k vocabulary.
# I've got a big external hard drive; it's probably worth it to just trade space
# for time as they matrix would have to be densified before use anyway. 
class CooccurrenceCounter:
    """
    Computes a cooccurrence matrix over a corpus of documents. 
    Each document should be on a separate line. 
    """
    def __init__(self, vocab, scaling='distance', window=20):
        self.vocab = vocab
        self.vlookup = {word:i for i, word in enumerate(vocab)}
        self.scaling = scaling
        self.window = window
        self.matrix = np.zeros((len(vocab), len(vocab)))

    def count(self, documents):
        if self.scaling == 'distance':
            for document in documents:
                for i, target in enumerate(document):
                    if target in self.vlookup.keys():
                        for j, other in enumerate(document[i+1:i+self.window]):
                            try:
                                self.matrix[self.vlookup[target], self.vlookup[other]] += 1./(j+1)
                            except KeyError:
                                continue
        elif self.scaling == 'constant':
            for document in documents:
                for i, target in enumerate(document):
                    if target in self.vlookup.keys():
                        for j, other in enumerate(document[i+1:i+self.window]):
                            try:
                                self.matrix[self.vlookup[target], self.vlookup[other]] += 1
                            except KeyError:
                                continue
        else:
            raise NotImplementedError("Scaling method '{}' not supported.".format(self.scaling))
