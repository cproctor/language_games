# What to do: 
# 1. Get just the keyedvectors, not the whole models. 

from gensim.models.keyedvectors import KeyedVectors
from gensim.matutils import argsort
from tqdm import tqdm
import numpy as np

class DiscourseCommunity:
    """
    Models a discourse community with word embeddings of language snapshots.
    """

    def __init__(self, word_vector_files):
        """
        Pass in a list of strings, each a filepath to a saved KeyedVectors. It is assumed
        that each KeyedVectors instance has the same vocabulary. 
        """
        self.wvs = []
        for f in tqdm(word_vector_files, "Loading word vectors"):
            wv = KeyedVectors.load(f) 
            wv.init_sims()
            self.wvs.append(wv)

    def greatest_shift(self, topn=10, restrict_vocab=None):
        """
        Using the first and last models, computes the words whose vector positions have shifted the most
        by taking the cosine similarity of one embedding with the other.
        
        topn: return a subset of the result
        restrict_vocab: instead of considering the entire vocabulary, consider only a slice. Only 
            makes sense if the vocabulary is sorted by frequency. 
        """
        if len(self.wvs) < 2: raise ValueError("Cannot compute word shifts with fewer than 2 embeddings")
        begin = self.wvs[0].syn0norm[:restrict_vocab] if restrict_vocab else self.wvs[0].syn0norm
        end = self.wvs[-1].syn0norm[:restrict_vocab] if restrict_vocab else self.wvs[-1].syn0norm
        shifts = (begin *  end).sum(axis=1) # Dot product only for each word against itself

        if not topn: return shifts # Just a vector of cosine similiarities

        best = argsort(shifts, topn=topn, reverse=False) # Low cosine similarity means far apart
        return [(self.wvs[0].index2word[i], shifts[i]) for i in best]

if __name__ == '__main__':
    dc = DiscourseCommunity(["initial-wv", "2010-01-wv"])
    print(dc.greatest_shift(topn=10, restrict_vocab=100))
