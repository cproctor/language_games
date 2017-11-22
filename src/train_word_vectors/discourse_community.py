# What to do: 
# 1. Get just the keyedvectors, not the whole models. 
# Use a logistic transform before plotting

from gensim.models.keyedvectors import KeyedVectors
from gensim.matutils import argsort
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class DiscourseCommunity:
    """
    Models a discourse community with word embeddings of language snapshots.
    """

    def __init__(self, word_vector_files, labels=None):
        """
        Pass in a list of strings, each a filepath to a saved KeyedVectors. It is assumed
        that each KeyedVectors instance has the same vocabulary. 
        """
        self.wvs = []
        self.labels = labels
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

    def greatest_projected_shift(self, word1, word2, topn=10, restrict_vocab=None):
        """
        Projects words onto the line spanning `word1` and `word2`, once for the first model
        and once for the last model. Returns the words with the highest absolute value of shift between
        the two models
        """
        proj = {}
        for label, i in (('begin', 0), ('end', -1)):
            wv = self.wvs[i]
            words = wv.syn0[:restrict_vocab] if restrict_vocab else wv.syn0
            proj[label] = self.project(words, wv.word_vec(word1), wv.word_vec(word2))
            # TODO extend this and abstract it out to get all projects. Maybe project can just handle it.

        diffs = proj['end'] - proj['begin']
        absDiffs = np.abs(diffs)
        movers = argsort(absDiffs, topn=topn, reverse=True)
        return [(self.wvs[0].index2word[i], diffs[i]) for i in movers]

    def time_series_projections(self, word1, word2, words):
        """
        Computes a projection of `words` onto the line from `word1` to `word2` at each time step.
        This data can then be plotted. 
        """
        projections = []
        for wv in self.wvs: 
            vecs = np.array([wv.word_vec(word) for word in words])
            projections.append(self.project(vecs, wv.word_vec(word1), wv.word_vec(word2)))
        return projections

    def plot_time_series_projections(self, word1, word2, words):
        proj = self.time_series_projections(word1, word2, words)
        X = range(len(self.wvs))
        for p in zip(*proj):
            plt.plot(X, p)
        plt.plot([0, len(self.wvs)-1], [0,0], color="k")
        plt.plot([0, len(self.wvs)-1], [1,1], color="k")
        textMargin = 0.2
        plt.text(len(self.wvs)-1, 0, word1, ha='right', va="bottom")
        plt.text(len(self.wvs)-1, 1, word2, ha='right', va="bottom")
        plt.xticks(X, self.labels or X, rotation='vertical')
        plt.legend(words)
        plt.xlabel("Language model")
        plt.ylabel("Relative closeness to anchor words")
        plt.title("Shift in word meanings projected onto {}-{} axis".format(word1, word2))
        plt.show()

    def project(self, words, start, end):
        """
        Projects `words` onto the vector from `start` to `end`, and then returns a scalar
        representing the location of the scalar, where 0 means the point was projected onto
        `start` and 1 means the point was projected onto `end`. 
        """
        line = end - start
        return np.dot(words - start, line)/np.dot(line, line)

if __name__ == '__main__':
    dc = DiscourseCommunity(["initial-wv", "2010-01-wv"])
    print(dc.greatest_shift(topn=10, restrict_vocab=100))
    print(dc.greatest_projected_shift('man', 'woman', 10, 1000))
    words = ['smart', 'funny', 'bold', 'thoughtful']
    dc.plot_time_series_projections('man', 'woman', words)
