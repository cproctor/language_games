# What to do: 
# 1. Get just the keyedvectors, not the whole models. 
# Use a logistic transform before plotting

from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
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
        self.models = []
        self.labels = labels
        for f in tqdm(word_vector_files, "Loading word vectors"):
            model = Word2Vec.load(f) 
            model.wv.init_sims()
            self.models.append(model)

    def greatest_shift(self, topn=10, restrict_vocab=None):
        """
        Using the first and last models, computes the words whose vector positions have shifted the most
        by taking the cosine similarity of one embedding with the other.
        
        topn: return a subset of the result
        restrict_vocab: instead of considering the entire vocabulary, consider only a slice. Only 
            makes sense if the vocabulary is sorted by frequency. 
        """
        if len(self.models) < 2: raise ValueError("Cannot compute word shifts with fewer than 2 embeddings")
        begin = self.models[0].wv.syn0[:restrict_vocab] if restrict_vocab else self.models[0].wv.syn0
        end = self.models[-1].wv.syn0[:restrict_vocab] if restrict_vocab else self.models[-1].wv.syn0
        shifts = np.sqrt(np.einsum('ij,ij->i', (begin-end), (begin-end)))

        if not topn: return shifts # Just a vector of distances

        best = argsort(shifts, topn=topn, reverse=True) 
        return [(self.models[0].wv.index2word[i], shifts[i]) for i in best]

    def greatest_projected_shift(self, word1, word2, topn=10, restrict_vocab=None, first_model=0, last_model=-1):
        """
        Projects words onto the line spanning `word1` and `word2`, once for the first model
        and once for the last model. Returns the words with the highest absolute value of shift between
        the two models
        """
        proj = {}
        for label, i in (('begin', first_model), ('end', last_model)):
            model = self.models[i]
            words = model.wv.syn0[:restrict_vocab] if restrict_vocab else model.wv.syn0
            proj[label] = self.project(words, self.word_vec(model, word1), self.word_vec(model, word2))

        diffs = proj['end'] - proj['begin']
        absDiffs = np.abs(diffs)
        movers = argsort(absDiffs, topn=topn, reverse=True)
        return [(self.models[0].wv.index2word[i], diffs[i]) for i in movers]

    def time_series_projections(self, word1, word2, words):
        """
        Computes a projection of `words` onto the line from `word1` to `word2` at each time step.
        Anchors may be individual words or lists of words to be averaged. This data can then be plotted. 
        """
        projections = []
        for model in self.models: 
            vecs = np.array([model.wv.word_vec(word) for word in words])
            projections.append(self.project(vecs, self.word_vec(model, word1), self.word_vec(model, word2)))
        return projections

    def plot_time_series_projections(self, word1, word2, words):
        proj = self.time_series_projections(word1, word2, words)
        X = range(len(self.models))
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
        if isinstance(word1, str) and isinstance(word2, str):
            plt.title("Shift in word meanings projected onto {}-{} axis".format(word1, word2))
        else:
            plt.title("Shift in word meanings projected onto axis between clusters")
        plt.show()

    def project(self, words, start, end):
        """
        Projects `words` onto the vector from `start` to `end`, and then returns a scalar
        representing the location of the scalar, where 0 means the point was projected onto
        `start` and 1 means the point was projected onto `end`. 
        """
        line = end - start
        return np.dot(words - start, line)/np.dot(line, line)

    def word_vec(self, model, word):
        """
        Given a model and a word or a list of words, returns the word vector or mean of word vectors
        """
        if isinstance(word, str):
            return model.wv.word_vec(word)
        elif isinstance(word, list):
            return np.mean([self.word_vec(model, w) for w in word], axis=0)
        else:
            raise ValueError("Can only look up words and lists of words")

    def word_label(self, word):
        if isinstance(word, str): 
            return word
        elif isinstance(word, list):
            return "({})".format('-'.join(word))
        else:
            raise ValueError("Expecting a string or list of strings")

if __name__ == '__main__':
    dc = DiscourseCommunity(["initial-wv", "2010-01-wv"])
    print(dc.greatest_shift(topn=10, restrict_vocab=100))
    print(dc.greatest_projected_shift('man', 'woman', 10, 1000))
    words = ['smart', 'funny', 'bold', 'thoughtful']
    dc.plot_time_series_projections('man', 'woman', words)
