# A common interface for interacting with different language models

from settings import *
import pandas as pd
import numpy as np
import kenlm
from gensim.models.word2vec import Word2Vec

class LanguageModel:
    """
    An abstract class to represent different kinds of language models. Each language model 
    needs to be able to report on the likelihood of particular sentences.
    """
    def score(self, sentences):
        "Returns the likelihood of seeing these sentences in the model"
        raise NotImplemented("LanguageModel is an abstract class")

class BigramModel(LanguageModel):
    
    @classmethod
    def monthly(cls, year, month, **kwargs):
        return BigramModel(get_month_lm_filepath(year, month, 'binary'), **kwargs)       

    def __init__(self, filepath, clip=30, bos=True, eos=False, ce=True):
        """
        filepath: path to kenlm-generated model file (extension: .binary)
        clip: max length of comments to use (starting from beginning)
        box: use beginning-of-sentence token 
        eos: use end-of-sentence token
        ce: use cross-entropy (instead of perplexity)
        """
        self.model = kenlm.Model(filepath)
        self.clip = clip
        self.bos = bos
        self.eos = eos
        self.ce = ce

    def score(self, sentences):
        "Uses perplexity or (if set) cross-entropy to score sentences"
        if self.clip:
            sentences = [s[:self.clip] for s in sentences]
        scores = np.array([self.model.score(s, bos=self.bos, eos=self.eos) for s in sentences])
        if ce:
            scores = self.cross_entropy(scores)
        return scores

    def cross_entropy(self, perplexity):
        if perplexity <= 0: return -100
        return np.log(-perplexity)/np.log(2)

class VectorSpaceModel(LanguageModel):
    
    @classmethod
    def monthly_word2vec(cls, year, month, **kwargs):
        return VectorSpaceModel(get_month_embedding_filepath(year, month), **kwargs)

    def __init__(self, filepath, clip=30):
        """
        filepath: path to trained gensim embedding file (not just the word vectors)
        """
        self.model = Word2Vec.load(filepath)
        self.clip = clip

    def score(self, sentences):
        "Uses log-likelihood of each word with respect to its context window to score sentences"
        if self.clip:
            sentences = [s[:self.clip] for s in sentences]
        return self.model.score(sentences)


