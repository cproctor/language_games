# Poke around Google's huge pretrained Google News embedding
# with a goal of saving it down to a smaller vocabular size

# TODO: Use Phraser to phrase up our corpus...

from gensim.models.word2vec import Word2Vec, PathLineSentences, LineSentence
from os.path import join
import arrow
from discourse_community import DiscourseCommunity
from tqdm import tqdm
from settings import *

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def create_initial_model():
    """
    Does (very slow) initial work of reading over entire corpus, generating the vocabulary, 
    importing relevant word vectors from GoogleNews embedding, and then saving this iniital 
    model so we don't have to do it again. 
    """
    sentences = PathLineSentences(HN_MONTHLY_CORPUS_DIR)
    model = Word2Vec(size=300, sg=1, hs=1, negative=0) # Hey! Is no negative sampling a problem? 
                                                       # Hopefully this just means a bad non-optimization
                                                       # rather than invalid results. Check the gensim code.
                                                       # I believe that not using negative sampling just means
                                                       # I'm using the (very) inefficient approach of computing
                                                       # the softmax over all non-context words at each iteration.
                                                       # So not invalid, just inefficient, probably contributing to 
                                                       # worse vectors than I would have otherwise.
    print("building vocabulary...")
    model.build_vocab(sentences)
    print("intersecting pretrained word vectors...")
    model.intersect_word2vec_format(GOOGLE_NEWS_EMBEDDING_FILE, lockf=1.0, binary=True)
    print("saving...")
    model.save(INITIAL_MODEL)
    return model

def train_monthly_models(weighted=False, epochs=10):
    """
    Load the initial model (vocabulary with Google-News-trained vectors), and then iterate
    over each month's corpus, training and saving the model. This takes ~3 hours, and requires
    about 50gb to write the models :/
    """
    model = Word2Vec.load(INITIAL_MODEL)
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print("TRAINING {}".format(begin.format("YYYY-MM")))
        corpus = get_month_corpus_filepath(begin.year, begin.month, weighted)
        sentences = LineSentence(corpus)
        model.train(sentences, epochs=epochs, start_alpha=0.1, total_examples=file_len(corpus))
        model.save(get_month_embedding_filepath(begin.year, begin.month, weighted))
        model.wv.save(get_month_word_vectors_filepath(begin.year, begin.month, weighted))

def save_readonly_word_vectors():
    """
    Unless we plan to keep training the model, it is enough to just work with 
    This function is only needed because I didn't save the word vectors on the first pass. 
    Now it's included in train_monthly_models
    """
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in tqdm(months):
        model = Word2Vec.load(get_month_embedding_filepath(begin.year, begin.month))
        model.wv.save(get_month_word_vectors_filepath(begin.year, begin.month))
        del model

#model = create_initial_model()
train_monthly_models(weighted=True, epochs=5)
#save_readonly_word_vectors()
#dc = plot_time_series()



