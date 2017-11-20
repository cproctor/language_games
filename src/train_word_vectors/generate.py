# Poke around Google's huge pretrained Google News embedding
# with a goal of saving it down to a smaller vocabular size

# TODO: Use Phraser to phrase up our corpus...

from gensim.models.word2vec import Word2Vec, PathLineSentences, LineSentence
from os.path import join
import arrow

GOOGLE_NEWS_EMBEDDING_FILE = "../../data/GoogleNews-vectors-negative300.bin"
HN_MONTHLY_CORPUS_DIR = "../../data/hn_corpus_monthly"
HN_MONTHLY_CORPUS_TEMPLATE = "hn_corpus_{}_{}.txt"
HN_MONTHLY_MODELS_DIR = "../../data/hn_embeddings_monthly"
HN_MONTHLY_MODEL_TEMPLATE = "hn_embed_{}_{}"
INITIAL_MODEL = join(HN_MONTHLY_MODELS_DIR, 'initial')

START_MONTH = "2007-02"
END_MONTH = "2017-09"

def get_month_corpus_filepath(year, month):
    return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

def get_month_embedding_filepath(year, month):
    return join(HN_MONTHLY_MODELS_DIR, HN_MONTHLY_MODEL_TEMPLATE.format(year, month))

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
    model = Word2Vec(size=300, sg=1, hs=1, negative=0)
    print("building vocabulary...")
    model.build_vocab(sentences)
    print("intersecting pretrained word vectors...")
    model.intersect_word2vec_format(GOOGLE_NEWS_EMBEDDING_FILE, lockf=1.0, binary=True)
    print("saving...")
    model.save(INITIAL_MODEL)
    return model

def train_monthly_models():
    model = Word2Vec.load(INITIAL_MODEL)
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print("TRAINING {}".format(begin.format("YYYY-MM")))
        corpus = get_month_corpus_filepath(begin.year, begin.month)
        sentences = LineSentence(corpus)
        model.train(sentences, epochs=10, start_alpha=0.1, total_examples=file_len(corpus))
        model.save(get_month_embedding_filepath(begin.year, begin.month))

#model = create_initial_model()
train_monthly_models()
