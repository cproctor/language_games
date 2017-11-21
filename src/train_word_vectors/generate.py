# Poke around Google's huge pretrained Google News embedding
# with a goal of saving it down to a smaller vocabular size

# TODO: Use Phraser to phrase up our corpus...

from gensim.models.word2vec import Word2Vec, PathLineSentences, LineSentence
from os.path import join
import arrow
from discourse_community import DiscourseCommunity
from tqdm import tqdm

GOOGLE_NEWS_EMBEDDING_FILE = "../../data/GoogleNews-vectors-negative300.bin"
HN_MONTHLY_CORPUS_DIR = "../../data/hn_corpus_monthly"
HN_MONTHLY_CORPUS_TEMPLATE = "hn_corpus_{}_{}.txt"
HN_MONTHLY_MODELS_DIR = "/Volumes/Chris Proctor Backup/language_games/hn_embeddings_monthly/"
HN_MONTHLY_MODEL_TEMPLATE = "hn_embed_{}_{}"

HN_MONTHLY_WVS_DIR = "/Volumes/Chris Proctor Backup/language_games/hn_word_vectors_monthly/"
HN_MONTHLY_WV_TEMPLATE = "hn_wv_{}_{}"

INITIAL_MODEL = join(HN_MONTHLY_MODELS_DIR, 'initial')

START_MONTH = "2007-02"
END_MONTH = "2017-09"

def get_month_corpus_filepath(year, month):
    return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

def get_month_embedding_filepath(year, month):
    return join(HN_MONTHLY_MODELS_DIR, HN_MONTHLY_MODEL_TEMPLATE.format(year, month))

def get_month_word_vectors_filepath(year, month):
    return join(HN_MONTHLY_WVS_DIR, HN_MONTHLY_WV_TEMPLATE.format(year, month))

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
    """
    Load the initial model (vocabulary with Google-News-trained vectors), and then iterate
    over each month's corpus, training and saving the model. This takes ~3 hours, and requires
    about 50gm to write the models :/
    """
    model = Word2Vec.load(INITIAL_MODEL)
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    for begin, end in months:
        print("TRAINING {}".format(begin.format("YYYY-MM")))
        corpus = get_month_corpus_filepath(begin.year, begin.month)
        sentences = LineSentence(corpus)
        model.train(sentences, epochs=10, start_alpha=0.1, total_examples=file_len(corpus))
        model.save(get_month_embedding_filepath(begin.year, begin.month))
        model.wv.save(get_month_word_vectors_filepath(begin.year, begin.month))

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

def plot_time_series():
    years = arrow.Arrow.span_range('year', arrow.get("2008-01"), arrow.get("2017-01"))
    models = ['initial-wv'] + [get_month_word_vectors_filepath(y.year, y.month) for y, _ in years]
    labels = ["Google News"] + [y.format('YYYY-MM') for y, _ in years]
    dc = DiscourseCommunity(models, labels)
    words = ['smart', 'funny', 'bold', 'thoughtful', 'caring']
    dc.plot_time_series_projections('man', 'woman', words)
    return dc
    
#model = create_initial_model()
#train_monthly_models()
#save_readonly_word_vectors()
dc = plot_time_series()



