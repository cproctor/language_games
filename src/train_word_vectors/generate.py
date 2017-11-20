# Poke around Google's huge pretrained Google News embedding
# with a goal of saving it down to a smaller vocabular size

# TODO: Use Phraser to phrase up our corpus...


import gensim
from os.path import join

CORPUS_DIR = "../../data/hn_corpus_monthly"
MODEL_FILE = "../../data/GoogleNews-vectors-negative300.bin"
MONTHLY_MODELS_DIR = "../../data/hn_embeddings_monthly"

def create_initial_model():
    sentences = gensim.models.word2vec.PathLineSentences(CORPUS_DIR)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences, size=300, sg=1, hs=1, negative=0)
    model.intersect_word2vec_format(MODEL_FILE, lockf=1.0, binary=True)
    model.save(join(MONTHLY_MODELS_DIR, 'initial'))
    return model

model = create_initial_model()
