# Shared settings for all parts of the codebase. 
# Mostly, these are file and directory locations.
from os.path import join

LOCAL_DATA = "/Users/chris/Documents/4-PhD/Research/ActiveProjects/LanguageIdeology/language_games/data"
RESULTS = "/Users/chris/Documents/4-PhD/Research/ActiveProjects/LanguageIdeology/language_games/results"
REMOTE_DATA = "/Volumes/Chris Proctor Backup/language_games/data"

HN_DATA = join(LOCAL_DATA, "hn_comments_clean.csv")
HN_CLEAN_DATA = join(LOCAL_DATA, "hn_comments_utf8_text.csv")

KENLM_LMPLZ = "../lib/kenlm/build/bin/lmplz"
KENLM_BUILD_BINARY = "../lib/kenlm/build/bin/build_binary"

HN_MONTHLY_DIR = join(REMOTE_DATA, "hn_monthly")
HN_MONTHLY_WEIGHTED_DIR = join(REMOTE_DATA, "hn_monthly_weighted")
HN_MONTHLY_TEMPLATE = "hn_comments_{}_{}.csv"
HN_MONTHLY_COUNTS = join(LOCAL_DATA, "hn_monthly_count.csv")
HN_MONTHLY_COUNT_CHART = "../results/hn_monthly_comments.png"

HN_MONTHLY_CORPUS_DIR = join(REMOTE_DATA, "hn_corpus_monthly")
HN_MONTHLY_WEIGHTED_CORPUS_DIR = join(REMOTE_DATA, "hn_corpus_monthly_weighted")
HN_MONTHLY_CORPUS_TEMPLATE = "hn_corpus_{}_{}.txt"

HN_MONTHLY_LM_DIR = join(REMOTE_DATA, "hn_monthly_lm")
HN_MONTHLY_LM_TEMPLATE = "lm_{}_{}.{}"

START_MONTH = "2007-02"
END_MONTH = "2017-09"

USERS = join(LOCAL_DATA, "hn_users.csv")
USER_COUNTS = join(LOCAL_DATA, "hn_user_counts.csv")
USER_INTERVALS = join(LOCAL_DATA, "hn_user_intervals.csv")
USER_BASE = '../data/hn_user_base.csv'
USER_BASE_CHART = '../results/hn_user_base.png'

UPVOTES_HIST_DATA = join(LOCAL_DATA, "hn_upvotes_hist.csv")
UPVOTES_HIST_CHART = "../results/hn_upvotes_hist.png"

USER_GRAPH = join(LOCAL_DATA, "hn_user_graph.txt")
USER_DEGREE_DIST = join(LOCAL_DATA, "hn_user_degree_dist.npy")
USER_DEGREE_CF_HIST = join(LOCAL_DATA, "hn_user_degree_cf_hist.npy")
USER_DEGREE_DIST_CHART = "../results/hn_user_degree_distribution.png"
USER_DEGREE_DIST_CHART_WITH_EST = "../results/hn_user_degree_distribution_with_est.png"
USER_DEGREE_CF_HIST_CHART = "../results/hn_user_clustering_coefficients.png"

GOOGLE_NEWS_EMBEDDING_FILE = join(REMOTE_DATA, "GoogleNews-vectors-negative300.bin")
HN_MONTHLY_MODELS_DIR = join(REMOTE_DATA, "hn_embeddings_monthly/")
HN_MONTHLY_WEIGHTED_MODELS_DIR = join(REMOTE_DATA, "hn_embeddings_monthly_weighted/")
HN_MONTHLY_MODEL_TEMPLATE = "hn_embed_{}_{}"

HN_MONTHLY_WVS_DIR = join(REMOTE_DATA, "hn_word_vectors_monthly/")
HN_MONTHLY_WEIGHTED_WVS_DIR = join(REMOTE_DATA, "hn_word_vectors_monthly_weighted/")
HN_MONTHLY_WV_TEMPLATE = "hn_wv_{}_{}"

INITIAL_MODEL = join(HN_MONTHLY_MODELS_DIR, 'initial')

HN_SCORED_COMMENTS = join(REMOTE_DATA, 'scored_comments.csv')
HN_SCORED_COMMENTS_INITIAL_MODEL = join(REMOTE_DATA, 'scored_comments_initial_model.csv')
HN_SCORED_COMMENTS_FULL = join(REMOTE_DATA, 'scored_comments_bloated.csv')
HN_SCORED_COMMENT_BOW_WV = join(REMOTE_DATA, 'scored_comment_bow_wvs.npy')
HN_SCORED_COMMENT_BOW_WV_BASELINE = join(REMOTE_DATA, 'scored_comment_bow_wvs_baseline.npy')

# All users who have at least 20 comments (of at least 30 words)
HN_CLASSIFIED_USERS = join(REMOTE_DATA, 'classified_users.csv') 

# These are the original splits (60, 20, 20) of users.
TRAIN_EXAMPLES = join(LOCAL_DATA, 'train_20_50_200.csv')
DEV_EXAMPLES = join(LOCAL_DATA, 'dev_20_50_200.csv')
TEST_EXAMPLES = join(LOCAL_DATA, 'test_20_50_200.csv')

# These features are the binned similarity scores, from when we were only using the 
# embedding to compute log likelihood of speech.
WV_INITIAL_MODEL_FEATURES = join(REMOTE_DATA, 'features_wv_initial_model.csv')

LIFE_JACCARD_CHART = join(RESULTS, "life_stage_jaccard.png")
LIFE_DIST_BIGRAM_CHART = join(RESULTS, "life_stage_distance_bigram.png")
LIFE_DIST_WV_CHART = join(RESULTS, "life_stage_distance_wv.png")

# These are, for each user, their first 20 comments, where each comment is
# looked up in its monthly embedding (300 dimensions) and then the word vectors
# for the comment are averaged (bag of words strategy).
# The first 10 features are the standard frequency and activity bins; 
# then the next 20 * 300 = 6000 are these word vectors.
TRAIN_NN_FEATURES = join(REMOTE_DATA, 'train_nn_features.npz')
DEV_NN_FEATURES = join(REMOTE_DATA, 'dev_nn_features.npz')
TEST_NN_FEATURES = join(REMOTE_DATA, 'test_nn_features.npz')

# Same as above, except always using the Google News (initial) embedding.
BASELINE_TRAIN_NN_FEATURES = join(REMOTE_DATA, 'train_nn_features_baseline.npz')
BASELINE_DEV_NN_FEATURES = join(REMOTE_DATA, 'dev_nn_features_baseline.npz')
BASELINE_TEST_NN_FEATURES = join(REMOTE_DATA, 'test_nn_features_baseline.npz')

DNN_MODEL_DIR = join(REMOTE_DATA, 'tensorflow', 'DNN')

def get_month_filepath(year, month):
    "Returns the path to a CSV of a month's comments"
    return join(HN_MONTHLY_DIR, HN_MONTHLY_TEMPLATE.format(year, month))

def get_month_corpus_filepath(year, month, weighted=False):
    "Returns the filepath to a month's corpus (one comment per line, tokenized)"
    if weighted:
        return join(HN_MONTHLY_WEIGHTED_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))
    else:
        return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

def get_month_lm_filepath(year, month, ext="arpa"):
    "Returns the filepath to a month's language model file (in arpa or binary format)"
    return join(HN_MONTHLY_LM_DIR, HN_MONTHLY_LM_TEMPLATE.format(year, month, ext))

def get_month_embedding_filepath(year, month, weighted=False):
    "Returns filepath to largest (trainable) word2vec models"
    if weighted:
        return join(HN_MONTHLY_WEIGHTED_MODELS_DIR, HN_MONTHLY_MODEL_TEMPLATE.format(year, month))
    else:
        return join(HN_MONTHLY_MODELS_DIR, HN_MONTHLY_MODEL_TEMPLATE.format(year, month))

def get_month_word_vectors_filepath(year, month, weighted=False):
    """
    Returns the filepath to readonly word vectors (no more training possible) to be loaded:
    
        gensim.KeyedVectors.load(get_month_word_vectors_filepath(2007, 1))
    """
    if weighted:
        return join(HN_MONTHLY_WEIGHTED_WVS_DIR, HN_MONTHLY_WV_TEMPLATE.format(year, month))
    else:
        return join(HN_MONTHLY_WVS_DIR, HN_MONTHLY_WV_TEMPLATE.format(year, month))



