# Shared settings for all parts of the codebase. 
# Mostly, these are file and directory locations.
from os.path import join

REMOTE_STORATE = "/Volumes/Chris Proctor Backup/language_games/"

HN_DATA = "../data/hn_comments_clean.csv"
HN_CLEAN_DATA = "../data/hn_comments_utf8_text.csv"

KENLM_LMPLZ = "../lib/kenlm/build/bin/lmplz"
KENLM_BUILD_BINARY = "../lib/kenlm/build/bin/build_binary"

HN_MONTHLY_DIR = "../data/hn_monthly"
HN_MONTHLY_TEMPLATE = "hn_comments_{}_{}.csv"
HN_MONTHLY_COUNTS = "../data/hn_monthly_count.csv"
HN_MONTHLY_COUNT_CHART = "../results/hn_monthly_comments.png"

HN_MONTHLY_CORPUS_DIR = "../data/hn_corpus_monthly"
HN_MONTHLY_CORPUS_TEMPLATE = "hn_corpus_{}_{}.txt"

START_MONTH = "2007-02"
END_MONTH = "2017-09"

USERS = "../data/hn_users.csv"
USER_COUNTS = "../data/hn_user_counts.csv"
USER_INTERVALS = "../data/hn_user_intervals.csv"
USER_BASE = '../data/hn_user_base.csv'
USER_BASE_CHART = '../results/hn_user_base.png'

UPVOTES_HIST_DATA = "../data/hn_upvotes_hist.csv"
UPVOTES_HIST_CHART = "../results/hn_upvotes_hist.png"

USER_GRAPH = "../data/hn_user_graph.txt"
USER_DEGREE_DIST = "../data/hn_user_degree_dist.npy"
USER_DEGREE_CF_HIST = "../data/hn_user_degree_cf_hist.npy"
USER_DEGREE_DIST_CHART = "../results/hn_user_degree_distribution.png"
USER_DEGREE_DIST_CHART_WITH_EST = "../results/hn_user_degree_distribution_with_est.png"
USER_DEGREE_CF_HIST_CHART = "../results/hn_user_clustering_coefficients.png"

GOOGLE_NEWS_EMBEDDING_FILE = "../data/GoogleNews-vectors-negative300.bin"
HN_MONTHLY_MODELS_DIR = join(REMOTE_STORAGE, "hn_embeddings_monthly/")
HN_MONTHLY_MODEL_TEMPLATE = "hn_embed_{}_{}"

HN_MONTHLY_WVS_DIR = join(REMOTE_STORAGE, "hn_word_vectors_monthly/")
HN_MONTHLY_WV_TEMPLATE = "hn_wv_{}_{}"

INITIAL_MODEL = join(HN_MONTHLY_MODELS_DIR, 'initial')
