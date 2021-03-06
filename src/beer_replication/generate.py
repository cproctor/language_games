# Partially replicating a baseline analytical strategy

# Danescu-Niculescu-Mizil, C., West, R., Jurafsky, D., Leskovec, J., & 
# Potts, C. (2013, May). No country for old members: User lifecycle and 
# linguistic change in online communities. In Proceedings of the 22nd 
# international conference on World Wide Web (pp. 307-318). ACM.

# author cp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
#from helpers import *
from os.path import join
from os import listdir
import arrow
from tqdm import tqdm
import nltk
import io
import os
from collections import Counter, defaultdict
import csv
from settings import *
from feature_extraction import *
from gensim.models.keyedvectors import KeyedVectors
import kenlm

def get_month_filepath(year, month):
    return join(HN_MONTHLY_DIR, HN_MONTHLY_TEMPLATE.format(year, month))

def get_month_corpus_filepath(year, month):
    return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

STANFORD = ["#53284f", "#0098db", "#eaab00", "#009b76", "#007c92", "#e98300"]
S = STANFORD

# Table 1 (p. 3)
if False: # Extremely slow. ~ 2 hours.
    tokenizer = nltk.WordPunctTokenizer()
    with io.open(HN_DATA, 'r', encoding='utf-8') as cf:
        comments = pd.read_csv(cf, header=None,
        names=["comment_text","points","author","created_at","object_id","parent_id"],
        usecols=['comment_text'])
    comments['text'] = comments.apply(lambda s: str(s.comment_text).decode('ascii', errors='replace').encode('ascii', 'ignore'), axis=1)
    comments['sentences'] = comments.apply(lambda s: len(nltk.sent_tokenize(s.text)), axis=1)
    comments['words'] = comments.apply(lambda s: len(tokenizer.tokenize(s.text)), axis=1)
    print("Number of posts: {}".format(len(comments)))
    print("Median number of words per post: {}".format(comments.words.median()))
    print("Median number of sentences per post: {}".format(comments.sentences.median()))
    comments[['sentences', 'words']].to_csv("hn_comments_length_stats.csv")

if False:
    comments = pd.read_csv(HN_DATA, header=None,
        names=["comment_text","points","author","created_at","object_id","parent_id"],
        usecols=['author'])
    userCommentCount = Counter(comments.author.values)
    usernames, counts = zip(*userCommentCount.items())
    pd.DataFrame({"username": usernames, "comment_count": counts}).to_csv(USER_COUNTS)

if False: # Get number of users over 50
    user_counts = pd.read_csv(USER_COUNTS)
    print(len(user_counts[user_counts.comment_count > 50]))

if False:  # Generate number of users coming and going over the years
    users = pd.read_csv(USERS)
    comments = pd.read_csv(HN_DATA, header=None,
        names=["comment_text","points","author","created_at","object_id","parent_id"],
        usecols=['author', 'created_at'], parse_dates=['created_at'])
    bounds = defaultdict(lambda: {'enter': None, 'exit': None})
    for i, comment in comments.iterrows():
        if not bounds[comment.author]['enter']: 
            bounds[comment.author]['enter'] = comment.created_at
        bounds[comment.author]['exit'] = comment.created_at
    boundsDf = pd.DataFrame.from_dict(bounds, orient='index')
    users = users.merge(boundsDf, left_on='username', right_index=True)
    users.to_csv(USER_INTERVALS)

if False: # Continued
    users = pd.read_csv(USER_INTERVALS, parse_dates=['enter', 'exit'])
    years = defaultdict(lambda: {'enter': 0, 'exit': 0, 'bounce': 0, 'stay': 0})
    year = []
    bounce = []
    enter = []
    exit = []
    stay = []
    for begin, end in arrow.Arrow.span_range('year', arrow.get(START_MONTH), arrow.get(END_MONTH)):
        begins = (users.enter > begin.datetime) & (users.enter < end.datetime)
        ends = (users.exit > begin.datetime) & (users.exit < end.datetime)
        year.append(begin.year)
        bounce.append(len(users[begins & ends]))
        enter.append(len(users[begins & ~ends]))
        exit.append(len(users[~begins & ends]))
        stay.append(len(users[(users.enter < begin.datetime) & (users.exit > end.datetime)]))
    pd.DataFrame({'year': year, 'bounce': bounce, 'enter': enter, 'exit': exit, 'stay': stay}).to_csv(USER_BASE)
    plt.clf()
    plt.bar(year[:-1], enter[:-1], 0.5, color=S[0])
    plt.bar(year[:-1], bounce[:-1], 0.5, bottom=enter[:-1], color=S[1])
    plt.bar(year[:-1], exit[:-1], 0.5, bottom=np.sum([enter, bounce], axis=0)[:-1], color=S[2])
    plt.bar(year[:-1], stay[:-1], 0.5, bottom=np.sum([enter, bounce, exit], axis=0)[:-1], color=S[3])
    plt.xlabel("Year")
    plt.ylabel("Number of users")
    plt.title("Change in Hacker News User Base")
    plt.legend(["Joining", "Bouncing", "Leaving", "Staying"], loc=2)
    plt.savefig(USER_BASE_CHART)
        
if False: # Generate chart for number of upvotes
    comments = pd.read_csv(HN_DATA, header=None,
        names=["comment_text","points","author","created_at","object_id","parent_id"],
        usecols=['points'])
    upvotes = Counter(comments['points'])
    with open(UPVOTES_HIST_DATA, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(upvotes.items())
    plt.hist(comments['points'], bins=100, log=True)
    plt.title("Point values of HN comments")
    plt.xlabel("Points")
    plt.ylabel("Number of comments")
    plt.savefig(UPVOTES_HIST_CHART)

if False: # Score each comment we care about so we can later compute linguistic scores
    w = 20
    departed = 50
    living = 200
    WORDS_TO_CONSIDER = 30
    
    scored_comments = pd.DataFrame()
    user_counts = pd.read_csv(USER_COUNTS, usecols=['username', 'comment_count'])
    valid_users = user_counts[
        (user_counts.comment_count > w) & 
        ((user_counts.comment_count < departed) | (user_counts.comment_count >= living))
    ]
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))

    for begin, end in months:
        print(begin.format("YYYY-MM"))
        comments = pd.read_csv(get_month_filepath(begin.year, begin.month))
        clip = lambda text: text.split()[:WORDS_TO_CONSIDER]
        comments = comments.assign(clipped=comments['comment_text'].astype(str).apply(clip))
        comments = comments[comments.clipped.str.len() == WORDS_TO_CONSIDER]
        comments = valid_users.merge(comments, left_on='username', right_on='author')

        bigram_model = kenlm.Model(get_month_lm_filepath(begin.year, begin.month, 'binary'))
        comments = comments.assign(bigram_score = comments['clipped'].apply(
                lambda text: cross_entropy(bigram_model.score(" ".join(text)))))
        del bigram_model
        wv_model = KeyedVectors.load(get_month_embedding_filepath(begin.year, begin.month))
        comments = comments.assign(wv_score = cross_entropy(wv_model.score(comments.clipped)))
        del wv_model
        scored_comments = scored_comments.append(comments[['object_id', 'bigram_score', 
                'wv_score', 'created_at', 'username']])
    scored_comments.to_csv(HN_SCORED_COMMENTS)

# June 13: Adding points and popularity cutoffs to scored comments. 
# Points is the number of points, pop1, pop3, pop5, pop10, pop20 
# are binary indicators of a threshold popularity.
# Also re-sorted scored comments by created_at (see last line) so that subsequent
# feature extraction will be based on the correct comments.
# June 13: DONE
if False:
    scored_comments_full = pd.read_csv(HN_SCORED_COMMENTS_FULL, index_col=0)
    scored_comments_full = scored_comments_full.assign(
        pop1=scored_comments_full.points >= 1,
        pop3=scored_comments_full.points >= 3,
        pop5=scored_comments_full.points >= 5,
        pop10=scored_comments_full.points >= 10,
        pop20=scored_comments_full.points >= 20,
    )
    scored_comments = pd.read_csv(HN_SCORED_COMMENTS, index_col=0)
    scored_comments = scored_comments.merge(
        scored_comments_full[['object_id', 'points', 'pop1', 'pop3', 'pop5', 'pop10', 'pop20']],
        left_on='object_id', right_on='object_id'
    )
    scored_comments.sort_values('created_at').to_csv(HN_SCORED_COMMENTS)


if False: #Score comments based on the everyday language model (Google News)
    WORDS_TO_CONSIDER = 30
    COMMENTS_UPPER_BOUND = 7000000 # For word2vec scoring algorithm
    
    comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=['object_id', 'comment_text'])
    wv_model = KeyedVectors.load(INITIAL_MODEL)

    clip = lambda text: text.split()[:WORDS_TO_CONSIDER]
    comments = comments.assign(clipped=comments['comment_text'].astype(str).apply(clip))
    comments = comments[comments.clipped.str.len() == WORDS_TO_CONSIDER]

    comments = comments.assign(initial_wv_score = cross_entropy(wv_model.score(comments.clipped, 
            total_sentences=COMMENTS_UPPER_BOUND)))
    comments.to_csv(HN_SCORED_COMMENTS_INITIAL_MODEL)

# Updated June 13
# Working with HN_SCORED_COMMENTS (filtered to include only comments from departed or living users),
# map users (and comments) -> examples (features and labels)
# Then generate train, dev, test set split of users.
# June 13: DONE
if False: 
    featureExtractors = ([ActivityFeatureExtractor(), LinguisticFeatureExtractor()] + 
        [PopularityFeatureExtractor(threshold=n) for n in [1,3,5,10,20]])
    examples = []
    
    users = pd.read_csv(HN_CLASSIFIED_USERS, usecols=['id', 'username', 'label'])
    comments = pd.read_csv(HN_SCORED_COMMENTS, parse_dates=['created_at'])
    for i, u in tqdm(users.iterrows(), total=17896):
        uc = comments[comments.username == u.username][:20]
        if len(uc) < 20: continue # I checked before for user comment counts, but 
        example = {'label': u.label, 'username': u.username}
        for fe in featureExtractors:
            example.update(fe.extract_features(uc))
        examples.append(example)

    examples = pd.DataFrame(examples)
    shuffled_examples = examples.sample(frac=1, random_state=RANDOM_STATE)
    train, dev, test = np.split(shuffled_examples, [int(.6*len(examples)), int(.8*len(examples))])
    train.to_csv(TRAIN_EXAMPLES, index=False)
    dev.to_csv(DEV_EXAMPLES, index=False)
    test.to_csv(TEST_EXAMPLES, index=False)

# So our original word vector results aren't great. To test out more features, I want a quicker mapping
# of users to their comments. Now I'm going to generate word vector features
if False: 
    comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=['comment_text', 'created_at'],
            parse_dates=['created_at'])
    comment_wvs = np.zeros((len(comments), 300))
    groups = comments.groupby([comments.created_at.dt.year, comments.created_at.dt.month], sort=False)
    for (year, month), month_comments in groups:
        print("{}-{}".format(year, month))
        wv_model = KeyedVectors.load(get_month_word_vectors_filepath(year, month))
        for ix, c in month_comments.iterrows():
            cl = [w for w in str(c.comment_text).split() if w in wv_model.vocab]
            if any(cl):
                wv = wv_model.wv[cl].mean(axis=0)  
                comment_wvs[ix] = wv
        del wv_model
    np.save(HN_SCORED_COMMENT_BOW_WV, comment_wvs)

# Repeating the previous section, but now creating a lookup for the GloVE BoW vector
# of all scored comments (or at least those falling within the trained models.
if True: 
    def glove2dict(glove_filename):
        with open(glove_filename) as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            embed = {}
            vocab = []
            for line in reader:
                embed[line[0]] = np.array(list(map(float, line[1:])))
                vocab.append(line[0])
        return vocab, embed
    
    vocab, _ = glove2dict(TRUNCATED_GLOVE_EMBEDDING)
    comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=['comment_text', 'created_at'],
            parse_dates=['created_at']).sort_values('created_at')
    comment_wvs = np.zeros((len(comments), 300))
    groups = comments.groupby([comments.created_at.dt.year, comments.created_at.dt.month], sort=False)
    for (year, month), month_comments in groups:
        if not os.path.exists(get_month_glove_embedding_filepath(year, month)):
            print("Skipping {}-{} because there is no embedding.".format(year, month))
            continue
        print("{}-{}".format(year, month))
        arr = np.load(get_month_glove_embedding_filepath(year, month))
        emb = pd.Series([np.array(x) for x in arr.tolist()], index=vocab)
        for ix, c in month_comments.iterrows():
            cl = [w for w in str(c.comment_text).split() if w in vocab]
            if any(cl):
                wv = emb[cl].mean(axis=0)  
                comment_wvs[ix] = wv
    np.save(HN_SCORED_COMMENT_GLOVE_BOW_WV, comment_wvs)

# Those word vectors worked great. For comparison, I want to see what happens if I just use the original
# Word2Vec embedding instead of using each month's. This could be done in a more straightforward way, 
# But I already had the code for looking up via each monthly model and wanted to reduce the chance of errors.
if False:
    print("PRE-LOOKUP OF COMMENTS FROM BASELINE EMBEDDING")
    comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=['comment_text', 'created_at'],
            parse_dates=['created_at'])
    comment_wvs = np.zeros((len(comments), 300))
    groups = comments.groupby([comments.created_at.dt.year, comments.created_at.dt.month], sort=False)
    wv_model = KeyedVectors.load(INITIAL_MODEL).wv
    for (year, month), month_comments in groups:
        print("{}-{}".format(year, month))
        for ix, c in month_comments.iterrows():
            cl = [w for w in str(c.comment_text).split() if w in wv_model.vocab]
            if any(cl):
                wv = wv_model.wv[cl].mean(axis=0)  
                comment_wvs[ix] = wv
    np.save(HN_SCORED_COMMENT_BOW_WV_BASELINE, comment_wvs) 

# NOTE: (June 13) HN_SCORED_COMMENTS is now sorted by `created_at`, which is necessary for 
# accurately generating binned comments. HN_SCORED_COMMENTS_FULL is not sorted, preserving the
# original indexing on which the word vector lookup was generated. 

# I don't have time to re-generate the embedding BoW lookups. So the thing to do is capture
# `original_index` of HN_SCORED_COMMENTS_FULL, then sort by created_at, then grab 
# (now-scrambled) `original_index` and use this as a sort order on dimension 0 of
# HN_SCORED_COMMENT_BOW_WV_BASELINE and HN_SCORED_COMMENT_BOW_WV_BASELINE
# Need to write some tests on this...
# June 13: Complete
if False:
    print("PROPERLY SORTING BoW EMBEDDING LOOKUPS")
    ORIGINAL_INDEX = 0
    CREATED_AT = 7
    comments = pd.read_csv(HN_SCORED_COMMENTS_FULL, usecols=[ORIGINAL_INDEX, CREATED_AT], 
            parse_dates=['created_at'])
    comments.rename(columns={"Unnamed: 0": "original_index"}, inplace=True)
    comments = comments.sort_values('created_at')
    new_index = comments.original_index.values

    for embeddingFile in [HN_SCORED_COMMENT_BOW_WV, HN_SCORED_COMMENT_BOW_WV_BASELINE]:
        emb = np.load(embeddingFile, 'r')
        np.save(embeddingFile, emb[new_index])




