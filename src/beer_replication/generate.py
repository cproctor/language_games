# Partially replicating a baseline analytical strategy

# Danescu-Niculescu-Mizil, C., West, R., Jurafsky, D., Leskovec, J., & 
# Potts, C. (2013, May). No country for old members: User lifecycle and 
# linguistic change in online communities. In Proceedings of the 22nd 
# international conference on World Wide Web (pp. 307-318). ACM.

# author cp

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from helpers import *
from os.path import join
from os import listdir
import arrow
from tqdm import tqdm
import nltk
import io
from collections import Counter, defaultdict
import csv
from settings import *
from feature_extraction import ActivityFeatureExtractor
from gensim.models.keyedvectors import KeyedVectors
import kenlm

def get_month_filepath(year, month):
    return join(HN_MONTHLY_DIR, HN_MONTHLY_TEMPLATE.format(year, month))

def get_month_corpus_filepath(year, month):
    return join(HN_MONTHLY_CORPUS_DIR, HN_MONTHLY_CORPUS_TEMPLATE.format(year, month))

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
    plt.bar(year[:-1], enter[:-1], 0.5, color='green')
    plt.bar(year[:-1], bounce[:-1], 0.5, bottom=enter[:-1], color='yellow')
    plt.bar(year[:-1], exit[:-1], 0.5, bottom=np.sum([enter, bounce], axis=0)[:-1], color='red')
    plt.bar(year[:-1], stay[:-1], 0.5, bottom=np.sum([enter, bounce, exit], axis=0)[:-1], color='blue')
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

if True: # Score each comment we care about so we can later compute linguistic scores
    w = 20
    departed = 50
    living = 200
    WORDS_TO_CONSIDER = 30
    
    scored_comments = pd.DataFrame()
    user_counts = u = pd.read_csv(USER_COUNTS, usecols=['username', 'comment_count'])
    valid_users = u[(u.comment_count > w) & ((u.comment_count < departed) | u.comment_count >= living)]
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))

    for begin, end in months:
        print(begin.format("YYYY-MM"))
        comments = pd.read_csv(get_month_filepath(begin.year, begin.month))
        comments = valid_users.merge(comments, left_on='username', right_on='author')
        comments['clipped'] = comments['comment_text'].apply(lambda ct: " ".join(ct.split()[:WORDS_TO_CONSIDER]))

        bigram_model = kenlm.Model(get_month_lm_filepath(begin.year, begin.month, 'binary'))
        comments['bigram_score'] = comments['clipped'].map(lambda text: cross_entropy(bigram_model.score(text)))
        del bigram_model

        wv_model = KeyedVectors.load(get_month_word_vectors_filepath(begin.year, begin.month))
        comments['wv_score'] = comments['clipped'].apply(lambda text: wv_model.score(text.split()))
        del wv_model
        
        scored_comments.append(comments)

    scored_comments.to_csv(HN_SCORED_COMMENTS)


# EVERYTHING PAST HERE IS A DRAFT...
if False:
    comments = pd.read_csv(HN_DATA, header=None,
            names=["comment_text","points","author","created_at","object_id","parent_id"],
            parse_dates=['created_at'])
    comments = comments.merge(user_counts, left_on='username', right_on='username')
    comments = comments[(comments.comment_count > w) & 
            ((comments.comment_count < departed) | comments.comment_count > living)]
    comments.month = 12 * comments.created_at.dt.year + comments.created_at.dt.month
    

    comment_scores = comments['object_id']

if False: # Generate train, dev, test set split of users, and generate features for each.
    users = pd.read_csv(USERS, usecols=['id', 'username'])
    comments = pd.read_csv(HN_DATA, header=None,
            names=["comment_text","points","author","created_at","object_id","parent_id"],
            parse_dates=['created_at'])
    user_counts = pd.read_csv(USER_COUNTS, usecols=['username', 'comment_count'])
    users = users.merge(user_counts, left_on='username', right_on='username')
    #train, validate, test = np.split(users.sample(frac=1), [int(.6*len(users)), int(.8*len(users))])
    dummy = users.sample(100)
    dummy = classify_users(dummy, 20, 40, 200)

    examples = pd.DataFrame()
    activityFE = ActivityFeatureExtractor()

    for u in dummy:
        print(u.username)
        uc = comments[comments.username == u.username]
        features = activityFE.extract_features(uc[:20])
        features['label'] = u.label
        examples.append(features)

# Get all comments. 
# Merge on user_counts
# Filter out user counts we don't care about.
# Go through chronologically, generating scores for each comment we care about. Save to csv

# Now group comments by user.
# For each user, take the first 20. 
# Make features, lookup scores, save features to CSV. [username, features...., label]

# Divide users into train, dev, test
    






