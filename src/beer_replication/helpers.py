# Helpers for generate.py

import arrow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
import nltk
import snap
from html.parser import HTMLParser
import io
import kenlm
from palettable.colorbrewer.qualitative import Set1_6 as colors
from sklearn.metrics import precision_recall_fscore_support

def create_db_if_missing(csvfile, dbfile):
    "If the specified db does not exist, created it"
    if not os.path.exists(dbfile):
        conn = sqlite3.connect(dbfile)
        connn.text_factory = str
        for chunk in pd.read_csv(csvfile, header=None, chunksize=25000,
                names=["comment_text","points","author","created_at","object_id","parent_id"]):
            chunk.to_sql(name='comments', con=conn, if_exists='append', index=False,
            infer_datetime_format=True)
                
def split_comments_by_month(csvfile, get_monthly_filename, start_month="2007-01", end_month="2017-09"):
    """
    Splits out a huge file full of comments into separate files, one per month.

    Params:
        csvfile: a file with all comments
        get_monthly_filename: a function mapping (year, month) -> filename
        start_month: string or datetime
        end_month: string or datetime
    """
    counts = {}
    comments = pd.read_csv(csvfile, header=None,
            names=["comment_text","points","author","created_at","object_id","parent_id"],
            parse_dates=["created_at"])
    months = arrow.Arrow.span_range('month', arrow.get(start_month), arrow.get(end_month))
    for begin, end in months:
        currComments = comments[(comments.created_at >= begin.datetime) & (comments.created_at <= end.datetime)]
        print("{}-{}: {}".format(begin.format("YYYY-MM-DD"), end.format("YYYY-MM-DD"), len(currComments)))
        counts[begin.format("YYYY-MM")] = len(currComments)
        currComments.to_csv(get_monthly_filename(begin.year, begin.month))
    return pd.DataFrame(counts.items(), columns=["month", "comments"])

htmlParser = HTMLParser()
tokenizer = nltk.WordPunctTokenizer()

def get_thread_text(comments):
    "Groups comments into threads, then concatenates the text of each thread."
    comments.object_id = comments.object_id.astype(int)
    comments.parent_id = comments.parent_id.astype(int)
    nodes = set(comments.object_id).union(set(comments.parent_id))
    commentsGraph = snap.TUNGraph.New()
    for node in nodes: 
        commentsGraph.AddNode(node)
    for edge in comments[['object_id', 'parent_id']].values.tolist(): 
        commentsGraph.AddEdge(*edge)
    commentThreads = snap.TCnComV()
    snap.GetSccs(commentsGraph, commentThreads)
    threadText = []
    for commentThread in commentThreads:
        commentsInThread = comments[comments['object_id'].isin(commentThread)]
        commentsInThread = commentsInThread.comment_text.astype(str) # No more floats in here...
        commentsInThread = [c.decode('ascii', errors='replace').encode('ascii', 'ignore') for c in commentsInThread]
        commentsInThread = [htmlParser.unescape(c) for c in commentsInThread]
        threadText.append(" ".join(commentsInThread))
    return " ".join(threadText)

def tokenize(commentsfile):
    """
    Converts a comments file into one-sentence-per-line tokens. 
    1. Group comments into threads
    3. Split into sentences
    4. Tokenize each sentence

    Params:
        commentsfile: filename (with headers)
    """
    with io.open(commentsfile, 'r', encoding='utf-8') as cf:
        comments = pd.read_csv(cf, usecols=['comment_text', 'object_id', 'parent_id'])
    text = get_thread_text(comments)
    sentences = nltk.sent_tokenize(text)
    return [tokenizer.tokenize(sentence) for sentence in sentences]
    
def cross_entropy(perplexity):
    if perplexity <= 0: return -100
    return np.log(-perplexity)/np.log(2)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class LongitudinalCE:
    "Pass a list of strings, each a filepath to a kenlm model"
    def __init__(self, model_files, labels=None):
        self.models = [kenlm.Model(f) for f in model_files]
        self.labels = labels or range(len(self.models))

    def score(self, sent):
        return [cross_entropy(model.score(sent)) for model in self.models]

    def plot(self, sentences):
        if isinstance(sentences, basestring):
            sentences = [sentences]
        x = range(len(self.labels))[4:-2]
        ax = plt.subplot()
        ax.set_color_cycle(colors.mpl_colors)
        for sentence in sentences:
            scores = self.score(sentence)
            #plt.scatter(x, scores, lw=0, alpha=0.5)
            plt.plot(x, smooth(scores, 6)[4:-2], lw=2)
        spacedLabels = [label if i % 6==0 else '' for i, label in enumerate(self.labels)][4:-2]
        plt.xticks(x, spacedLabels, rotation='vertical')
        plt.legend(sentences, loc=0)
        plt.xlabel("Time")
        plt.ylabel("Cross-entropy (less-surprising is lower)")
        plt.title("Hacker News Community change over time")
        plt.show()

def classify_users(users, visible, departed, living):
    """
    Classifies and filters users according to visible, departed, living. 
    Any users that have between visible and departed are departed; any with more
    than living are living. Others are filtered out.
    """
    users = users.assign(label=-1)
    users.loc[(users.comment_count > visible) & (users.comment_count < departed), 'label'] = 0
    users.loc[users.comment_count > living, 'label'] = 1
    return users[users.label != -1]

def add_min(data, feature_class_name):
    "Given a feature name, adds a new feature indicating the index of the minimum value for that feature"
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_min': np.argmin(data[features].values, axis=1)})

def add_max(data, feature_class_name):
    "Given a feature name, adds a new feature indicating the index of the maximum value for that feature"
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_max': np.argmax(data[features].values, axis=1)})

def feature_class(name, use_max=True, use_min=False):
    "Generates a list of strings for four buckets of features, optionally including min and max"
    suffixes = ['0', '1', '2', '3']
    if use_min: suffixes += ['min']
    if use_max: suffixes += ['max']
    return ["{}_{}".format(name, suffix) for suffix in suffixes]

def evaluate_model(description, model, trainX, testX, trainY, testY, with_train=False, 
        trainLabel="Train", testLabel="Test"):
    print("Training " + description + " on " + trainLabel + " -> " + testLabel)
    model.fit(trainX, trainY)
    results = []
    if with_train:
        yHat = model.predict(trainX)
        p, r, f1, s = precision_recall_fscore_support(trainY, yHat)
        results.append([trainLabel, trainLabel, description, p[1], r[1], f1[1], s[1]])
    yHat = model.predict(testX)
    p, r, f1, s = precision_recall_fscore_support(testY, yHat)
    results.append([trainLabel, testLabel, description, p[1], r[1], f1[1], s[1]])
    return results
