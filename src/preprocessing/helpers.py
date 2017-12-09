# Helpers for generate.py

import arrow
import pandas as pd
import sqlite3
import os
import nltk
import snap
from HTMLParser import HTMLParser
import io

def create_db_if_missing(csvfile, dbfile):
    "If the specified db does not exist, created it"
    if not os.path.exists(dbfile):
        conn = sqlite3.connect(dbfile)
        connn.text_factory = str
        for chunk in pd.read_csv(csvfile, header=None, chunksize=25000,
                names=["comment_text","points","author","created_at","object_id","parent_id"]):
            chunk.to_sql(name='comments', con=conn, if_exists='append', index=False,
            infer_datetime_format=True)

htmlParser = HTMLParser()
tokenizer = nltk.WordPunctTokenizer()

def clean_comment_text(series):
    text = str(series.comment_text).decode('ascii', errors='replace').encode('ascii', 'ignore').lower()
    text = htmlParser.unescape(text)
    return " ".join(tokenizer.tokenize(text))
                
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
    with io.open(csvfile, 'r', encoding='utf-8') as cf:
        comments = pd.read_csv(cf, header=None,
                names=["comment_text","points","author","created_at","object_id","parent_id"],
                parse_dates=["created_at"])
    comments['comment_text'] = comments.apply(clean_comment_text, axis=1)
    months = arrow.Arrow.span_range('month', arrow.get(start_month), arrow.get(end_month))
    for begin, end in months:
        currComments = comments[(comments.created_at >= begin.datetime) & (comments.created_at <= end.datetime)]
        print("{}-{}: {}".format(begin.format("YYYY-MM-DD"), end.format("YYYY-MM-DD"), len(currComments)))
        counts[begin.format("YYYY-MM")] = len(currComments)
        currComments.to_csv(get_monthly_filename(begin.year, begin.month), encoding='utf-8')
    return pd.DataFrame(counts.items(), columns=["month", "comments"])


def get_thread_text(comments):
    "Groups comments into threads, then concatenates the text of each thread."
    comments.object_id = comments.object_id.astype(int)
    comments.parent_id = comments.parent_id.astype(int)
    comments.points = comments.points.astype(float).astype(int)
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
        #commentsInThread = [c.encode('ascii', 'ignore') for c in commentsInThread]
        commentsInThread = [c.decode('ascii', errors='replace').encode('ascii', 'ignore') for c in commentsInThread]
        commentsInThread = [htmlParser.unescape(c) for c in commentsInThread]
        threadText.append(" ".join(commentsInThread))
    return " ".join(threadText)

def tokenize(commentsfile, weighted=False):
    """
    Converts a comments file into one-sentence-per-line tokens. 
    1. [if weighted] duplicate comments according to their weight]
    2. Group comments into threads
    3. Split into sentences
    4. Tokenize each sentence

    Params:
        commentsfile: filename (with headers)
    """
    with io.open(commentsfile, 'r', encoding='utf-8') as cf:
        comments = pd.read_csv(cf, usecols=['comment_text', 'object_id', 'parent_id', 'points'])

    if weighted:
        comments.points = comments.points.astype(float).astype(int)
        weightedComments = pd.DataFrame()
        for count in comments.points.unique():
            if count > 1:
                cComments = comments[comments.points == count]
                weightedComments = weightedComments.append([cComments] * (count-1), ignore_index=True)
        comments = comments.append(weightedComments)
        comments = comments.sample(frac=1).reset_index(drop=True) # shuffles comment order

    text = get_thread_text(comments)
    sentences = nltk.sent_tokenize(text)
    return [tokenizer.tokenize(sentence) for sentence in sentences]
    
