# Generates graphs from csv comment data. 
# Author: cproctor

# Note: Requires snap.py (http://snap.stanford.edu/snappy/) for graph manipulation. 
# Not supported in venv, so follow local install instructions and make sure venv
# is not active. Also, GraphViz must be installed. `brew install graphviz` if you use
# homebrew. 

import pandas
import arrow
import snap
import csv
from itertools import combinations
from timer import Timer
import matplotlib.pyplot as plt
from tqdm import tqdm

HN_DATA = "../../data/hn_comments_clean.csv"
USERS = "../../data/hn_users.csv"
USER_GRAPH = "../../data/hn_user_graph.txt"

def make_edge(n1, n2):
    "Normalizes node order in an edge"
    return (n1, n2) if n1 < n2 else (n2, n1)

def labels(G):
    "Generates a labels vector with node ids"
    labels = snap.TIntStrH()
    for NI in Graph.Nodes():
        labels[NI.GetId()] = str(NI.GetId())
    return labels

def build_comment_graph(limit=None):
    """
    Iterates over comments, building a tree graph. For testing, may specify limit of how many comments to process.
    """
    comments = pandas.read_csv(HN_DATA, header=None, nrows=limit, 
            names=["comment_text","points","author","created_at","object_id","parent_id"],
            usecols=["author","object_id","parent_id"])
    comments.object_id = comments.object_id.astype(int)
    comments.parent_id = comments.parent_id.astype(int)
    nodes = set(comments['object_id']).union(set(comments['parent_id']))
    commentsGraph = snap.TUNGraph.New()
    for node in nodes: 
        try: 
            commentsGraph.AddNode(node)
        except TypeError:
            print("Error for: {}".format(node))
            continue

    for edge in comments[['object_id', 'parent_id']].values.tolist(): commentsGraph.AddEdge(*edge)
    commentThreads = snap.TCnComV()
    snap.GetSccs(commentsGraph, commentThreads)
    usersGraph = snap.TUNGraph.New()
    users = pandas.DataFrame({'username': comments['author'].unique()})
    for i, user in users.iterrows(): usersGraph.AddNode(i)
    for commentThread in tqdm(commentThreads):
        commentsInThread = comments[comments['object_id'].isin(commentThread)]
        usersInThread = users[users['username'].isin(commentsInThread['author'].values)]
        for u1, u2 in combinations(usersInThread.index, 2):
            usersGraph.AddEdge(u1, u2)
    return users, usersGraph
            
if __name__ == '__main__':
    users, G = build_comment_graph()
    snap.SaveEdgeList(G, USER_GRAPH)
    users.to_csv(USERS, index_label='id')
    #snap.DrawGViz(G, snap.gvlNeato, "output.png", "Comment graph structure")
