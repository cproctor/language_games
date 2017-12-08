# Generates graphs from csv comment data. 
# Author: cproctor

# Note: Requires snap.py (http://snap.stanford.edu/snappy/) for graph manipulation. 
# Not supported in venv, so follow local install instructions and make sure venv
# is not active. Also, GraphViz must be installed. `brew install graphviz` if you use
# homebrew. 

import pandas as pd
import arrow
import snap
import csv
from itertools import combinations
from timer import Timer
import matplotlib.pyplot as plt
from tqdm import tqdm

class UserGraphBuilder:
    "Given a CSV file of comments, builds a user graph based on users who participated in the same conversation"

    def __init__(self, datafile, chunksize=25000, verbose=True):
        self.datafile = datafile
        self.chunksize = chunksize
        self.verbose = verbose

    def build(self):
        "Iterates over chunks of the csv file, taking 2 at a time so there are no seams."
        self.usersGraph = snap.TUNGraph.New() 
        self.user_ids = {}
        self.user_names = {}
        self.next_user_id = 0
        for df in pandas.read_csv(self.datafile, header=None, chunksize=self.chunksize,
                names=["comment_text","points","author","created_at","object_id","parent_id"],
                usecols=["author","object_id","parent_id"]):
            if not hasattr(self, 'newchunk'): 
                self.newchunk = df
                continue
            self.oldchunk = self.newchunk
            self.newchunk = df
            self.build_chunk()
        return self.user_names, self.usersGraph

    def build_chunk(self):
        comments = pd.concat([self.oldchunk, self.newchunk])
        self.register_users(comments.author.unique())
        commentsGraph = self.build_comment_graph(comments)
        commentThreads = snap.TCnComV()
        snap.GetSccs(commentsGraph, commentThreads)
        for commentThread in commentThreads:
            commentsInThread = comments[comments['object_id'].isin(commentThread)]
            userIdsInThread = [self.user_ids[un] for un in commentsInThread.author.values]
            for u1, u2 in combinations(set(usersIdsInThread), 2):
                if not self.usersGraph.IsEdge(u1, u2):
                    self.usersGraph.AddEdge(u1, u2)
        
    def register_users(self, usernames):
        for username in usernames:
            if self.user_ids.get(username) is None:
                uid = self.new_user_id()
                self.user_ids[username] = uid
                self.user_names[uid] = username
                self.usersGraph.AddNode(uid)

    def build_comment_graph(self, comments):
        comments.object_id = comments.object_id.astype(int)
        comments.parent_id = comments.parent_id.astype(int)
        nodes = set(comments.object_id).union(set(comments.parent_id))
        commentsGraph = snap.TUNGraph.New()
        for node in nodes: 
            commentsGraph.AddNode(node)
        for edge in comments[['object_id', 'parent_id']].values.tolist(): 
            commentsGraph.AddEdge(*edge)
        return commentsGraph

    def new_user_id(self):
        self.next_user_id += 1
        return self.next_user_id - 1

if __name__ == '__main__':
    HN_DATA = "../../data/hn_comments_clean.csv"
    USERS = "../../data/hn_users.csv"
    USER_GRAPH = "../../data/hn_user_graph.txt"

    builder = UserGraphBuilder(HN_DATA)
    users, G = builder.build()
    snap.SaveEdgeList(G, USER_GRAPH)
    pd.DataFrame(users.items(), columns=["username", "id"]).to_csv(USERS)

    #snap.DrawGViz(G, snap.gvlNeato, "output.png", "Comment graph structure")


