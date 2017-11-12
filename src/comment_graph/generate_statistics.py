# Generate some initial statistics from the user graph
# Author: cp

# Goals:
# average clustering coefficient
# plot degree distribution
import snap
import numpy as np
import matplotlib.pyplot as plt
from timer import Timer
from random import sample
import yaml

USER_GRAPH = "../../data/hn_user_graph.txt"
USER_DEGREE_DIST = "../../data/hn_user_degree_dist.npy"
USER_DEGREE_CF_HIST = "../../data/hn_user_degree_cf_hist.npy"
USER_DEGREE_DIST_CHART = "../../results/hn_user_degree_distribution.png"
USER_DEGREE_CF_HIST_CHART = "../../results/hn_user_clustering_coefficients.png"

def get_nodes(G):
    "Returns a list of node ids"
    return [n.GetId() for n in G.Nodes()]

def get_degree_distribution(Graph):
    "Returns degree distribution data points"
    vec = snap.TIntPrV()
    snap.GetDegCnt(Graph, vec)
    bins = [(b.GetVal1(), b.GetVal2()) for b in vec]
    X, Y = zip(*bins)
    Y = np.array(Y) / float(np.sum(Y))
    return X, Y

def get_clustering_coefficient(Graph, sample=100000):
    cfVec = snap.TFltPrV()
    cf = snap.GetClustCf(Graph, cfVec, sample)
    return cf, [(pair.GetVal1(), pair.GetVal2()) for pair in cfVec]

class MLLEPowerDist:
    def fit(self, sample):
        self.x_min = 1
        self.alpha = np.max(np.roots([1, -1, -len(sample)/(np.array(sample) ** -1).sum()]))
        return self

    def predict(self, x):
        return np.exp(self.x_min - self.alpha * np.log(x))

# =============================================================
timer = Timer()
G = snap.LoadEdgeList(snap.PUNGraph, USER_GRAPH, 0, 1, '\t')
timer.elapsed("Time to load graph")

if True:
    plt.clf()
    X, Y = get_degree_distribution(G)
    np.save(USER_DEGREE_DIST, np.array([X, Y]))
    plt.loglog(X, Y)
    plt.title("Hacker News degree distribution")
    plt.xlabel('Node degree (log)')
    plt.ylabel('Proportion of nodes with a given degree (log)')
    plt.savefig(USER_DEGREE_DIST_CHART)
    timer.elapsed("Time to get degree distribution")

if True:
    plt.clf()
    samplesize = 10000
    cf, cfHist = get_clustering_coefficient(G, sample=samplesize)
    X, Y = zip(*cfHist)
    np.save(USER_DEGREE_CF_HIST, np.array([X, Y]))
    print("Average CF: {}".format(cf))
    plt.loglog(X, Y)
    plt.title("Hacker News average clustering coefficient by degree")
    plt.xlabel('Node degree (log)')
    plt.ylabel('Average clustering coefficient (log)')
    plt.savefig(USER_DEGREE_CF_HIST_CHART)
    timer.elapsed("Time to get cf with {} samples".format(samplesize))

if True:
    plt.clf()
    samplesize = 10000
    nodeIds = sample(get_nodes(G), samplesize)
    degrees = np.array([G.GetNI(nid).GetOutDeg() for nid in nodeIds]).astype(float)
    model = MLLEPowerDist().fit(degrees)
    domain = np.logspace(0, 4.8, 50)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(domain, model.predict(domain), color="red")
    X, Y = np.load(USER_DEGREE_DIST)
    plt.scatter(X, Y)
    plt.legend(["Est. power law dist (alpha={})".format(model.alpha), "Degree distribution"])
    plt.title("Hacker News degree distribution")
    plt.xlabel('Node degree (log)')
    plt.ylabel('Proportion of nodes with a given degree (log)')
    plt.savefig(USER_DEGREE_DIST_CHART)
    timer.elapsed("Time to estimate power law alpha with {} samples".format(samplesize))



