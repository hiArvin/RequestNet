import numpy as np
import networkx as nx
from utils import*

def random_network(num_nodes):
    adj=np.random.randint(0,2,[num_nodes,num_nodes]).astype(np.float32)
    adj=np.triu(adj)
    adj+= adj.T+np.diag(adj.diagonal())
    G = nx.from_numpy_matrix(adj)
    A = nx.laplacian_matrix(G).toarray().astype(np.float32)
    return adj, A


def random_features(num_nodes, num_features):
    norm=2
    f = np.random.randint(0,2,[num_nodes,num_features]).astype(np.float32)
    return f/norm,

def random_quests(low, high, num_node, num_quest):
    s_d=np.random.randint(low,high,[num_quest,2]).astype(np.float32)
    flow = np.random.randint(0,num_node,[num_quest,1]).astype(np.float32)
    quests=np.concatenate([s_d,flow],axis=1)
    return quests


def random_slices(num_node,max_quest,quest_range,num_quests):

    slices = []
    flow_sum = []
    for i in range(3):
        quest = random_quests(quest_range[i][0], quest_range[i][-1], num_node, num_quests[i])
        flow_sum.append(np.sum(quest, axis=1)[-1])
        slices.append(quest)

    l = np.array(flow_sum)[np.newaxis, :]
    l = np.repeat(l, num_node, axis=0)

    for i in range(3):
        for j in quest_range[i]:
            l[j, i] *= 2

    l = l / np.max(l)  # normalization
    l = np.around(l,decimals=1)
    return slices, softmax(l)
