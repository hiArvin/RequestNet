from __future__ import division
from __future__ import print_function

import time
import random
import tensorflow as tf
from models import *
from utils import *

from topology import Topology
import networkx as nx
from label import gen_label
from datetime import datetime

topo = Topology(num_core=1, num_converge=1, num_access=1)
graph = topo.graph
# graph=nx.connected_caveman_graph(l=3,k=3)

NUM_EDGES = nx.number_of_edges(graph)
NUM_NODES = nx.number_of_nodes(graph)
NUM_PATHS = 5  # atleast 4
NUM_QUESTS = 20



Min_Cap = 10  # *100
Max_Cap = 16  # *100
EPOCHS = 10

support, f_layer, f_bandwidth = nodeGraph_to_edgeGraph(graph, support=True)
flow_generator = topo.gen_flows(size_percent=0.05)

# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(NUM_EDGES, NUM_EDGES)),
    # 'support': tf.placeholder(tf.float32),
    'features': tf.placeholder(tf.float32, shape=(None, 2+NUM_PATHS*NUM_QUESTS)),
    'labels': tf.placeholder(tf.int64, shape=(NUM_QUESTS, NUM_PATHS)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'paths': tf.placeholder(tf.int64),
    'index': tf.placeholder(tf.int64),
    'sequences': tf.placeholder(tf.int64),
    'flow_size': tf.placeholder(tf.float32, shape=(NUM_QUESTS, 3))
}

# Create model
model = PEM(num_paths=NUM_PATHS, num_edges=NUM_EDGES, num_quests=NUM_QUESTS, placeholders=placeholders,
            gcn_input_dim=2+NUM_PATHS*NUM_QUESTS,gcn_hidden_dim=16, gcn_output_dim=8, pe_output_dim=2)

# Initialize session
sess = tf.Session()

# Load model
save_path = 'logs/0402-1910/'
model.load(sess, save_path)


def gen_feed_dict():
    bandwidth = random_capacity(Min_Cap, Max_Cap, NUM_EDGES)
    flow = []
    for i in range(NUM_QUESTS):
        flow.append(next(flow_generator))
    label_start =time.time()
    paths, idx, seqs, labels, flow_size, sp, simple_paths = gen_label(topo.graph, flow, bandwidth, num_paths=NUM_PATHS,
                                                    num_quests=NUM_QUESTS,max_cap=Max_Cap)
    print("Labels   :\t",np.argmax(labels,axis=1))
    print("labeling time: \t",time.time()-label_start)
    deduce_att = deduce_attention(simple_paths,labels)
    print("Deduced attention:\t",deduce_att)
    # normalize
    f = np.concatenate([[f_layer/np.max(f_layer)],[bandwidth/(Max_Cap*100)]],axis=0)
    f = np.concatenate([f,sp/np.tile(bandwidth,[NUM_QUESTS*NUM_PATHS,1])])
    # Construct feed dictionary
    feed_dict = construct_feed_dict(f.T, support, labels, paths, idx, seqs, placeholders)
    feed_dict.update({placeholders['dropout']: 0.})
    # feature[1, :],flag = update_capacity(feature[1, :], occupy)
    return feed_dict, labels

def deduce_attention(sp,labels):
    actions = np.nanargmax(labels,axis=1)
    paths = []
    for q in range(NUM_QUESTS):
        paths.append(sp[q][actions[q]])
    importance = np.zeros([NUM_QUESTS,NUM_QUESTS],dtype=np.int64)
    for q in range(NUM_QUESTS-1):
        for qq in range(q + 1, NUM_QUESTS):
            importance[q,qq]+=len(set(paths[q]).intersection(set(paths[qq])))
    importance += importance.T
    return np.nanargmax(importance,axis=1)

    # importance = np.zeros([NUM_QUESTS,NUM_QUESTS],dtype=np.int64)
    # for q in range(NUM_QUESTS-1):
    #     for qq in range(q+1,NUM_QUESTS):
    #         for p in range(NUM_PATHS):
    #             for pp in range(NUM_PATHS):
    #                 importance[q,qq]+=len(set(sp[q][p]).intersection(set(sp[qq][pp])))
    # importance+=importance.T
    # return np.nanargmax(importance,axis=1)

# Evaluate model
success_num = 0
opt_success_num = 0
for epoch in range(EPOCHS):
    print("Epoch:", '%04d' % (epoch + 1))
    feed_dict, labels = gen_feed_dict()
    # Training step
    predict_start = time.time()
    outs = sess.run([model.outputs, model.accuracy, model.layers[-2].att,model.layers[-3].att], feed_dict=feed_dict)
    # Print results

    pred = np.nanargmax(softmax(outs[0]),axis=1)
    path_att = outs[-2]
    link_att = np.squeeze(outs[-1])
    link_att_ch = np.zeros([NUM_QUESTS,outs[-1].shape[1]])
    for i in range(NUM_QUESTS):
        link_att_ch[i] = link_att[pred[i]+NUM_PATHS*i]
    print('Predictor:\t', pred)
    print('Prediction time:\t',time.time()-predict_start)
    print('Predicted Path Attention:\t',np.nanargmax(path_att,axis=1))
    print('Predicted Link Attention:\t',np.nanargmax(link_att_ch,axis=1))



