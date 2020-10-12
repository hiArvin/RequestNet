from __future__ import division
from __future__ import print_function

import time
import random
import tensorflow as tf
from models import *


from topology import Topology
import networkx as nx
import pandas as pd
from e_graph import *
from labelling import *

topo = Topology(num_core=1)
graph = topo.graph
# graph=nx.connected_caveman_graph(l=3,k=3)

num_edges = nx.number_of_edges(graph)
num_nodes = nx.number_of_nodes(graph)
num_paths = 3
num_quests = 5

def reset_features(graph):
    _,feature=nodeGraph_to_edgeGraph(graph)
    return feature

support, feature = nodeGraph_to_edgeGraph(graph,support=True)

flow_generator = topo.gen_flows()



# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32,shape=(num_edges,num_edges)),
    'features': tf.placeholder(tf.float32),
    'labels': tf.placeholder(tf.int64,shape=(num_quests,num_paths)),
    'dropout': tf.placeholder_with_default(0.,shape=()),
    'paths':tf.placeholder(tf.int64),
    'index':tf.placeholder(tf.int64),
    'sequences':tf.placeholder(tf.int64),
    'flow_size':tf.placeholder(tf.float32,shape=(num_quests,num_paths))
}

# Create model
model = PEM(num_paths=num_paths,num_quests=num_quests,placeholders=placeholders,learning_rate=0.01, input_dim=2, logging=True)

# Initialize session
sess = tf.Session()
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logs/', graph=sess.graph)

#
# # Define model evaluation function
def evaluate(features, support, labels,paths,idx,seqs, placeholders,flow_size):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels,paths,idx,seqs, placeholders,flow_size)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)
#
#
# Init variables
sess.run(tf.global_variables_initializer())

# Train model
epochs=100
for epoch in range(epochs):
    flow=[]
    for i in range(num_quests):
        flow.append(next(flow_generator))
    paths,idx,seqs,labels,occupy,flow_size = gen_label(topo.graph, flow, feature[1, :], num_paths=num_paths,num_quests=num_quests)

    feature[1,:]=update_capacity(feature[1,:],occupy)

    t = time.time()
    # Construct feed dictionary
    # f=np.zeros_like(feature,dtype=np.float)
    # f[0,:]=feature[0,:]/4
    # f[1,:]=feature[1,:]/100
    feed_dict = construct_feed_dict(feature.T, support, labels, paths,idx,seqs, placeholders,flow_size)
    feed_dict.update({placeholders['dropout']: 0.})

    # Training step
    outs = sess.run([model.outputs, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(feature.T, support, labels, paths,idx,seqs, placeholders,flow_size)

    # summary
    summary = sess.run(merged, feed_dict=feed_dict)
    summary_writer.add_summary(summary, epoch)

    if epoch % 20 ==0:
        feature=reset_features(graph)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
#
# # Testing
# paths,idx,seqs,labels,occupy = gen_label(topo.graph, [next(flow_generator)], feature[1, :], num_paths=num_paths)
# test_cost, test_acc, test_duration = evaluate(feature.T, adj, labels, paths,idx,seqs, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
