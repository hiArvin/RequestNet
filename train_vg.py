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

# tf config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

topo = Topology(num_core=1, num_converge=1, num_access=1)
graph = topo.graph

NUM_EDGES = nx.number_of_edges(graph)
NUM_NODES = nx.number_of_nodes(graph)
NUM_PATHS = 5  # at least 4
NUM_QUESTS = 20
refresh = 5
LR = 0.005  # 0.005 is the best until now
Min_Cap = 200  # *100
Max_Cap = 500  # *100
EPOCHS = 300
Fix_Graph = True
SAVE = False



support, layer, bd = nodeGraph_to_edgeGraph(graph, support=True)

flow_generator = topo.gen_flows(size_percent=0.05)


# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(NUM_EDGES, NUM_EDGES)),
    # 'support': tf.placeholder(tf.float32),
    'features': tf.placeholder(tf.float32),
    'labels': tf.placeholder(tf.int64, shape=(NUM_QUESTS, NUM_PATHS)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'paths': tf.placeholder(tf.int64),
    'index': tf.placeholder(tf.int64),
    'sequences': tf.placeholder(tf.int64)
}

# Create model
model = PEM(num_paths=NUM_PATHS, num_edges=NUM_EDGES, num_quests=NUM_QUESTS, placeholders=placeholders,
            learning_rate=LR, gcn_input_dim=2+ NUM_PATHS * NUM_QUESTS,
            gcn_hidden_dim=16, gcn_output_dim=8, pe_output_dim=4,att_layers_num=3)

# Initialize session
sess = tf.Session(config=config)
if SAVE:
    merged = tf.summary.merge_all()
    save_path = './logs/' + datetime.now().strftime('%m%d-%H%M')
    save_path+='-Q'+str(NUM_QUESTS)+'-P'+str(NUM_PATHS)+'-R'+str(refresh)
    summary_writer = tf.summary.FileWriter(save_path, graph=sess.graph)


def update(choose, sp, traffic, init_bd):
    # c = np.argmax(choose, axis=1)
    # print('predictor:',c)
    success = np.ones([NUM_QUESTS], dtype=np.int16)
    for q in range(NUM_QUESTS):
        p = q * NUM_PATHS + choose[q]
        tmp = init_bd - traffic
        tmp = np.where(tmp <= 0, 1, tmp)
        if np.any(tmp - sp[p, :] < 0):
            success[q] = 0
        traffic = traffic + sp[p, :]
    return traffic, np.nansum(success)


def gen_feed_dict():
    bandwidth = np.random.randint(Min_Cap,Max_Cap,size=NUM_EDGES) * 100
    traffic = np.random.randint(Min_Cap//5 * 100, Min_Cap//3 * 100, size=NUM_EDGES)

    print(traffic, bandwidth)
    flow = []
    for i in range(NUM_QUESTS):
        flow.append(next(flow_generator))
    paths, idx, seqs, labels, sp, shortest_path = gen_label(topo.graph, flow, traffic, bandwidth, target='delay',
                                                            num_paths=NUM_PATHS,
                                                            num_flows=NUM_QUESTS)
    print(shortest_path)
    # normalize
    ff = np.concatenate([[layer / np.max(layer)], [traffic / bandwidth]], axis=0)
    # process sp_numpy feature
    fp1 = sp / np.tile(bandwidth, [NUM_QUESTS * NUM_PATHS, 1])
    rest_cap = bandwidth - traffic
    s_fp = np.where(rest_cap <= 0, -100, rest_cap)
    fp2 = sp / np.tile(s_fp, [NUM_QUESTS * NUM_PATHS, 1])
    # repeat feature of flow
    f = np.concatenate([ff,fp1])
    # Construct feed dictionary
    feed_dict = construct_feed_dict(f.T, support, labels, paths, idx, seqs, placeholders)
    feed_dict.update({placeholders['dropout']: 0.})
    # feature[1, :],flag = update_capacity(feature[1, :], occupy)
    return feed_dict, labels, sp


# Init variables
sess.run(tf.global_variables_initializer())

# Train model
acc_num = 0
early_stop=0
for epoch in range(EPOCHS):
    feed_dict, labels, sp = gen_feed_dict()
    lb = np.nanargmax(labels, axis=1)
    print("Labels   :\t", lb)
    # update
    # if lb.max() == lb.min():
    #     continue
    # if (epoch+1) % refresh == 0:
    #     print('Bandwidth Reset!')
    #     traffic = np.zeros_like(bandwidth)
    # else:
    #     traffic, _ = update(lb, sp_numpy, traffic,bandwidth)
    # Training step
    outs = sess.run([model.outputs, model.loss, model.accuracy, model.opt_op], feed_dict=feed_dict)
    # Print results
    print('Predictor:\t', np.nanargmax(softmax(outs[0]), axis=1))
    # print("Outputs(line1):", outs[0][0])
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]))

    if SAVE:
        summary = sess.run(merged, feed_dict=feed_dict)
        summary_writer.add_summary(summary, epoch)

    # early-stop conditions
    # if outs[2]>=0.95:
    #     early_stop+=1
    #     if early_stop>5:
    #         break
    # else:
    #     early_stop=0

    # summary

    # if epoch >= TEST_START:
    #     acc_num += np.sum(np.equal(np.argmax(outs[0],axis=1), np.argmax(labels, axis=1)))
    #     print('Accumulated accuracy:', acc_num)

if SAVE:
    model.save(sess, save_path)
print("Optimization Finished!")
# print('Test accuracy:',acc_num/((EPOCHS-TEST_START)*NUM_QUESTS))
