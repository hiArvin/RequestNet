import numpy as np
import tensorflow as tf

import time
import os
import random
import tensorflow as tf
from models import PEM
from utils import *

from topology import Topology
import networkx as nx
import pandas as pd
from label import gen_paths, solver

'''
与单条路径选择相比
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

topo = Topology(num_core=1, num_converge=1, num_access=1)
graph = topo.graph
# graph=nx.connected_caveman_graph(l=3,k=3)

NUM_EDGES = nx.number_of_edges(graph)
NUM_NODES = nx.number_of_nodes(graph)
NUM_PATHS = 8  # at least 4
NUM_QUESTS = 20

Min_Cap = 20  # *100
Max_Cap = 50  # *100

NUM_EPOCHES = 20
Reset_frq = 1

support, layer, bandwidth = nodeGraph_to_edgeGraph(graph, support=True)
bandwidth = np.random.randint(Min_Cap,Max_Cap,size=NUM_EDGES)*100
traffic = np.random.randint(Min_Cap//5 *100,Min_Cap//2 *100 ,size=NUM_EDGES )
pre_traffic = np.copy(traffic)
opt_traffic = np.copy(traffic)
sp_traffic = np.copy(traffic)
init_cap = np.copy(bandwidth)

flows = gen_flows_zte(topo, NUM_QUESTS * NUM_EPOCHES, 0.05)
print(flows)
import random
random.shuffle(flows)
# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(NUM_EDGES, NUM_EDGES)),
    'features': tf.placeholder(tf.float32),
    'labels': tf.placeholder(tf.int64, shape=(NUM_QUESTS, NUM_PATHS)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'paths': tf.placeholder(tf.int64),
    'index': tf.placeholder(tf.int64),
    'sequences': tf.placeholder(tf.int64)
}

# Create model
model = PEM(num_paths=NUM_PATHS, num_edges=NUM_EDGES, num_quests=NUM_QUESTS, placeholders=placeholders,
            gcn_input_dim=2 + NUM_PATHS * NUM_QUESTS, gcn_hidden_dim=16,
            gcn_output_dim=8, pe_output_dim=4, att_layers_num=3)

# Initialize session
sess = tf.Session(config=config)

# Load model

save_path = 'logs/0608-1957-Q30-P5-R5'  # 后面别加（/）
# save_path = 'logs/' + 'Q' + str(NUM_QUESTS) + '/' + 'P' + str(NUM_PATHS) + '-R' + str(Reset_frq)  # 后面别加（/）
model.load(sess, save_path)


# sess.run(tf.global_variables_initializer()) 这个千万别加


# update rest capacity
def update(choose, sp, traffic, init_cap):
    # c = np.argmax(choose, axis=1)
    # print('predictor:',c)
    success = np.ones([NUM_QUESTS], dtype=np.int16)
    for q in range(NUM_QUESTS):
        p = q * NUM_PATHS + choose[q]
        tmp = init_cap-traffic
        tmp = np.where(tmp<=0,0,tmp)
        tmp = tmp-sp[p, :]
        if np.any(tmp < 0):
            success[q] = 0
        traffic = traffic + sp[p, :]
    return traffic, np.nansum(success)


# Evaluate model
pre_success_num = 0
opt_success_num = 0
sp_success_num = 0

all_shortest_path = []

results_delay = np.zeros([3, NUM_EPOCHES // Reset_frq])
results_suc = np.zeros([3, NUM_EPOCHES // Reset_frq], dtype=np.int16)

traffic_pre = np.zeros(NUM_EDGES)
traffic_opt = np.zeros(NUM_EDGES)
traffic_sp = np.zeros(NUM_EDGES)
for epoch in range(NUM_EPOCHES):
    ep_flows = flows[epoch * NUM_QUESTS:(epoch + 1) * NUM_QUESTS]
    # predictor
    paths, idx_list, seqs, sp, shortest_path = \
        gen_paths(topo.graph, ep_flows, NUM_PATHS, NUM_QUESTS)
    ff = np.concatenate([[layer / np.max(layer)], [pre_traffic / bandwidth]], axis=0)
    fp = sp / np.tile(bandwidth, [NUM_QUESTS * NUM_PATHS, 1])
    f = np.concatenate([ff,fp])

    feed_dict = {
        placeholders['support']: support,
        placeholders['features']: f.T,
        placeholders['paths']: paths,
        placeholders['index']: idx_list,
        placeholders['sequences']: seqs,
    }

    t1 = time.time()
    out = sess.run(model.outputs, feed_dict=feed_dict)
    out = np.nanargmax(out, axis=1)
    print("Predictor outputs:", out)
    t2 = time.time()
    pre_traffic, suc_num = update(out, sp, pre_traffic,bandwidth)
    pre_success_num += suc_num

    # no parallel solver - shortest path selection
    sp_out = np.zeros_like(out)
    sp_t = 0
    sp_suc = 0
    for q in range(NUM_QUESTS):
        single_f = [ep_flows[q]]
        sp_sp = [shortest_path[q]]
        sp_o, sp_traffic, suc, t3 = solver(graph, sp_sp, single_f, sp_traffic,bandwidth, NUM_PATHS, 1)
        sp_out[q] = sp_o[0]
        sp_t += t3
        sp_suc += suc
    print('SP outputs:', sp_out)
    sp_success_num += sp_suc

    # OPT-Solver
    tmp = opt_traffic
    opt_out, opt_traffic, opt_suc_num, opt_t = solver(graph, shortest_path, ep_flows, opt_traffic, bandwidth, NUM_PATHS, NUM_QUESTS)
    print("OPT outputs:", opt_out)
    opt_success_num += opt_suc_num
    print('Epoch{epoch}: Predictor / Solver: {p_succ}/ {s_succ}'.format(epoch=epoch + 1, p_succ=suc_num,
                                                                        s_succ=opt_suc_num))

    # Summary
    print("Processing time:\n", 'Predictor:\t {:>.5f}\n'.format(t2 - t1),
          'SP:\t {:>.5f}\n'.format(sp_t), 'OPT:\t {:>.5f}\n'.format(opt_t))
    for q in range(NUM_QUESTS):
        c1 = q * NUM_PATHS + out[q]
        c2 = q * NUM_PATHS + sp_out[q]
        c3 = q * NUM_PATHS + opt_out[q]
        traffic_pre += sp[c1, :]
        traffic_sp += sp[c2, :]
        traffic_opt += sp[c3, :]

    if (epoch + 1) % Reset_frq == 0:
        batch = epoch // Reset_frq
        '''
        output info
        0 - pred
        1 - sp
        2 - opt
        '''
        pd_delay = cal_total_delay(traffic_pre, bandwidth)
        sp_delay = cal_total_delay(traffic_sp, bandwidth)
        opt_delay = cal_total_delay(traffic_opt, bandwidth)
        results_delay[0][batch] = pd_delay
        results_delay[1][batch] = sp_delay
        results_delay[2][batch] = opt_delay
        results_suc[0][batch] = pre_success_num
        results_suc[1][batch] = sp_success_num
        results_suc[2][batch] = opt_success_num

        print("Predictor : SP : OPT :\n {p}:{o}:{opt}".format(p=pre_success_num, o=sp_success_num,opt=opt_success_num))
        print("Predictor Delay: {:>.5f}".format(pd_delay))
        print("SP Delay:    {:>.5f}".format(sp_delay) )
        print("OPT Delay:    {:>.5f}".format(opt_delay))
        # reset cap
        print('Reset!')
        bandwidth = np.random.randint(Min_Cap, Max_Cap, size=NUM_EDGES) * 100
        traffic = np.random.randint(Min_Cap // 5 * 100, Min_Cap // 2 * 100, size=NUM_EDGES)
        pre_traffic = np.copy(traffic)
        opt_traffic = np.copy(traffic)
        sp_traffic = np.copy(traffic)
        # pre_traffic = np.zeros_like(bandwidth)
        # opt_traffic = np.zeros_like(bandwidth)
        # sp_traffic = np.zeros_like(bandwidth)
        pre_success_num = 0
        sp_success_num = 0
        opt_success_num = 0

print(results_delay)
print(results_suc)


np.save(save_path + '/delay', results_delay)
np.save(save_path + '/success', results_suc)

# interpretable
# att = sess.run(model.layers[2].att, feed_dict=feed_dict)
# idx = []
# for q in range(NUM_QUESTS):
#     idx.append(q*NUM_PATHS+out[q])
# att2 = np.average(np.squeeze(att)[idx,:],axis=0)
# print(att2)
# nx.draw(graph,with_labels=True)
# import matplotlib.pyplot as plt
# plt.show()
