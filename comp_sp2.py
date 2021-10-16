import numpy as np
import tensorflow as tf

import time
import random
import tensorflow as tf
from models import PEM
from utils import *

from topology import Topology
import networkx as nx
import pandas as pd
from label import gen_paths, solver
'''
与计算式加权最短路径规划相比
'''
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True


topo = Topology(num_core=1, num_converge=1, num_access=1)
graph = topo.graph
# graph=nx.connected_caveman_graph(l=3,k=3)

NUM_EDGES = nx.number_of_edges(graph)
NUM_NODES = nx.number_of_nodes(graph)
NUM_PATHS = 5  # atleast 4
NUM_QUESTS = 200

NUM_EPOCHES = 1
Reset_frq = 1


support, layer, bandwidth = nodeGraph_to_edgeGraph(graph, support=True)
pre_cap = np.copy(bandwidth)
sp_cap = np.copy(bandwidth)
init_cap = np.copy(bandwidth)

flows = gen_flows_zte(topo,NUM_QUESTS*NUM_EPOCHES,0.05)

# # Define placeholders
# placeholders = {
#     'support': tf.placeholder(tf.float32, shape=(NUM_EDGES, NUM_EDGES)),
#     'features': tf.placeholder(tf.float32),
#     'labels': tf.placeholder(tf.int64, shape=(NUM_QUESTS, NUM_PATHS)),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'paths': tf.placeholder(tf.int64),
#     'index': tf.placeholder(tf.int64),
#     'sequences': tf.placeholder(tf.int64)
# }

# Create model
# model = PEM(num_paths=NUM_PATHS, num_edges=NUM_EDGES, num_quests=NUM_QUESTS, placeholders=placeholders,
#             gcn_input_dim=2+NUM_PATHS*NUM_QUESTS)

# Initialize session
# sess = tf.Session(config=config)

# Load model
# save_path = 'logs/1012-1935'+'-Q'+str(NUM_QUESTS)+'-P'+str(NUM_PATHS)+'-R'+str(Reset_frq) # 后面别加（/）
# model.load(sess, save_path)

# sess.run(tf.global_variables_initializer())


# update rest capacity
def update(choose, sp, capacity):
    # c = np.argmax(choose, axis=1)
    # print('predictor:',c)
    success = np.ones([NUM_QUESTS], dtype=np.int16)
    for q in range(NUM_QUESTS):
        p = q*NUM_PATHS+choose[q]
        tmp =capacity - sp[p,:]
        if np.any(tmp < 0):
            success[q]=0
        else:
            capacity=tmp
    return capacity, np.sum(success)

# Evaluate model
pre_success_num = 0
sp_success_num = 0

all_shortest_path = []
for epoch in range(NUM_EPOCHES):
    ep_flows = flows[epoch*NUM_QUESTS:(epoch+1)*NUM_QUESTS]
    # # predictor
    paths, idx_list, seqs, sp, shortest_path = \
        gen_paths(topo.graph,ep_flows,NUM_PATHS,NUM_QUESTS)
    # f = np.concatenate([[layer / np.max(layer)], [pre_cap / init_cap ]], axis=0)
    # f = np.concatenate([f, sp_numpy / np.tile(init_cap, [NUM_QUESTS * NUM_PATHS, 1])])
    #
    # feed_dict = {
    #     placeholders['support']: support,
    #     placeholders['features']: f.T,
    #     placeholders['paths']: paths,
    #     placeholders['index']: idx_list,
    #     placeholders['sequences']: seqs,
    # }

    t1 = time.time()
    # out = sess.run(model.outputs, feed_dict=feed_dict)
    # out = np.nanargmax(out,axis=1)
    # print("Predictor outputs:",out)
    t2 = time.time()
    # pre_cap, suc_num = update(out, sp_numpy, pre_cap)
    # pre_success_num += suc_num

    # no parallel solver - shortest path selection
    sp_out = np.zeros(NUM_QUESTS)
    sp_t = 0
    sp_suc = 0
    for q in range(NUM_QUESTS):
        single_f = [ep_flows[q]]
        sp_sp = [shortest_path[q]]
        sp_o, sp_cap, suc, t3 = solver(graph, sp_sp, single_f, sp_cap, NUM_PATHS, 1)
        sp_out[q] = sp_o[0]
        sp_t += t3
        sp_suc += suc
    print('SP outputs:', sp_out)
    sp_success_num+=sp_suc
    print("Processing time:", t2-t1, ' VS ', sp_t)

    if (epoch + 1) % Reset_frq == 0:
        # output info
        pd_delay = cal_total_delay(pre_cap, init_cap)
        sp_delay = cal_total_delay(sp_cap, init_cap)

        print("Predictor VS Solver : {p}/{o}".format(p=pre_success_num, o=sp_success_num))
        print("Predictor Delay: %.5f" % pd_delay)
        print("Solver Delay:    %.5f" % sp_delay)
        # reset cap
        print('Reset!')
        sp_cap = np.copy(bandwidth)
        pre_cap = np.copy(bandwidth)
        pre_success_num = 0
        sp_success_num = 0


# print("pd cap VS opt cap:",pre_cap,opt_cap)
# pre_delay=0
# opt_delay=0
# for q in range(TOTAL_FLOWS):
#     x,y = q//NUM_QUESTS,q%NUM_QUESTS
#     p_tmp = cal_path_delay(all_shortest_path[x][y][pd_outs[x][y]], pre_cap, init_cap)
#     o_tmp = cal_path_delay(all_shortest_path[x][y][sol_outs[x][y]], opt_cap, init_cap)
#     print(all_shortest_path[x][y][pd_outs[x][y]]==all_shortest_path[x][y][sol_outs[x][y]])
#     print("Pre VS Sol:",p_tmp,o_tmp)
#     pre_delay+=cal_path_delay(all_shortest_path[x][y][pd_outs[x][y]], pre_cap, init_cap)
#     opt_delay+= cal_path_delay(all_shortest_path[x][y][sol_outs[x][y]], opt_cap, init_cap)
# pre_delay/=TOTAL_FLOWS
# opt_delay/=TOTAL_FLOWS


