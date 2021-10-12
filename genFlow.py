import os
import random
import networkx as nx
import numpy as np
from label import gen_paths, solver

# get paths
all_data_path = []
for f in os.listdir('dataset/'):
    if '.graphml' in f:
        all_data_path.append('dataset/' + f)
# random choose a graph
# g = random.choice(all_data_path)
g = all_data_path[1]
graph = nx.read_graphml(g, node_type=int, edge_key_type=int, force_multigraph=False)

print(graph.nodes, graph.edges)

n_edges = nx.number_of_edges(graph)
n_nodes = nx.number_of_nodes(graph)
bd = np.ones(n_edges) * 1000

# random traffic
traffic = np.random.randint(0, 90, n_edges) * 10

# flows
n_flows = 40
n_paths = 5

s_d = np.zeros([n_flows, 2])
for f in range(n_flows):
    s, d = random.sample(graph.nodes,2)
    assert s != d
    s_d[f][0] = s
    s_d[f][1] = d
# 可以用资源的紧张程度来衡量
mu = np.nanmean(bd-traffic)//(n_flows-n_flows/2)
sigma = 1
flow_size = np.random.normal(mu,sigma,[n_flows,1])
flow_size = np.where(flow_size<=0,1,flow_size)
print(flow_size)
flows = np.concatenate([s_d,flow_size],axis=1).astype(np.int16)
print(flows)
paths, idx_list, seqs, sp, shortest_path=gen_paths(graph,flows,n_paths,n_flows)

# opt solver
opt_out, opt_occupy, opt_success,opt_delay, opt_t=solver(graph,shortest_path,flows,traffic,bd,n_paths,n_flows)
opt_delay = np.where(opt_delay<=0,1,opt_delay)
print(opt_delay)

# seq solver
sol_occupy = np.copy(traffic)
sol_suc = 0
for f in range(n_flows):
    single_f = [flows[f]]
    single_paths =[shortest_path[f]]
    sol_out, sol_occupy, sol_success, sol_delay, sol_t = solver(graph, single_paths, single_f, sol_occupy, bd, n_paths, 1)
    sol_suc+=sol_success
print(sol_delay)
print('Delay:',sol_delay/opt_delay)
print("Suceess Ratio:",opt_success,'vs',sol_suc)
print('Delay Ratio:',np.nansum(sol_delay)/np.nansum(opt_delay))
