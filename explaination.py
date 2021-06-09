from label import gen_paths, solver
from topology import Topology
from utils import *
import statsmodels.api as sm
import networkx as nx
import numpy as np

topo = Topology(num_core=1, num_converge=1, num_access=1)
graph = topo.graph
support, layer, bandwidth = nodeGraph_to_edgeGraph(graph, support=True)


NUM_PATHS = 7  # atleast 4
NUM_QUESTS = 10
flows = gen_flows_zte(topo,NUM_QUESTS,0.05)
paths, idx_list, seqs, sp, shortest_path = \
        gen_paths(topo.graph,flows,NUM_PATHS,NUM_QUESTS,max_cap=bandwidth)
opt_out,opt_cap, opt_suc_num, t3= solver(graph, shortest_path, flows, bandwidth, NUM_PATHS, NUM_QUESTS)
label=np.eye(NUM_QUESTS,NUM_PATHS,dtype=np.int16)[opt_out]
label=label.reshape(NUM_PATHS*NUM_QUESTS,1)
print(label.shape)

mask = np.tile(np.max(sp,axis=1,keepdims=True),[1,nx.number_of_edges(graph)])
x = sp/mask

print(x.shape)
x = sm.add_constant(x) # 若模型中有截距，必须有这一步
regression = sm.OLS(label, x).fit() # 构建最小二乘模型并拟合
print(regression.summary()) # 输出回归结果
weights=regression.params
print(weights)
print(bandwidth-opt_cap)