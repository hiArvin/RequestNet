import gurobipy as gb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from topology import *
from utils import *
import time
import random

M = 100

pre_start = time.time()
topo = Topology()
graph = topo.graph
# graph=nx.connected_caveman_graph(l=10, k=3)
# nx.draw(graph)
# plt.show()

num_edges = nx.number_of_edges(graph)
num_nodes = nx.number_of_nodes(graph)
num_quest = 10
num_paths = 3

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

quest = np.random.randint(0, num_nodes, [num_quest, 2])
for i in range(num_quest):
    if quest[i, 0] == quest[i, 1]:
        quest[i, 1] = random.randint(0, num_nodes - 1)

shortest_path = []
for q in range(num_quest):
    k_sp = k_shortest_paths(graph, source=quest[q, 0], target=quest[q, 1], k=num_paths)
    for i in range(num_paths - len(k_sp)):
        k_sp.append(k_sp[0])
    shortest_path.append(k_sp)

# transform shortest path into edge index
edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])
print(edges)
flow = np.random.randint(20, 100, [num_quest],dtype=np.int)
# print(flow)
sp = np.zeros([num_quest, num_paths, num_edges],dtype=np.int)
for q in range(num_quest):
    for p in range(num_paths):
        path=shortest_path[q][p]
        for i in range(len(path) - 1):
            idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                        ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
            sp[q, p, idx] = flow[q]

capacity = np.repeat([1000], num_edges)

pre_end = time.time()
print(f"Preprocessing time: {pre_end - pre_start}")

model = gb.Model()
slices = model.addVars(range(num_edges), range(num_quest), lb=0,ub=1000,vtype=gb.GRB.INTEGER, name='slice')
select_paths = model.addVars(range(num_quest), range(num_paths), lb=0, vtype=gb.GRB.BINARY, name='select_paths')
edge_flow = model.addVars(range(num_edges), range(num_quest), lb=0,ub=1000,vtype=gb.GRB.INTEGER, name='flow')

# diff = model.addVars(range(num_edges), range(num_quest), name='diff')
success = model.addVars(range(num_edges), range(num_quest), vtype=gb.GRB.BINARY, name='success')
scc = model.addVars(range(num_quest), vtype=gb.GRB.BINARY, name='scc')

model.update()

# objective
model.setObjective(gb.quicksum(scc), gb.GRB.MAXIMIZE)

# constraints
model.addConstrs(select_paths.sum(q, '*') == 1 for q in range(num_quest))
model.addConstrs(slices.sum(e, '*') == capacity[e] for e in range(num_edges))
model.addConstrs(scc[q] == gb.min_(success.select('*', q)) for q in range(num_quest))
model.addConstrs(edge_flow[e, q] == gb.quicksum(select_paths[q, p] * sp[q, p, e] for p in range(num_paths))
                 for q in range(num_quest) for e in range(num_edges))

# conditional constraints
# model.addConstrs( (success[e, q] == 1) >> (slices[e, q] >= edge_flow[e, q])
#                  for e in range(num_edges) for q in range(num_quest))
# model.addConstrs((success[e, q] == 0) >> (slices[e, q] < edge_flow[e, q])
#                  for e in range(num_edges) for q in range(num_quest))
model.addConstrs((success[e,q])*(slices[e,q].ub-edge_flow[e,q].lb)>= slices[e,q]-edge_flow[e,q]
                 for e in range(num_edges) for q in range(num_quest))
model.addConstrs((success[e, q])*1+(1-success[e,q])*(slices[e,q].lb-edge_flow[e,q].ub)<= slices[e,q]-edge_flow[e,q]
                 for e in range(num_edges) for q in range(num_quest))

model.optimize()

opt_time = time.time()

print("Obj:", model.objVal)
print("Runtime:",model.runtime)
res_scc = np.ones([num_quest], dtype=np.int)
res_selct = np.zeros([num_quest, num_paths], dtype=np.int)
res_slice = np.zeros([num_edges,num_quest],dtype=np.int)
res_flow = np.zeros([num_edges,num_quest],dtype=np.int)
for q in range(num_quest):
    v = model.getVarByName(f"scc[{q}]")
    if v.x == 0:
        res_scc[q] = 0
    for p in range(num_paths):
        var = model.getVarByName(f"select_paths[{q},{p}]")
        if var.x == 1:
            res_selct[q,p] = 1

for e in range(num_edges):
    for q in range(num_quest):
        ve = model.getVarByName(f"slice[{e},{q}]")
        vf = model.getVarByName(f"flow[{e},{q}]")
        if ve.x != 0:
            res_slice[e,q]=ve.x
        if vf.x!=0:
            res_flow[e,q]=vf.x


print(res_scc)
df_slt=pd.DataFrame(res_selct)
df_slt.to_csv('lp/res/select_paths.csv')
df_slice=pd.DataFrame(res_slice)
df_slice.to_csv('lp/res/slices_results.csv')
df_flow=pd.DataFrame(res_flow)
df_flow.to_csv('lp/res/flow_results.csv')
print(f"Failure rate:{1-(model.objVal / num_quest)}")
print(f"Time for modelling optimization: {opt_time - pre_end}")

f=np.multiply(res_flow,res_scc)
print(np.sum(f,axis=1))
capacity=capacity-np.sum(f,axis=1)
print(capacity)
# model.printAttr('X')
