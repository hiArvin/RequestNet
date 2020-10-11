import gurobipy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

pre_start = time.time()
graph = nx.connected_caveman_graph(l=1000, k=3)
nx.draw(graph)
plt.show()

num_edges = nx.number_of_edges(graph)
num_nodes = nx.number_of_nodes(graph)
num_quest = 1000
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

quest = np.random.randint(1, num_nodes, [num_quest, 2])
shortest_path = []

for q in range(num_quest):
    shortest_path.append(nx.shortest_path(graph, source=quest[q, 0], target=quest[q, 1]))

# transform shortest path into edge index
edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])

sp = []
for q in shortest_path:
    e = []
    for i in range(len(q) - 1):
        idx = edges[((edges.src == q[i]) & (edges.dst == q[i + 1])) |
                    ((edges.dst == q[i]) & (edges.src == q[i + 1]))].index[0]
        e.append(idx)
    sp.append(e)

# capacity = np.random.randint(1000, 2000, [num_edges])
capacity = np.repeat([1000], num_edges)
flow = np.random.randint(30, 50, [num_quest])
net_flow = np.zeros([num_edges,num_quest],dtype=np.int)
for q in range(num_quest):
    for e in sp[q]:
        net_flow[e,q]=flow[q]
# print(net_flow)
pre_end=time.time()
print(f"Preprocessing time: {pre_end-pre_start}")

def compute_delay(f, c):
    if 3 * f <= c:
        return f
    elif 3 * f > c and 3 * f <= 2 * c:
        return 3 * f - 2 / 3 * c
    elif 3 * f > 2 * c and 10 * f <= 9 * c:
        return 10 * f - 16 / 3 * c
    elif 10 * f > 9 * c and f <= c:
        return 70 * f - 178 / 3 * c
    elif f > c and 10 * f <= c * 11:
        return 500 * f - 1468 / 3 * c
    else:
        return 5000 * f - 16318 / 3 * c


model = gurobipy.Model()
'''
## slice mode
slice = model.addVars(range(num_edges), range(num_quest), name='slice', lb=1, vtype=gurobipy.GRB.INTEGER)
utility = model.addVars(range(num_edges), range(num_quest), name='utility')
q_util = model.addVars(range(num_quest), name='utility of a quest')
# delay = model.addVars(range(num_edges),range(num_quest),name='delay')
model.update()
# utility objective
model.setObjective(gurobipy.quicksum(q_util.select('*')), sense=gurobipy.GRB.MAXIMIZE)

# constraints
# split into slices
model.addConstrs(slice.sum(e, '*') == capacity[e] for e in range(num_edges))
# utility
model.addConstrs(utility[e, q] == slice[e, q] / flow[q] for q in range(num_quest) for e in sp[q])
# set unused edges to 10
model.addConstrs(utility[e, q] == 10 for q in range(num_quest) for e in range(num_edges) if e not in sp[q])
model.addConstrs(q_util[q] == gurobipy.min_(utility.select('*', q)) for q in range(num_quest))
'''

'''
# delay mode
slice = model.addVars(range(num_edges),range(num_quest),name='slice',lb=1, vtype=gurobipy.GRB.INTEGER)
delay = model.addVars(range(num_edges),range(num_quest),name='delay')
q_delay= model.addVars(range(num_quest),name='delay of a quest')
# delay objective
model.setObjective(gurobipy.quicksum(q_delay), gurobipy.GRB.MINIMIZE)

model.update()

# constraints
model.addConstrs(slice.sum(e, '*') == capacity[e] for e in range(num_edges))
model.addConstrs(q_delay[q] == gurobipy.max_(delay.select('*',q)) for q in range(num_quest) for e in range(num_edges))

# model.addConstrs(ratio.sum(e,'*') == 1 for e in range(num_edges))
# model.addConstrs(sum(flow[q] for e in range(num_edges) for q in range(num_quest) if e in sp[q]) <=capacity[e] )


# for q in range(num_quest):
#     model.addConstr(delay[q] == max(compute_delay(flow[q], capacity[e]*ratio[e,q]) for e in sp[q]))


for q in range(num_quest):
    for e in sp[q]:
        if 3 * flow[q] <= slice[e, q].lb:
            model.addConstr(delay[e,q]==flow[q])
        if 3 * flow[q] > slice[e, q].ub and 3 * flow[q] <= 2 * slice[e, q].lb:
            model.addConstr(delay[e,q]==3*flow[q]-2/3*slice[e,q] )
        if 3 * flow[q] > 2 * slice[e, q].ub and 10 * flow[q] <= 9 * slice[e, q].lb:
            model.addConstr(delay[e,q]==10*flow[q]-16/3*slice[e,q])
        if 10 * flow[q] > 9 * slice[e, q].ub and flow[q] <= slice[e, q].lb:
            model.addConstr(delay[e,q]==70*flow[q]-178/3*slice[e,q])
        if flow[q] > slice[e, q].ub and 10 * flow[q] <= 11 * slice[e, q].lb:
            model.addConstr(delay[e,q]==500*flow[q]-1468/3*slice[e,q])
        # if 10 * flow[q] > 11 * slice[e, q].ub:
        else:
            model.addConstr(delay[e,q]==5000*flow[q]-16318/3*slice[e,q] )
model.addConstrs(delay[e,q]==1 for q in range(num_quest) for e in range(num_edges) if e not in sp[q])
'''

# SOS1 delay model
slices = model.addVars(range(num_edges), range(num_quest), lb=0, name='slice')
delay = model.addVars(range(num_edges), range(num_quest), name='delay')

# sos1 parameters
w = model.addVars(range(num_edges), range(num_quest), range(7), lb=0, ub=1, name='omega')
z = model.addVars(range(num_edges), range(num_quest), range(6), vtype=gurobipy.GRB.BINARY, name='z')

# weight of b
b = [0, 10 / 11, 1, 10 / 9, 3 / 2, 3, 100]
f_b = [5000, 500 - 14680 / 33, 70 - 178 / 3, 10 - 160 / 27, 2, 1, 1]

# obj
model.setObjective(gurobipy.quicksum(delay), gurobipy.GRB.MINIMIZE)

model.update()

# Constraints
model.addConstrs(slices.sum(e, '*') == capacity[e] for e in range(num_edges))
model.addConstrs(w.sum(e, q, '*') == 1 for e in range(num_edges) for q in range(num_quest))
model.addConstrs(z.sum(e, q, '*') == 1 for e in range(num_edges) for q in range(num_quest))


for e in range(num_edges):
    for q in range(num_quest):
        if net_flow[e, q]!=0:
            model.addConstr(slices[e, q] == gurobipy.quicksum([b[i] * net_flow[e, q] * w[e, q, i] for i in range(7)]))
            model.addConstr(delay[e, q] == gurobipy.quicksum([f_b[i] * net_flow[e, q] * w[e, q, i] for i in range(7)]))
            model.addConstr(w[e, q, 0] <= z[e, q, 0])
            model.addConstr(w[e, q, 6] <= z[e, q, 5])
            model.addConstrs(w[e, q, i] <= z[e, q, i - 1] + z[e, q, i] for i in range(1, 5))

model.optimize()
opt_time=time.time()
print("Obj:", model.objVal)
res=np.zeros([num_edges,num_quest])
for e in range(num_edges):
    for q in range(num_quest):
        try:
            v=model.getVarByName(f"slice[{e},{q}]")
            if v.x != 0:
                res[e,q]=v.x
        except :
            print(gurobipy.GRB.ERROR_NOT_IN_MODEL)

res_df = pd.DataFrame(res)
print(res_df)
print(f"Time for modelling optimization: {opt_time-pre_end}")