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


def random_quests(num_nodes, num_quests, num_paths, num_edges, edges, graph):
    quest = np.random.randint(0, num_nodes, [num_quests, 2])
    for i in range(num_quests):
        while not nx.has_path(graph, quest[i, 0], quest[i, 1]):
            quest[i, 1] = random.randint(0, num_nodes - 1)

    shortest_path = []
    for q in range(num_quests):
        k_sp = k_shortest_paths(graph, source=quest[q, 0], target=quest[q, 1], k=num_paths)
        for i in range(num_paths - len(k_sp)):
            k_sp.append(k_sp[0])
        shortest_path.append(k_sp)

    # transform shortest path into edge index

    flow = np.random.randint(20, 100, [num_quests], dtype=np.int)
    # print(flow)
    sp = np.zeros([num_quests, num_paths, num_edges], dtype=np.int)
    for q in range(num_quests):
        for p in range(num_paths):
            path = shortest_path[q][p]
            for i in range(len(path) - 1):
                idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                            ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
                sp[q, p, idx] = flow[q]
    return sp


def solve_slice(num_edges, num_quests, num_paths, sp, capacity):
    total_start = time.time()
    model = gb.Model()
    slices = model.addVars(range(num_edges), range(num_quests), lb=0, ub=1000, vtype=gb.GRB.INTEGER, name='slice')
    select_paths = model.addVars(range(num_quests), range(num_paths), lb=0, vtype=gb.GRB.BINARY, name='select_paths')
    edge_flow = model.addVars(range(num_edges), range(num_quests), lb=0, ub=1000, vtype=gb.GRB.INTEGER, name='flow')

    # diff = model.addVars(range(num_edges), range(num_quests), name='diff')
    success = model.addVars(range(num_edges), range(num_quests), vtype=gb.GRB.BINARY, name='success')
    scc = model.addVars(range(num_quests), vtype=gb.GRB.BINARY, name='scc')

    model.update()

    # objective
    model.setObjective(gb.quicksum(scc), gb.GRB.MAXIMIZE)

    # constraints
    model.addConstrs(select_paths.sum(q, '*') == 1 for q in range(num_quests))
    model.addConstrs(slices.sum(e, '*') == capacity[e] for e in range(num_edges))
    model.addConstrs(scc[q] == gb.min_(success.select('*', q)) for q in range(num_quests))
    model.addConstrs(edge_flow[e, q] == gb.quicksum(select_paths[q, p] * sp[q, p, e] for p in range(num_paths))
                     for q in range(num_quests) for e in range(num_edges))

    # conditional constraints
    model.addConstrs((success[e, q]) * (slices[e, q].ub - edge_flow[e, q].lb) >= slices[e, q] - edge_flow[e, q]
                     for e in range(num_edges) for q in range(num_quests))
    model.addConstrs((success[e, q])*1+(1 - success[e, q]) * (slices[e, q].lb - edge_flow[e, q].ub) <= slices[e, q] - edge_flow[e, q]
                     for e in range(num_edges) for q in range(num_quests))

    model.optimize()
    failure_rate = (num_quests - model.objVal) / num_quests
    fails = num_quests - model.objVal

    res_scc = np.ones([num_quests], dtype=np.int)
    res_flow = np.zeros([num_edges, num_quests], dtype=np.int)
    for q in range(num_quests):
        v = model.getVarByName(f"scc[{q}]")
        if v.x == 0:
            res_scc[q] = 0

    for e in range(num_edges):
        for q in range(num_quests):
            vf = model.getVarByName(f"flow[{e},{q}]")
            if vf.x != 0:
                res_flow[e, q] = vf.x
    occupy = np.multiply(res_flow, res_scc)
    modelling_time = time.time() - total_start

    print('=======================')
    print('Number of quests:', num_quests)
    print("Failure rate:", failure_rate)
    print("Runtime", model.runtime)
    print("Runtime", modelling_time)

    return model.runtime, modelling_time, failure_rate, occupy, fails


def modify_capacity(capacity, flow):
    f = np.sum(flow, axis=1)
    capacity = capacity - f
    return capacity


if __name__ == "__main__":
    topo = Topology(num_core=4)
    graph = topo.graph
    num_edges = nx.number_of_edges(graph)
    num_nodes = nx.number_of_nodes(graph)
    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])
    capacity = np.repeat([1000], num_edges)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print('=====End of Constructing Graph======')
    num_quests = 1
    num_paths = 6
    res_opt_time = []
    res_model_time = []
    res_f = []
    epochs = 1500
    for epoch in range(epochs):
        sp = random_quests(num_nodes, num_quests, num_paths, num_edges, edges, graph)
        opt_time, mod_time, f_rate, s_flow, fails = solve_slice(num_edges, num_quests, num_paths, sp, capacity)
        capacity = modify_capacity(capacity, s_flow)

        res_opt_time.append(opt_time)
        res_model_time.append(mod_time)
        res_f.append(fails)

    # save to file
    filename = open('res.txt', 'w')
    for value in res_opt_time:
        filename.write(str(value) + '\t')
    filename.write('\n')
    for value in res_model_time:
        filename.write(str(value) + '\t')
    filename.write('\n')
    for value in res_f:
        filename.write(str(value) + '\t')
    filename.close()
    print("Total optimizing time: ", sum(res_opt_time))
    print("Total modelling time: ", sum(res_model_time))
    print("Total failure rate: ", sum(res_f) / 1500)

    print("Rest capacity:", capacity)
