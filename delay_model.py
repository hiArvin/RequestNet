import pandas as pd
import numpy as np
import networkx as nx
import gurobipy as gb

from utils import k_shortest_paths, random_capacity
from topology import Topology


def delay_solver(num_edges, num_quests, num_paths, sp, capacity):
    model=gb.Model()
    select_paths = model.addVars(range(num_quests), range(num_paths), lb=0, vtype=gb.GRB.BINARY,
                                 name='select_paths')
    # select_flow = model.addVars(range(num_quests), vtype=gb.GRB.BINARY, name='select_flow')
    edge_flow = model.addVars(range(num_edges), range(num_quests), lb=0, ub=1000, vtype=gb.GRB.INTEGER,
                              name='flow')
    flow = model.addVars(range(num_edges))
    # utility = model.addVars(range(num_edges),lb=0,ub=1, vtype=gb.GRB.CONTINUOUS, name='occupy')
    delay = model.addVars(range(num_edges), vtype=gb.GRB.CONTINUOUS, name='delay')
    obj = model.addVar(vtype=gb.GRB.CONTINUOUS, name='obj')
    model.update()

    # objective
    model.setObjective(obj, gb.GRB.MINIMIZE)
    # constraints
    model.addConstrs(select_paths.sum(q, '*') == 1 for q in range(num_quests))
    model.addConstrs(
        flow[e] == edge_flow.sum(e, '*')
        for e in range(num_edges))
    model.addConstrs(flow[e] <= capacity[e] for e in range(num_edges))
    # model.addConstrs(utility[e] == flow[e]/ capacity[e] for e in range(num_edges))
    for e in range(num_edges):
        model.addGenConstrPWL(flow[e], delay[e],
                              [0, 1 / 3 * capacity[e], 2 / 3 * capacity[e], 9 / 10 * capacity[e], 1 * capacity[e]],
                              [0, 1 / 3 * capacity[e], 4 / 3 * capacity[e], 11 / 3 * capacity[e],
                               78 / 3 * capacity[e]])
    model.addConstr(obj == gb.quicksum(delay))
    # model.addConstr(obj == gb.quicksum(select_flow) * num_edges - gb.quicksum(util))

    # conditional constraints
    model.addConstrs(edge_flow[e, q] == gb.quicksum(select_paths[q, p] * sp[q, p, e] for p in range(num_paths))
                     for q in range(num_quests) for e in range(num_edges))

    model.optimize()

    # get decision results
    res_selct = np.zeros([num_quests, num_paths], dtype=np.int64)

    for q in range(num_quests):
        for p in range(num_paths):
            var = model.getVarByName(f"select_paths[{q},{p}]")
            if var.X == 1:
                res_selct[q, p] = 1
                break

    delay = np.zeros([num_edges], dtype=np.float32)
    for e in range(num_edges):
        var = model.getVarByName(f"delay[{e}]")
        delay[e] = var.X

    # sel_flow = np.zeros([num_quests], dtype=np.int64)
    # for q in range(num_quests):
    #     var = model.getVarByName(f"select_flow[{q}]")
    #     sel_flow[q] = var.X
    # print(delay)
    # print(res_selct)
    # print(sel_flow)


    return res_selct


def gen_label(graph, flow, capacity, num_paths, num_quests=1, max_cap=10, print_info=False):
    num_edges = nx.number_of_edges(graph)
    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])

    shortest_path = []
    for q in range(num_quests):
        k_sp = k_shortest_paths(graph, source=flow[q][0], target=flow[q][1], k=num_paths)
        for i in range(num_paths - len(k_sp)):
            k_sp.append(k_sp[0])
        shortest_path.append(k_sp)
    # print('shortest paths (nodes): ',shortest_path)
    sp = np.zeros([num_quests, num_paths, num_edges], dtype=np.int)
    sp_f = np.zeros([num_quests * num_paths, num_edges])
    paths = []
    idx_list = []
    seqs = []
    for q in range(num_quests):
        for p in range(num_paths):
            path = shortest_path[q][p]
            pp = []
            for i in range(len(path) - 1):
                idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                            ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
                pp.append(idx)
                sp[q, p, idx] = flow[q][2]
                sp_f[q * num_paths + p] = flow[q][2]
            paths.append(pp)
    for i in range(len(paths)):
        seqs.append(list(range(len(paths[i]))))
        for j in range(len(paths[i])):
            idx_list.append(i)
    # exclude 0 as fraction
    for i in range(len(capacity)):
        if capacity[i] == 0:
            capacity[i] = 1

    delay_solver(num_edges, num_quests, num_paths, sp, capacity)


if __name__ == "__main__":
    topo = Topology(num_core=1, num_converge=1, num_access=1)
    graph = topo.graph
    NUM_EDGES = nx.number_of_edges(graph)
    NUM_NODES = nx.number_of_nodes(graph)
    NUM_PATHS = 5  # at least 4
    NUM_QUESTS = 8
    Min_Cap = 5  # *100
    Max_Cap = 10  # *100

    bandwidth = random_capacity(Min_Cap, Max_Cap, NUM_EDGES)
    flow_generator = topo.gen_flows(size_percent=0.05)
    flow = []
    for i in range(NUM_QUESTS):
        flow.append(next(flow_generator))
    gen_label(topo.graph, flow, bandwidth, num_paths=NUM_PATHS, num_quests=NUM_QUESTS, max_cap=Max_Cap)
