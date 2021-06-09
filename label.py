import gurobipy as gb
import networkx as nx
import numpy as np
import pandas as pd
import time
import random
from iteration_utilities import deepflatten

from utils import normalization, k_shortest_paths,cal_total_delay


def solve_slice(num_edges, num_quests, num_paths, sp, capacity):
    with gb.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gb.Model(env=env) as model:
            select_paths = model.addVars(range(num_quests), range(num_paths), lb=0, vtype=gb.GRB.BINARY,
                                         name='select_paths')
            select_flow = model.addVars(range(num_quests), vtype=gb.GRB.BINARY, name='select_flow')
            edge_flow = model.addVars(range(num_edges), range(num_quests), lb=0, ub=1000, vtype=gb.GRB.INTEGER,
                                      name='flow')
            flow = model.addVars(range(num_edges))
            util = model.addVars(range(num_edges), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name='utility')
            obj = model.addVar(vtype=gb.GRB.CONTINUOUS, name='obj')
            model.update()

            # objective
            model.setObjective(obj, gb.GRB.MAXIMIZE)

            # constraints
            model.addConstrs(select_paths.sum(q, '*') == 1 for q in range(num_quests))
            model.addConstrs(
                flow[e] == gb.quicksum(edge_flow[e, q] * select_flow[q] for q in range(num_quests))
                for e in range(num_edges))
            model.addConstrs(flow[e] <= capacity[e] for e in range(num_edges))
            model.addConstrs(util[e] == flow[e] / capacity[e] for e in range(num_edges))
            model.addConstr(obj == gb.quicksum(select_flow) * num_edges - gb.quicksum(util))

            # conditional constraints
            model.addConstrs(edge_flow[e, q] == gb.quicksum(select_paths[q, p] * sp[q, p, e] for p in range(num_paths))
                             for q in range(num_quests) for e in range(num_edges))

            model.optimize()

            # get decision results
            res_selct = np.zeros([num_quests, num_paths], dtype=np.int)

            for q in range(num_quests):
                for p in range(num_paths):
                    var = model.getVarByName(f"select_paths[{q},{p}]")
                    if var.X == 1:
                        res_selct[q, p] = 1
                        break

            utility = np.zeros([num_edges], dtype=np.float)
            for e in range(num_edges):
                var = model.getVarByName(f"utility[{e}]")
                utility[e] = var.X

            sel_flow = np.zeros([num_quests], dtype=np.int)
            for q in range(num_quests):
                var = model.getVarByName(f"select_flow[{q}]")
                sel_flow[q] = var.X
            # print(sel_flow)
            # print(utility)

    return res_selct, sel_flow


def delay_solver(num_edges, num_quests, num_paths, sp, opy, capacity):
    # mask = np.where(sp>0,1,0)
    with gb.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gb.Model(env=env) as model:
            select_paths = model.addVars(range(num_quests), range(num_paths), lb=0, vtype=gb.GRB.BINARY,
                                         name='select_paths')
            # select_flow = model.addVars(range(num_quests), vtype=gb.GRB.BINARY, name='select_flow')
            edge_flow = model.addVars(range(num_edges), range(num_quests), lb=0, ub=1000, vtype=gb.GRB.INTEGER,
                                      name='flow')
            flow = model.addVars(range(num_edges))
            f_plus_o = model.addVars(range(num_edges))
            # utility = model.addVars(range(num_edges),lb=0,ub=1, vtype=gb.GRB.CONTINUOUS, name='occupy')
            delay = model.addVars(range(num_edges), vtype=gb.GRB.CONTINUOUS, name='delay')
            # mask_q = model.addVars(range(num_quests),range(num_edges), vtype=gb.GRB.BINARY,name='flow_mask')
            # delay_qe = model.addVars(range(num_quests),range(num_edges),vtype=gb.GRB.CONTINUOUS)
            # delay_q = model.addVars(range(num_quests),vtype=gb.GRB.CONTINUOUS,name='flow_delay')
            obj = model.addVar(vtype=gb.GRB.CONTINUOUS, name='obj')
            model.update()

            # objective
            model.setObjective(obj, gb.GRB.MINIMIZE)
            # constraints
            model.addConstrs(select_paths.sum(q, '*') == 1 for q in range(num_quests))
            model.addConstrs(
                flow[e] == edge_flow.sum(e, '*')
                for e in range(num_edges))
            model.addConstrs(f_plus_o[e]==flow[e]+opy[e] for e in range(num_edges))
            # model.addConstrs(flow[e] <= capacity[e] for e in range(num_edges))
            # model.addConstrs(utility[e] == flow[e]/ capacity[e] for e in range(num_edges))
            for e in range(num_edges):
                model.addGenConstrPWL(f_plus_o[e], delay[e],
                                      [0, 1 / 3 * capacity[e], 2 / 3 * capacity[e], 9 / 10 * capacity[e],
                                       1 * capacity[e], 11 / 10 * capacity[e], 3 / 2 * capacity[e]],
                                      [0, 1 / 3 * capacity[e], 4 / 3 * capacity[e], 11 / 3 * capacity[e],
                                       78 / 3 * capacity[e], 182 / 3 * capacity[e], 6182 / 3 * capacity[e]])

            '''
            constraints for delay of flow
            # model.addConstrs(mask_q[q,e]==gb.quicksum(mask[q,p,e]*select_paths[q,p] for p in range(num_paths))
            #                  for q in range(num_quests) for e in range(num_edges))
            # model.addConstrs(delay_qe[q,e] == delay[e] * mask_q[q,e] for e in range(num_edges) for q in range(num_quests))
            # model.addConstrs(delay_q[q] == gb.max_(delay_qe[q,e] for e in range(num_edges))
            #                  for q in range(num_quests))
            '''
            model.addConstr(obj == gb.quicksum(delay))

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
            return res_selct



def gen_label(graph, flow, ocupy, init_bd, num_paths, target, num_quests=1):
    num_edges = nx.number_of_edges(graph)
    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])

    shortest_path = []

    for q in range(num_quests):
        k_sp = k_shortest_paths(graph, source=flow[q][0], target=flow[q][1], k=num_paths)
        for i in range(num_paths - len(k_sp)):
            k_sp.append(k_sp[0])
        random.shuffle(k_sp)
        shortest_path.append(k_sp)

    # print('shortest paths (nodes): ',shortest_path)
    sp = np.zeros([num_quests, num_paths, num_edges], dtype=np.int32)
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

    if target == 'utility':
        capacity = init_bd - ocupy
        # exclude 0 as fraction
        capacity = np.where(capacity <= 0, 1, capacity)
        res_selct, sel_flow = solve_slice(num_edges, num_quests, num_paths, sp, capacity)
    elif target == 'delay':
        res_selct = delay_solver(num_edges, num_quests, num_paths, sp, ocupy, init_bd)
    else:
        res_selct = np.zeros([num_quests, num_paths], dtype=np.int64)
    # convert to label
    label = np.zeros([num_quests, num_paths], dtype=np.int64)
    for q in range(num_quests):
        for p in range(num_paths):
            if res_selct[q, p] == 1:
                label[q, p] = 1
                break
    paths = list(deepflatten(paths))
    seqs = list(deepflatten(seqs))

    # # normalize sp
    # sp_f = sp_f / (max_cap*100)
    return paths, idx_list, seqs, label, sp_f, shortest_path


def gen_paths(graph, flow, num_paths, num_quests=1):
    num_edges = nx.number_of_edges(graph)
    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])

    shortest_path = []
    for q in range(num_quests):
        k_sp = k_shortest_paths(graph, source=flow[q][0], target=flow[q][1], k=num_paths)
        for i in range(num_paths - len(k_sp)):
            k_sp.append(k_sp[0])
        random.shuffle(k_sp)
        shortest_path.append(k_sp)
    # print('shortest paths (nodes): ',shortest_path)
    sp = np.zeros([num_quests, num_paths, num_edges], dtype=np.int)

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
            paths.append(pp)
    for i in range(len(paths)):
        seqs.append(list(range(len(paths[i]))))
        for j in range(len(paths[i])):
            idx_list.append(i)

    paths = list(deepflatten(paths))
    seqs = list(deepflatten(seqs))
    sp = sp.reshape([num_quests * num_paths, num_edges])

    return paths, idx_list, seqs, sp, shortest_path


def solver(graph, shortest_path, flow, occupy, capacity, num_paths, num_quests=1, print_info=False):
    num_edges = nx.number_of_edges(graph)
    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])
    sp = np.zeros([num_quests, num_paths, num_edges], dtype=np.int16)
    for q in range(num_quests):
        for p in range(num_paths):
            path = shortest_path[q][p]
            pp = []
            for i in range(len(path) - 1):
                idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                            ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
                pp.append(idx)
                sp[q, p, idx] = flow[q][2]
    # # exclude 0 as fraction
    # capacity = np.where(capacity <= 0, 1, capacity)
    t = time.time()
    label = delay_solver(num_edges, num_quests, num_paths, sp, occupy, capacity)
    t_ = time.time()
    c = np.nanargmax(label, axis=1)

    # update
    success = num_quests

    for q in range(num_quests):
        tmp = capacity - occupy
        tmp = np.where(tmp <= 0, 0, tmp)
        tmp = tmp - sp[q, c[q], :]
        # print(tmp)
        if np.any(tmp < 0):
            success -= 1
        occupy = occupy + sp[q, c[q], :]
    return c, occupy, success, t_ - t
