from topology import Topology
import networkx as nx
import pickle
from utils import *
import numpy as np
from collections import defaultdict
import pprint





if __name__ == "__main__":
    topo = Topology(num_core=4)
    graph = topo.graph
    num_edges = nx.number_of_edges(graph)
    num_nodes = nx.number_of_nodes(graph)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    # sp=nx.all_shortest_paths(graph,source=0,target=4)
    # print(next(sp))

    sp_dict = defaultdict(defaultdict)
    num_paths = 3
    for i in range(num_nodes):
        print(i)
        temp = {}
        for j in range(num_nodes):
            if nx.has_path(graph,source=i,target=j):
                k_sp = k_shortest_paths(graph, i, j, k=num_paths)
                for k in range(num_paths - len(k_sp)):
                    k_sp.append(k_sp[0])
                    temp[j] = k_sp
                print(f"({i},{j}) done.")
        sp_dict[i] = temp

    save_obj(sp_dict, 'test')
    dd = load_obj('test')
    print(dd[1][100])