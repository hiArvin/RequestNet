import os
import random
import networkx as nx
import numpy as np
from label import gen_paths, solver

# get paths
all_data_path=[]
for f in os.listdir('dataset/'):
    if '.graphml' in f:
        all_data_path.append('dataset/'+f)
print(all_data_path)
# random choose a graph
graph = nx.read_graphml(random.choice(all_data_path),node_type=int,edge_key_type=int,force_multigraph=False)

print(graph.nodes,graph.edges)

n_edges = nx.number_of_edges(graph)
n_nodes = nx.number_of_nodes(graph)
bd = np.ones(n_edges)*1000

# random traffic
traffic = np.random.randint(0,90,n_edges)*10

# flows
n_flows = 10
