import os
import networkx as nx
file_names = os.listdir('dataset/')
for f in file_names:
    if f.split('.')[1]=='graphml':
        graph= nx.read_graphml('dataset/'+f,node_type=int)
        print(f,'\t',nx.number_of_nodes(graph),'\t',nx.number_of_edges(graph))
