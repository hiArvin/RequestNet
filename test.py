# # from utils import random_graph_input
# # graph,support,f_bandwidth = random_graph_input(num_nodes=10)
# # print(support)
# from topology import Topology
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
#
# d = np.load('logs/Q10/P7-R6/delay.npy')
# print(d)
# plt.figure()
# # ax1 = plt.subplot(121)
# plt.title('Delay')
# x_axis = list(range(20))
# # plt.plot(x_axis, d[0,:], color='red', label='Predictor delay')
# # plt.plot(x_axis, d[1,:], color='green', label='SP selection delay')
# plt.plot(x_axis, d[0,:]/d[1,:], color='red', label='Predictor/SP Ratio')
# # plt.plot(x_axis, d[1,:], color='green', label='SP selection Delay')
#
# plt.legend() # 显示图例
# plt.xlabel('Iteration times')
# plt.ylabel('Delay')
# # ax2 = plt.subplot(122)
# # # ax2.title('Number of Successful Transmitted Flow')
# # ax2.plot(x_axis, s[0,:], color='red', label='Predictor successful number')
# # ax2.plot(x_axis, s[1,:], color='green', label='SP selection successful number')
# # plt.savefig('logs/comp.jpg')
# plt.show()
import networkx as nx
dataset_path = "dataset/"
graph_path = dataset_path + "Aarnet.graphml"
graph = nx.read_graphml(graph_path, node_type=int)

print(nx.number_of_nodes(graph))