# from utils import random_graph_input
# graph,support,f_bandwidth = random_graph_input(num_nodes=10)
# print(support)
from topology import Topology
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

d = np.load('logs/0608-1920-Q20-P8-R5/delay.npy')
print(d)
plt.figure()
# ax1 = plt.subplot(121)
plt.title('Delay')
x_axis = list(range(d.shape[1]))
# plt.plot(x_axis, d[0,:], color='red', label='Predictor delay')
# plt.plot(x_axis, d[1,:], color='green', label='SP selection delay')
plt.plot(x_axis, d[0,:]/d[2,:], color='red', label='Predictor/OPT Ratio')
plt.plot(x_axis, d[1,:]/d[2,:], color='green', label='SP/OPT Ratio')

plt.ylim((0.9,1.5))
# plt.plot(x_axis, d[1,:], color='green', label='SP selection Delay')

plt.legend() # 显示图例
plt.xlabel('Iteration times')
plt.ylabel('Delay')
# ax2 = plt.subplot(122)
# # ax2.title('Number of Successful Transmitted Flow')
# ax2.plot(x_axis, s[0,:], color='red', label='Predictor successful number')
# ax2.plot(x_axis, s[1,:], color='green', label='SP selection successful number')
# plt.savefig('logs/comp.jpg')
plt.show()

