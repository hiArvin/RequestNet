from utils import *
import random
from iteration_utilities import deepflatten
from label import delay_solver


class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.random_bandwidth = args.random_bandwidth
        self.num_paths = args.num_paths
        self.num_flows = args.num_flows

        self.node_graph = self.get_topozoo_graph(args.training_graph)
        self.num_nodes = nx.number_of_edges(self.node_graph)
        self.num_edges = nx.number_of_nodes(self.node_graph)

        self.edge_graph = self.nodeGraph_2_edgeGraph()

        self.init_bandwidth()
        self.shortest_paths = self.gen_paths()

    def get_topozoo_graph(self, graph_name):
        dataset_path = "dataset/"
        graph_path = dataset_path + graph_name
        graph = nx.read_graphml(graph_path, node_type=int)
        return graph

    def nodeGraph_2_edgeGraph(self):
        '''
        返回 edge graph 和 对应的边属性
        :param graph:
        :return:
        '''
        num_nodes = self.node_graph.number_of_nodes()
        num_edges = self.node_graph.number_of_edges()
        e_graph = nx.empty_graph(num_edges)
        edges = list(nx.edges(self.node_graph))
        for i in range(num_nodes):
            idx = []
            for nei in nx.neighbors(self.node_graph, i):
                if nei > i:
                    idx.append(edges.index((i, nei)))
                else:
                    idx.append(edges.index((nei, i)))
            for j in idx:
                for k in idx:
                    if j != k:
                        e_graph.add_edge(j, k)
        return e_graph

    def get_laplacian_matrix(self):
        sup = nx.normalized_laplacian_matrix(self.edge_graph).todense()
        return sup

    def init_bandwidth(self):
        if not self.random_bandwidth:
            bandwidth = np.ones(self.num_edges) * 10000
        else:
            bandwidth = np.random.randint(1, 10, self.num_edges) * 10000
        self.bandwidth = bandwidth

    def gen_paths(self):
        # 字典套字典形式的前n个最短路径，不足的用最短路径补齐
        shortest_path = {}
        for src in range(self.num_nodes):
            shortest_path[src] = {}
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                k_sp = k_shortest_paths(self.node_graph, source=src, target=dst, k=self.num_paths)
                for i in range(self.num_paths - len(k_sp)):
                    k_sp.append(k_sp[0])
                random.shuffle(k_sp)
                shortest_path[src][dst]=k_sp
        return shortest_path

    def generate_flows(self,high_percent=0.05, low_percent=0.001):
        flows = []
        for f in range(self.num_flows):
            nodes = list(self.node_graph.nodes)
            s, d = np.random.choice(nodes,2)
            flow_size = np.random.randint(low_percent*self.bandwidth,high_percent*self.bandwidth,1)
            flows.append([s,d,int(flow_size)])
        return flows

    def flow_to_numpy(self,flows):
        # print('shortest paths (nodes): ',shortest_path)
        sp = np.zeros([len(flows),self.num_paths, self.num_edges], dtype=np.int)
        edges = pd.DataFrame(nx.edges(self.node_graph, nbunch=None), columns=['src', 'dst'])
        for i in range(len(flows)):
            src, dst, flow_size = flows[i]
            for p in range(self.num_paths):
                simple_path = self.shortest_paths[src][dst][p]
                idx = edges[((edges.src == simple_path[i]) & (edges.dst == simple_path[i + 1])) |
                            ((edges.dst == simple_path[i]) & (edges.src == simple_path[i + 1]))].index[0]
                for i in range(len(simple_path) - 1):
                    sp[i, p, idx] = flow_size

        sp = sp.reshape([len(flows) * self.num_paths, self.num_edges])
        return sp

    def generate_seqs(self,flows):
        paths = []
        idx_list = []
        seqs = []
        edges = pd.DataFrame(nx.edges(self.node_graph, nbunch=None), columns=['src', 'dst'])
        for i in range(len(flows)):
            src,dst,flow_size = flows[i]
            for p in range(self.num_paths):
                pp = []
                path = self.shortest_paths[src][dst][[]]
                for i in range(len(path) - 1):
                    idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                                ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
                    pp.append(idx)
                paths.append(pp)
        for i in range(len(paths)):
            seqs.append(list(range(len(paths[i]))))
            for j in range(len(paths[i])):
                idx_list.append(i)
        paths = list(deepflatten(paths))
        seqs = list(deepflatten(seqs))
        return paths, idx_list, seqs

    def generate_delay_label(self,sp,occupy, init_bd):
        res_selct, delay = delay_solver(self.num_edges, self.num_flows, self.num_paths, sp, occupy, init_bd)



if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.training_graph = "Aarnet.graphml"
            self.random_bandwidth = False
            self.num_paths = 5
            self.num_flows = 10


    args = Args()
    dp = DataProcessor(args)
    dp.init_bandwidth()
    flows = dp.generate_flows(20)
    print(flows)
