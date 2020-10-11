import networkx as nx
import numpy as np


class Flow:
    def __init__(self, num: int, source: int, destination: int, size: float):
        self.num_of_flow = num
        self.source = source
        self.destination = destination
        self.size = size
        self.route_state = False
        self.matrix = None
        self.belong_slice = None
        self.flow_bfs_nodes = None
        self.route_nodes = []
        self.route_links = []

    def route_flow(self, topology, weight='weight'):
        assert not self.route_state
        self.route_nodes.clear()
        self.route_links.clear()
        try:
            temp_sp = nx.shortest_path(topology, self.source, self.destination, weight=weight)
        except nx.NetworkXNoPath:
            self.route_state = False
            return
        unable_trans = self._reduce_band_in_top(topology, temp_sp)
        temp_band = []
        while len(unable_trans) > 0:
            for a, b, attrs in unable_trans:
                temp_band.append(tuple([a, b, topology[a][b]]))
                topology.remove_edge(a, b)
            try:
                temp_sp = nx.shortest_path(topology, self.source, self.destination, weight=weight)
            except nx.NetworkXNoPath:
                self.route_state = False
                for a, b, attrs in temp_band:
                    topology.add_edge(a, b)
                    for k, v in attrs.items():
                        topology[a][b][k] = v
                return
            unable_trans = self._reduce_band_in_top(topology, temp_sp)
        for a, b, attrs in temp_band:
            topology.add_edge(a, b)
            for k, v in attrs.items():
                topology[a][b][k] = v
        self.route_nodes = [x for x in temp_sp]
        self.route_state = True
        self._gen_paths()

    def _gen_paths(self):
        links = list(zip(self.route_nodes[:-1], self.route_nodes[1:]))
        self.route_links = links

    def _reduce_band_in_top(self, topology, tunnels):  # 返回最短路径中容量不够的链接
        paths = list(zip(tunnels[:-1:], tunnels[1:]))
        unable_to_trans = []
        sign = True
        for a, b in paths:
            if topology[a][b]['bandwidth'] < self.size:
                sign = False
                unable_to_trans.append(tuple([a, b, topology[a][b]]))
        if not sign:
            return unable_to_trans
        for a, b in paths:
            topology[a][b]['bandwidth'] -= self.size
        return unable_to_trans

    def disconnect_flow(self, topology):
        if not self.route_state:
            return
        self.route_state = False
        for a, b in self.route_links:
                topology[a][b]['bandwidth'] += self.size
        self.route_nodes.clear()
        self.route_links.clear()

    def gen_flow_bfs_nodes(self, topology):
        """
        apply the feature to the self.flow_bfs_nodes, it will be stack with the node feature in each slice

        Parameters
        ----------
        topology: the topology to generate the bfs tree feature of source and destination

        Returns
        -------
        void: apply the feature to the self.flow_bfs_nodes, its an ndarray.
        """
        bfs_nodes_feature = np.zeros((len(topology.nodes),), dtype=np.float)
        nodes = [self.source] + [v for u, v in nx.bfs_edges(topology, self.source, depth_limit=10)] + \
                [self.destination] + [v for u, v in nx.bfs_edges(topology, self.destination, depth_limit=10)]
        for i in nodes:
            bfs_nodes_feature[i] += 1
        self.flow_bfs_nodes = bfs_nodes_feature

    def test_flow_route(self, topology):
        """
        test if this can be route in specific slice

        Parameters
        ----------
        topology: the top to test

        Returns
        -------
        signal: bool

        """
        self.route_flow(topology)
        signal = self.route_state
        self.disconnect_flow(topology)
        return signal
