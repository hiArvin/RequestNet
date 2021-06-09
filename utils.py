import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import pandas as pd
import networkx as nx
import numpy as np
import pickle
from topology import Topology


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, paths, index, sequences, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['paths']: paths})
    feed_dict.update({placeholders['index']: index})
    feed_dict.update({placeholders['sequences']: sequences})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def edge_convert_node(graph):
    num_edges = nx.number_of_edges(graph)
    adj = np.zeros([num_edges, num_edges], dtype=np.int)

    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])


def k_shortest_paths(graph, source, target, k):
    all_path = nx.all_simple_paths(graph, source, target)
    paths = []
    count = 0
    for i in all_path:
        count += 1
        paths.append(i)
        if count % k == 0:
            break
    return paths


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def update_capacity(capacity, occupy):
    f = np.sum(occupy, axis=1)
    capacity = capacity - f
    flag = True
    for i in capacity:
        if i < 0:
            flag = False
    return capacity, flag


def normalization(data):
    return data / np.linalg.norm(data)


def normalize_features(feature, input_dim, max_cap=None):
    dim, length = feature.shape
    f = np.zeros([input_dim, length])
    for d in range(dim):
        f[d, :] = normalization(feature[d, :])
    return f


def random_capacity(a, b, num_edges):
    cap = np.random.randint(a * 10, b * 10, num_edges)
    return cap * 10


def nodeGraph_to_edgeGraph(graph, support=False):
    num_edges = graph.number_of_edges()
    edges_attr = {}
    idx = 0
    tran_dict = {'core layer': 1, 'converge layer': 2, 'access layer': 3, 'metro cross': 4}
    for u, v in graph.edges:
        attr = graph.get_edge_data(u, v)
        edges_attr[idx] = {'u': u, 'v': v, 'layer': tran_dict[attr['layer']], 'bandwidth': attr['bandwidth']}
        idx += 1
    edges_df = pd.DataFrame.from_dict(edges_attr)

    adj = np.zeros([num_edges, num_edges], dtype=np.int)
    for i in range(num_edges - 1):
        for j in range(i + 1, num_edges):
            if edges_attr[i]['u'] == edges_attr[i]['u'] \
                    or edges_attr[i]['v'] == edges_attr[i]['v'] \
                    or edges_attr[i]['u'] == edges_attr[i]['v'] \
                    or edges_attr[i]['v'] == edges_attr[i]['u']:
                adj[i, j] = 1
                adj[j, i] = 1
    sup = adj
    if support:
        edge_graph = nx.from_numpy_matrix(adj)
        sup = nx.normalized_laplacian_matrix(edge_graph).todense()
    f = edges_df.loc[['layer', 'bandwidth']]
    f = f.to_numpy()

    return sup, f[0, :], f[1, :]


def random_graph_input(num_nodes, p=0.3, support=True):
    graph = nx.erdos_renyi_graph(n=num_nodes, p=p)
    if not nx.is_connected(graph):
        c = list(nx.connected_components(graph))
        for j in range(len(c) - 1):
            graph.add_edge(list(c[j])[0], list(c[j + 1])[0])
    num_edges = graph.number_of_edges()
    edges_attr = {}
    idx = 0
    for u, v in graph.edges:
        edges_attr[idx] = {'u': u, 'v': v}
        idx += 1
    adj = np.zeros([num_edges, num_edges], dtype=np.int)
    for i in range(num_edges - 1):
        for j in range(i + 1, num_edges):
            if edges_attr[i]['u'] == edges_attr[i]['u'] \
                    or edges_attr[i]['v'] == edges_attr[i]['v'] \
                    or edges_attr[i]['u'] == edges_attr[i]['v'] \
                    or edges_attr[i]['v'] == edges_attr[i]['u']:
                adj[i, j] = 1
                adj[j, i] = 1
    edge_graph = nx.from_numpy_matrix(adj)
    sup = nx.normalized_laplacian_matrix(edge_graph).todense()
    bandwidth = np.ones([num_edges], dtype=np.float) * 1000

    return edge_graph, sup, bandwidth


def gen_flows(graph, size_percent: float(0 - 1.0) = 0.1, bandwidth=1000):
    source_nodes = list(graph.nodes)
    dest_nodes = list(graph.nodes)
    while True:
        s = int(np.random.choice(source_nodes, 1))
        d = int(np.random.choice(dest_nodes, 1))
        if s != d:
            yield [s, d, int(np.random.uniform(0.0e1 * bandwidth, size_percent * bandwidth))]


def gen_flows_zte(topo, num_flows, percent=0.05):
    flows = []
    flow_gen = topo.gen_flows(percent)
    for i in range(num_flows):
        flows.append(next(flow_gen))
    return flows


def cal_path_delay(path, now_cap, init_cap):
    delay = 0
    load = init_cap - now_cap
    for v in path:
        l = load[v]
        c = init_cap[v]
        d = 0
        if l / c <= 1 / 3:
            d = l
        elif 1 / 3 < l / c <= 2 / 3:
            d = 3 * l - 2 / 3 * c
        elif 2 / 3 < l / c <= 9 / 10:
            d = 10 * l - 16 / 3 * c
        else:
            d = 70 * l - 178 / 3 * c
        delay = max(delay, d)
    return delay


def cal_total_delay(traffic, init_bd):
    edges = len(init_bd)
    sum_delay = 0
    for e in range(edges):
        l = traffic[e]
        c = init_bd[e]
        if l / c <= 1 / 3:
            d = l
        elif 1 / 3 < l / c <= 2 / 3:
            d = 3 * l - 2 / 3 * c
        elif 2 / 3 < l / c <= 9 / 10:
            d = 10 * l - 16 / 3 * c
        elif 9 / 10 < l / c <= 1:
            d = 70 * l - 178 / 3 * c
        elif 1 < l / c < 11 / 10:
            d = 500 * l - 1468 / 3 * c
        else:
            d = 5000 * l - 16318 / 3 * c
        sum_delay += d
    return sum_delay
