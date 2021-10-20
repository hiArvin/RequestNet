import numpy as np
import time
import argparse
import tensorflow as tf
from utils import *
from models import PEM
from dataprocessor import DataProcessor

class Evaluate:
    def __init__(self,args,data_processor):
        self.args = args
        # hyper-parameters
        self.num_flows = args.num_flows
        self.num_paths = args.num_paths
        self.epochs = args.epochs
        self.model_path = args.model_path

        self.data_processor = data_processor
        self.num_edges = data_processor.num_edges
        self.placeholders = {
            'support': tf.placeholder(tf.float32, shape=(self.num_edges, self.num_edges)),
            'features': tf.placeholder(tf.float32),
            'labels': tf.placeholder(tf.int64, shape=(self.num_flows, self.num_paths)),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'paths': tf.placeholder(tf.int64),
            'index': tf.placeholder(tf.int64),
            'sequences': tf.placeholder(tf.int64)
        }
        self.model = PEM(num_paths=self.num_paths, num_edges=self.num_edges, num_quests=self.num_flows,
                         placeholders=self.placeholders,
                         learning_rate=self.lr, gcn_input_dim=self.num_paths * self.num_flows,
                         gcn_hidden_dim=16, gcn_output_dim=8, pe_output_dim=4, att_layers_num=3)
        self.sess = tf.Session()
        self.model.load(self.sess,self.model_path)

    def evaluate(self):
        for epoch in range(self.epochs):
            bandwidth = self.data_processor.bandwidth
            flows = self.data_processor.generate_flows()
            sp = self.data_processor.flow_to_numpy(flows)
            paths, idx, seqs = self.data_processor.generate_seqs(flows)
            support_matrix = self.data_processor.get_laplacian_matrix()
            sp_flatten = sp.reshape([len(flows) * self.num_paths, self.num_edges])
            fp2 = sp_flatten / np.tile(bandwidth * self.args.max_rate, [self.num_flows * self.num_paths, 1])
            feed_dict = {
                self.placeholders['support']: support_matrix,
                self.placeholders['features']: fp2.T,
                self.placeholders['paths']: paths,
                self.placeholders['index']: idx,
                self.placeholders['sequences']: seqs,
            }
            time1 = time.time()
            outs_pd = self.sess.run(self.model.outputs, feed_dict=feed_dict)
            time2 = time.time()
            outs_pd = np.nanargmax(softmax(outs_pd), axis=1)
            delay_pd = self.data_processor.cal_delay_for_model(sp, outs_pd)
            time3 = time.time()
            outs_seq, delay_seq = self.data_processor.sequential_delay_outputs(flows)
            time4 = time.time()
            traffic = np.zeros_like(bandwidth, dtype=int)
            label, delay_opt = self.data_processor.generate_delay_label(sp, traffic, bandwidth)
            time5 = time.time()
            print("延迟", sum(delay_pd), np.sum(delay_seq), np.sum(delay_opt))
            print("时间", time2 - time1, '\t', time4 - time3, '\t', time5 - time4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_flows", type=int, help="Number of flows process in an epoch")
    parser.add_argument("--num_paths", type=int, help="Number of candidate paths for a flow")
    parser.add_argument("--epochs", type=int, help="Evaluation epochs")
    parser.add_argument("--model_path",help="Path of trained model")
    parser.add_argument("--max_rate", type=float, default=0.05, help="flow size / bandwidth")
    parser.add_argument("--min_rate", type=float, default=0.001, help="flow size / bandwidth")
    parser.add_argument("--random_bandwidth", default=False)
    parser.add_argument("--graph_name", default="Aarnet.graphml")

    args = parser.parse_args()
    dp = DataProcessor(args)
    eva = Evaluate(args,dp)
    eva.evaluate()
