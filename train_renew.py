import time
import random
import tensorflow as tf
from models import *
from utils import *

import networkx as nx
from label import gen_label
from datetime import datetime
from dataprocessor import DataProcessor
import argparse


class Trainer:
    def __init__(self, args, data_procesor):
        self.args = args
        # hyper-parameters
        self.num_flows = args.num_flows
        self.num_paths = args.num_paths
        self.epoches = args.epoches
        self.lr = args.learning_rate
        self.save = args.save



        self.data_procesor = data_procesor
        self.num_edges = data_procesor.num_edges
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
        # tf config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if self.save:
            self.merged = tf.summary.merge_all()
            save_path = './logs/' + datetime.now().strftime('%m%d-%H%M')
            save_path += '-Q' + str(self.num_flows) + '-P' + str(self.num_paths)
            self.summary_writer = tf.summary.FileWriter(save_path, graph=self.sess.graph)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

    def gen_feed_dict(self):
        bandwidth=self.data_procesor.bandwidth
        # traffic = np.random.randint(Min_Cap // 5 * 100, Min_Cap // 3 * 100, size=NUM_EDGES)
        traffic = np.zeros(self.num_edges)
        flows = self.data_procesor.generate_flows()
        shortest_paths = self.data_procesor.shortest_paths
        sp = self.data_procesor.flow_to_numpy(flows)
        paths, idx, seqs = self.data_procesor.generate_seqs(flows)
        support_matrix = self.data_procesor.get_laplacian_matrix()
        labels, delay_opt = self.data_procesor.generate_delay_label(sp, traffic, bandwidth)

        sp_flatten = sp.reshape([len(flows) * self.num_paths, self.num_edges])
        print(sp_flatten)
        fp2 = sp_flatten / np.tile(bandwidth*self.args.max_rate, [self.num_flows * self.num_paths, 1])
        # Construct feed dictionary
        feed_dict = construct_feed_dict(fp2.T, support_matrix, labels, paths, idx, seqs, self.placeholders)
        feed_dict.update({self.placeholders['dropout']: 0.})
        return feed_dict, labels, sp, delay_opt

    def train(self):
        # Train model
        acc_num = 0
        early_stop = 0
        for epoch in range(self.epoches):
            feed_dict, labels, sp_numpy,delay_opt = self.gen_feed_dict()
            lb = np.nanargmax(labels, axis=1)
            print("Labels   :\t", lb)
            # Training step
            outs = self.sess.run([self.model.outputs, self.model.loss, self.model.accuracy, self.model.opt_op], feed_dict=feed_dict)
            # Print results
            print('Predictor:\t', np.nanargmax(softmax(outs[0]), axis=1))
            # print("Outputs(line1):", outs[0][0])
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]))
            if self.save:
                summary = self.sess.run(self.merged, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary, epoch)

    def evaluate(self):
        for epoche in range(int(5)):
            bandwidth = self.data_procesor.bandwidth
            flows = self.data_procesor.generate_flows()
            sp = self.data_procesor.flow_to_numpy(flows)
            paths, idx, seqs = self.data_procesor.generate_seqs(flows)
            support_matrix = self.data_procesor.get_laplacian_matrix()
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
            outs_pd= np.nanargmax(softmax(outs_pd), axis=1)
            delay_pd = self.data_procesor.cal_delay_for_model(sp,outs_pd)
            time3 = time.time()
            outs_seq,delay_seq= self.data_procesor.sequential_delay_outputs(flows)
            time4=time.time()
            traffic = np.zeros_like(bandwidth,dtype=int)
            label,delay_opt= self.data_procesor.generate_delay_label(sp,traffic ,bandwidth)
            time5 = time.time()
            print("延迟", sum(delay_pd), np.sum(delay_seq),np.sum(delay_opt))
            print("时间",time2-time1,'\t',time4-time3,'\t',time5-time4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_flows",type=int, help="Number of flows process in an epoch")
    parser.add_argument("--num_paths",type=int, help="Number of candidate paths for a flow")
    parser.add_argument("--epoches",type=int, help="training epoches")
    parser.add_argument("--max_rate",type=float, default=0.05,help="flow size / bandwidth")
    parser.add_argument("--min_rate",type=float, default=0.001,help="flow size / bandwidth")
    parser.add_argument("--random_bandwidth", default=False)
    parser.add_argument("--training_graph", default="Aarnet.graphml")
    parser.add_argument("--learning_rate", default=0.005)
    parser.add_argument("--save", default=True)

    args = parser.parse_args()
    data_procesor = DataProcessor(args)
    trainer = Trainer(args,data_procesor)
    trainer.train()
    trainer.evaluate()
