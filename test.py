from evaluate import Evaluate
from dataprocessor import DataProcessor

class Args:
    def __init__(self):
        self.graph_name = "ZTE-1-1-2"
        self.random_bandwidth = False
        self.num_paths = 5
        self.num_flows = 100
        self.max_rate = 0.05
        self.min_rate = 0.01
        self.epochs = 20
        self.model_path = 'logs/1028-0141-ZTE-1-1-2-F100-P5'


args = Args()
dp = DataProcessor(args)
eva = Evaluate(args, dp)
eva.evaluate()
# ig = eva.integrated_grads(M=20)
# eva.visualize(ig)
# eva.att_matrix()
