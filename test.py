from evaluate import Evaluate
from dataprocessor import DataProcessor

class Args:
    def __init__(self):
        self.graph_name = "Deltacom.graphml"
        self.random_bandwidth = False
        self.num_paths = 5
        self.num_flows = 150
        self.max_rate = 0.05
        self.min_rate = 0.001
        self.epochs = 20
        self.model_path = 'logs/1112-2120-Deltacom-F10-P5'
        self.model_flows = 10


args = Args()
dp = DataProcessor(args)
eva = Evaluate(args, dp)
eva.evaluate()
# ig = eva.integrated_grads(M=20)
# eva.visualize(ig)
# eva.att_matrix()
