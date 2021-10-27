from evaluate import Evaluate
from dataprocessor import DataProcessor

class Args:
    def __init__(self):
        self.graph_name = "ZTE"
        self.random_bandwidth = False
        self.num_paths = 5
        self.num_flows = 500
        self.max_rate = 0.05
        self.min_rate = 0.01
        self.epochs = 10
        self.model_path = 'logs/1027-1715-ZTE-F500-P5'


args = Args()
dp = DataProcessor(args)
eva = Evaluate(args, dp)
eva.evaluate()
# eva.att_matrix()
