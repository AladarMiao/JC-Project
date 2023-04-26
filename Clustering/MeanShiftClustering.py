from sklearn.cluster import MeanShift

from Clustering.Clustering import Clustering

class MeanShiftClustering(Clustering):
    def __init__(self, n_clusters=None):
        super().__init__()
        self.model = MeanShift()