from sklearn.ensemble import IsolationForest

from Clustering.Clustering import Clustering


class IsolationForestClustering(Clustering):
    def __init__(self, n_clusters=None):
        super().__init__()
        self.model = IsolationForest(contamination=0.05)
