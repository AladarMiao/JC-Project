from sklearn.cluster import DBSCAN

from Clustering.Clustering import Clustering

class DBSCANClustering(Clustering):
    def __init__(self, n_clusters=None):
        super().__init__()
        self.model = DBSCAN(eps=5000, min_samples=10)