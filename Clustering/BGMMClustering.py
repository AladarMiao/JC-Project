
from sklearn.mixture import BayesianGaussianMixture

from Clustering.Clustering import Clustering

class BGMMClustering(Clustering):
    def __init__(self, n_clusters=None):
        super().__init__()
        if not n_clusters:
            n_clusters = self.n_clusters
        self.model = BayesianGaussianMixture(n_components=n_clusters)

