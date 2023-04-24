
from sklearn.mixture import BayesianGaussianMixture

from Clustering.Clustering import Clustering

class BGMMClustering(Clustering):
    def __init__(self, n_clusters=2):
        super().__init__(n_clusters)

    def cluster(self, data, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        self.model = BayesianGaussianMixture(n_components=n_clusters)
        self.model.fit(data)
        labels = self.model.predict(data)

        #plot the clusters
        self.plot_clusters(data, labels)