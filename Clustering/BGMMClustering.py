
from sklearn.mixture import BayesianGaussianMixture

from Clustering.Clustering import Clustering

class BGMMClustering(Clustering):
    def __init__(self, data):
        super().__init__(data)

    def cluster(self, n_components=None):
        if n_components is None:
            n_components = 2
        self.model = BayesianGaussianMixture(n_components=n_components)
        self.model.fit(self.data)
        self.labels = self.model.predict(self.data)

        #plot the clusters
        self.plot_clusters(self.model)