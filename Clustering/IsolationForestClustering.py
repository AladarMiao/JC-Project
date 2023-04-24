from sklearn.ensemble import IsolationForest

from Clustering.Clustering import Clustering


class IsolationForestClustering(Clustering):
    def __init__(self, n_clusters=2):
        super().__init__(n_clusters)

    def cluster(self, data, n_clusters = None, contamination=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        if contamination is None:
            contamination = 0.05
        self.model = IsolationForest(contamination=contamination)
        self.model.fit(data)
        labels = self.model.predict(data)

        #plot the clusters
        self.plot_clusters(data, labels)
