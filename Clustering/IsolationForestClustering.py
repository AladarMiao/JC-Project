from sklearn.ensemble import IsolationForest

from Clustering.Clustering import Clustering


class IsolationForestClustering(Clustering):
    def __init__(self, data):
        super().__init__(data)

    def cluster(self, contamination=None):
        if contamination is None:
            contamination = 0.05
        self.model = IsolationForest(contamination=contamination)
        self.model.fit(self.data)
        self.labels = self.model.predict(self.data)

        #plot the clusters
        self.plot_clusters(self.model)