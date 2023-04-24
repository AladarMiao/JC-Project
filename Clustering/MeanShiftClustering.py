from sklearn.cluster import MeanShift

from Clustering.Clustering import Clustering

class MeanShiftClustering(Clustering):
    def __init__(self, n_clusters=2):
        super().__init__(n_clusters)

    def cluster(self, data):
        self.model = MeanShift()
        labels = self.model.fit_predict(data)
        self.centers = self.model.cluster_centers_

        #plot the clusters
        self.plot_clusters(data, labels)