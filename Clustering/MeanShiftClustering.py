from sklearn.cluster import MeanShift

from Clustering.Clustering import Clustering

class MeanShiftClustering(Clustering):
    def __init__(self, data):
        super().__init__(data)

    def cluster(self):
        self.model = MeanShift()
        self.labels = self.model.fit_predict(self.data)
        self.centers = self.model.cluster_centers_

        #plot the clusters
        self.plot_clusters(self.model)