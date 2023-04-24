from sklearn.cluster import AgglomerativeClustering

from Clustering.Clustering import Clustering

class AgglomerativeClusteringAlgorithm(Clustering):
    def __init__(self, data):
        super().__init__(data)

    # The linkage parameter specifies the method used to compute the distance between clusters.
    # It determines the way clusters are merged during the hierarchical clustering process.
    def cluster(self, n_clusters=2, linkage='ward'):
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = self.model.fit_predict(self.data)

        #plot the clusters
        self.plot_clusters(self.model)