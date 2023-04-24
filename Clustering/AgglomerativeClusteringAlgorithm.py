from sklearn.cluster import AgglomerativeClustering

from Clustering.Clustering import Clustering

class AgglomerativeClusteringAlgorithm(Clustering):
    def __init__(self, n_clusters=2):
        super().__init__(n_clusters)

    # The linkage parameter specifies the method used to compute the distance between clusters.
    # It determines the way clusters are merged during the hierarchical clustering process.
    def cluster(self, data, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = self.model.fit_predict(data)

        #plot the clusters
        self.plot_clusters(data, labels)