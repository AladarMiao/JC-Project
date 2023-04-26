from sklearn.cluster import AgglomerativeClustering

from Clustering.Clustering import Clustering

class AgglomerativeClusteringAlgorithm(Clustering):
    def __init__(self, n_clusters=None):
        super().__init__()
        if not n_clusters:
            n_clusters = self.n_clusters
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')

    # The linkage parameter specifies the method used to compute the distance between clusters.
    # It determines the way clusters are merged during the hierarchical clustering process.
