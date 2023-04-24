from sklearn.cluster import DBSCAN

from Clustering.Clustering import Clustering

class DBSCANClustering(Clustering):
    def __init__(self, n_clusters=2):
        super().__init__(n_clusters)

    def cluster(self, data, n_clusters=None, eps=None, min_samples=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        if eps is None:
            #             eps = np.sqrt(self.data.shape[1])
            eps=5000
        if min_samples is None:
            min_samples = 2 * self.data.shape[1]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.model = dbscan
        labels = dbscan.fit_predict(data)

        #plot the clusters
        self.plot_clusters(data, labels)