from sklearn.cluster import DBSCAN

from Clustering.Clustering import Clustering
import warnings

class DBSCANClustering(Clustering):
    def __init__(self, data):
        super().__init__(data)

    def cluster(self, eps=None, min_samples=None):
        warnings.filterwarnings("ignore")
        if eps is None:
            #             eps = np.sqrt(self.data.shape[1])
            eps=5000
        if min_samples is None:
            min_samples = 2 * self.data.shape[1]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.model = dbscan
        self.labels = dbscan.fit_predict(self.data)

        #plot the clusters
        self.plot_clusters(self.model)