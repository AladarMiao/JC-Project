from Clustering.Clustering import Clustering
from sklearn.cluster import KMeans

class KMeansClustering(Clustering):
    def __init__(self, data):
        super().__init__(data)

    def elbow_method(self):
        # Calculate the within-cluster sum of squares (WCSS) for each k value
        wcss = []

        # Define range of k values to test
        k_values = range(1, 10)

        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.plot(k_values, wcss, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')

        # Determine the optimal number of clusters
        diffs = np.diff(wcss)
        diff_ratios = diffs[1:] / diffs[:-1]
        optimal_k = k_values[np.argmin(diff_ratios) + 1]

        # Display the optimal number of clusters
        plt.axvline(x=optimal_k, linestyle='--', color='r', label=f'Optimal k={optimal_k}')
        plt.legend()
        plt.show()

        print(f"Optimal number of clusters: {optimal_k}")

        return optimal_k

    def cluster(self, n_clusters=None):
        warnings.filterwarnings("ignore")
        if n_clusters is None:
            n_clusters = self.elbow_method()
        kmeans = KMeans(n_clusters=n_clusters)
        self.model = kmeans
        self.labels = kmeans.fit_predict(self.data)

        #plot the clusters
        self.plot_clusters(self.model)