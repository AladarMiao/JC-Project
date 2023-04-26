from Clustering.Clustering import Clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

class KMeansClustering(Clustering):
    def __init__(self, n_clusters=None):
        super().__init__()
        if not n_clusters:
            n_clusters = self.n_clusters
        kmeans = KMeans(n_clusters=n_clusters)
        self.model = kmeans
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
        self.n_clusters = optimal_k
        return optimal_k
