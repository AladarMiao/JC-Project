import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, precision_score, silhouette_score
class Clustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = None
        self.pc_to_plot = 2

        #used to calculate different scores, if needed
        self.labels = None
    #
    # def f1_score(self, labels_true):
    #     return f1_score(labels_true, self.labels)
    #
    # def recall_score(self, labels_true):
    #     return recall_score(labels_true, self.labels)
    #
    # def precision_score(self, labels_true):
    #     return precision_score(labels_true, self.labels)

    def silhouette_score(self, data, labels):
        return silhouette_score(data, labels)

    def cluster(self, data):
        raise NotImplementedError()

    def plot_clusters(self, data, labels):
        # Fit PCA to the data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data)

        # Get the cluster labels and the number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Plot the data points colored by cluster
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
        plt.title(f'Clustering on PCA (Number of Clusters: {n_clusters})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Remove values on the x and y axes
        plt.tick_params(labelbottom=False, labelleft=False)

        #Show the plot
        plt.show()