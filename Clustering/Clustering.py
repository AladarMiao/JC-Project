import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, silhouette_score
class Clustering:
    def __init__(self, data):
        self.model = None
        self.data = data
        self.pc_to_plot = 2

        #used to calculate different scores, if needed
        self.labels = None

    def f1_score(self, labels_true):
        return f1_score(labels_true, self.labels)

    def recall_score(self, labels_true):
        return recall_score(labels_true, self.labels)

    def precision_score(self, labels_true):
        return precision_score(labels_true, self.labels)

    def silhouette_score(self, data):
        return silhouette_score(data, self.labels)

    def cluster(self):
        raise NotImplementedError()

    def plot_clusters(self, model):
        # Fit PCA to the data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.data)

        # Get the cluster labels and the number of clusters
        labels = self.labels
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