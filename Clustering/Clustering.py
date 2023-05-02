import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, precision_score, silhouette_score, accuracy_score


class Clustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = None
        self.pc_to_plot = 2

        # used to calculate different scores, if needed
        self.labels = None

    def cluster(self, data, actual_labels=None):
        predicted_labels = self.model.fit_predict(data)
        if actual_labels:
            accuracy = accuracy_score(actual_labels, predicted_labels)
            print(f"Accuracy: {accuracy}")
            f1 = f1_score(actual_labels, predicted_labels, average='weighted')
            print(f"F1: {f1}")
            recall = recall_score(actual_labels, predicted_labels, average='weighted')
            print(f"Recall: {recall}")
            precision = precision_score(actual_labels, predicted_labels, average='weighted')
            print(f"Precision: {precision}")
        else:
            score = silhouette_score(data, predicted_labels)
            print(f"Silhouette score: {score}")
        # plot the clusters
        self.plot_clusters(data, predicted_labels, self.n_clusters)

    def plot_clusters(self, data, labels, n_components=2):
        # Fit PCA to the data
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(data)

        # Get the cluster labels and the number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Plot the data points colored by cluster
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, s=50, cmap='viridis')
        plt.title(f'Clustering on PCA (Number of Clusters: {n_clusters})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Remove values on the x and y axes
        plt.tick_params(labelbottom=False, labelleft=False)

        # Show the plot
        plt.show()

        # Save the plot
        folder_name = 'clustering_plots'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig('{}/plot_clusters.png'.format(folder_name))
