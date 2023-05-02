import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, precision_score, silhouette_score, accuracy_score
from datetime import datetime
import pandas as pd

class Clustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = None
        self.pc_to_plot = 2

        # used to calculate different scores, if needed
        self.labels = None

    def cluster(self, data, actual_labels=None):
        predicted_labels = self.model.fit_predict(data)
        if actual_labels is not None:
            labels_cat = pd.Categorical(actual_labels)
            label_dict = dict(enumerate(labels_cat.categories))

            # Map each category to a unique integer code
            labels_codes = labels_cat.codes
            print(f"Labels codes: {labels_codes}")
            print(f"Labels mapping: {label_dict}")

            self.plot_clusters(data, labels_codes, n_components=self.n_clusters, title="Correct Labels")

            print(f"Predicted labels codes: {predicted_labels}")
            self.plot_clusters(data, predicted_labels, n_components=self.n_clusters, title="Predicted Labels")

            return predicted_labels, actual_labels.astype(str)

        else:
            score = silhouette_score(data, predicted_labels)
            print(f"Silhouette score: {score}")
            self.plot_clusters(data, predicted_labels, self.n_clusters)

    def plot_clusters(self, data, labels, n_components=2, title="Clustering"):
        # Fit PCA to the data
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(data)

        # Plot the data points colored by cluster
        fig, ax = plt.subplots()
        sc = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, s=50, cmap='viridis')

        # Create a legend for the scatter plot
        ax.legend(*sc.legend_elements())

        plt.title(title)
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

        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{folder_name}/plot_clusters_{now}.png'

        # Check if the file already exists
        if os.path.exists(filename):
            # If it does, add a counter to the filename
            i = 1
            while True:
                new_filename = f'{folder_name}/plot_clusters_{now}_{i}.png'
                if not os.path.exists(new_filename):
                    filename = new_filename
                    break
                i += 1

        plt.savefig(filename)
