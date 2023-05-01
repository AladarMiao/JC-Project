from Classifier.DecisionTree import DecisionTree
from Classifier.MLP import MLP
from Classifier.RandomForest import RandomForest
from Classifier.XGBoost import XGBoost
from Clustering.AgglomerativeClusteringAlgorithm import AgglomerativeClusteringAlgorithm
from Clustering.BGMMClustering import BGMMClustering
from Clustering.DBSCANClustering import DBSCANClustering
from Clustering.IsolationForestClustering import IsolationForestClustering
from Clustering.KMeansClustering import KMeansClustering
from Clustering.MeanShiftClustering import MeanShiftClustering
from DLModels.ConvolutionalAutoencoder import ConvolutionalAutoencoder
from DimensionReduction.MyPCA import MyPCA
from DimensionReduction.MyTSNE import MyTSNE

TABULAR = "tabular"
IMAGE = "image"


def get_clustering_class(user_input_clustering_algorithm, n_clusters=2):
    # create a dictionary that maps user input to the corresponding class
    clustering_class_dict = {
        "kmeans": KMeansClustering(n_clusters),
        "dbscan": DBSCANClustering(n_clusters),
        "agglomerative": AgglomerativeClusteringAlgorithm(n_clusters),
        "meanshift": MeanShiftClustering(n_clusters),
        "isolation forest": IsolationForestClustering(n_clusters),
        "bgmm": BGMMClustering(n_clusters)
    }

    return clustering_class_dict[user_input_clustering_algorithm]


def get_dim_reduction_class(user_input_dim_reduction_algorithm, n_components):
    # create a dictionary that maps user input to the corresponding class
    dim_reduction_class_dict = {
        "t-SNE": MyTSNE(n_components),
        "PCA": MyPCA(n_components)
    }

    return dim_reduction_class_dict[user_input_dim_reduction_algorithm]


def get_classifier_class(user_input_classification_algorithm):
    # create a dictionary that maps user input to the corresponding class
    classifier_class_dict = {
        "decision tree": DecisionTree(),
        "MLP": MLP(),
        "random forest": RandomForest(),
        "XGBoost": XGBoost()
    }

    return classifier_class_dict[user_input_classification_algorithm]


def get_dl_class(user_input_dl_algorithm):
    # create a dictionary that maps user input to the corresponding class
    dl_model_class_dict = {
        "ConvolutionalAutoencoder": ConvolutionalAutoencoder()
    }

    return dl_model_class_dict[user_input_dl_algorithm]
