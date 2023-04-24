from Clustering.AgglomerativeClusteringAlgorithm import AgglomerativeClusteringAlgorithm
from Clustering.BGMMClustering import BGMMClustering
from Clustering.DBSCANClustering import DBSCANClustering
from Clustering.IsolationForestClustering import IsolationForestClustering
from Clustering.KMeansClustering import KMeansClustering
from Clustering.MeanShiftClustering import MeanShiftClustering
from DimensionReduction.MyPCA import MyPCA
from DimensionReduction.MyTSNE import MyTSNE

def getClusteringClass(user_input_clustering_algorithm, n_clusters=2):
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

def getDimReductionClass(user_input_dim_reduction_algorithm, n_components):
    # create a dictionary that maps user input to the corresponding class
    dim_reduction_class_dict = {
        "t-SNE": MyTSNE(n_components),
        "PCA": MyPCA(n_components)
    }

    return dim_reduction_class_dict[user_input_dim_reduction_algorithm]

def isImage(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))