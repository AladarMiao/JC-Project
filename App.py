from DataPreprocessor.DataPreprocessor import DataPreprocessor
import json

from Util import getDimReductionClass, getClusteringClass


class App:
    def __init__(self):
        self.preprocessor = None
        self.clustering = None
        self.dim_reduction = None
        self.classifier = None

    def start(self):
        # Open the JSON file for reading
        with open('sample.json') as f:
            # Load the contents of the file into a variable
            json_parameters = json.load(f)

        #determines whether we are processing images or tabular data with our preprocessor
        if not json_parameters["is_image"]:
            self.preprocessor = DataPreprocessor(csv_train_path=json_parameters["csv_train_path"],
                                                 csv_validation_path=json_parameters["csv_validation_path"])
            train, val = self.preprocessor.return_train_data(), self.preprocessor.return_validation_data()
            if json_parameters["drop_duplicates"]:
                self.preprocessor.drop_duplicates()
            if json_parameters["impute_missing"]:
                train, val = self.preprocessor.impute_missing()
            if json_parameters["dimension_reduction_algorithm"]:
                self.dim_reduction = getDimReductionClass(json_parameters["dimension_reduction_algorithm"], json_parameters["n_components"])
                train = self.dim_reduction.fit_transform(train)
                val = self.dim_reduction.transform(val)
                print("Data has been reshaped into a {} array with {} features".format(train.shape[0], train.shape[1]))
            if json_parameters["clustering_algorithm"]:
                self.clustering = getClusteringClass(json_parameters["clustering_algorithm"])
                print("Training set clustering")
                self.clustering.cluster(train)

                self.clustering.silhouette_score()
        else:
            self.preprocessor = DataPreprocessor(images_train_path=json_parameters["images_train_path"],
                                                 images_validation_path=json_parameters["images_validation_path"])
            if json_parameters["new_width"]:
                self.preprocessor.resize_images(json_parameters["new_width"],  json_parameters["new_height"])

        json_parameters = self.preprocessor.return_data()
        print(json_parameters.shape)

        #Ask the user for dimension reduction options
        dim_reduction = input("Do you wish to perform dimension reduction? (y/n) ")
        if dim_reduction.lower()=="y":
            dimension_reduction_algorithm = input("""Which dimension reduction algorithm do you wish to use? We currently support the following:
                                        t-SNE, PCA """)
            n_components = int(input("""How many principal components are you looking for? """))
            self.dim_reduction = Util.getDimReductionClass(dimension_reduction_algorithm, n_components)
            json_parameters = self.dim_reduction.fit_transform(json_parameters)
            print("Data has been reshaped into a {} array with {} features".format(json_parameters.shape[0], json_parameters.shape[1]))

            # Ask the user if they want to download the reshaped data as a CSV file
            display(json_parameters)
            download = input("Do you want to download the reshaped data as a CSV file? (y/n)")

            if download.lower() == 'y':
                # Save the reshaped data as a CSV file
                np.savetxt("reshaped_data.csv", data_reshaped, delimiter=",")
                print("Reshaped data has been saved as reshaped_data.csv")
            else:
                print("Reshaped data was not downloaded")

        # Ask the user for clustering options
        clustering_algorithm = input("""Which clustering algorithm do you wish to use? We currently support the following:
                                        kmeans, dbscan, meanshift, agglomerative, isolation forest, bgmm """)

        self.clustering = Util.getClusteringClass(clustering_algorithm, json_parameters)

        # Perform clustering
        self.clustering.cluster()

        # Measure performance
        silhouette_score = self.clustering.silhouette_score(json_parameters)

        # Print the score
        print("Silhouette score:", silhouette_score)