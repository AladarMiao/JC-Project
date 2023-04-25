from DataPreprocessor.DataPreprocessor import DataPreprocessor
import json

from Constants import get_dim_reduction_class, get_clustering_class


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

        # determines whether we are processing images or tabular data with our preprocessor
        if not json_parameters["is_image"]:
            self.preprocessor = DataPreprocessor(csv_train_path=json_parameters["csv_train_path"],
                                                 csv_validation_path=json_parameters["csv_validation_path"])
            train, val = self.preprocessor.return_train_data(), self.preprocessor.return_validation_data()
            if json_parameters["drop_duplicates"]:
                self.preprocessor.drop_duplicates()
            if json_parameters["impute_missing"]:
                train, val = self.preprocessor.impute_missing()
            if json_parameters["dimension_reduction_algorithm"]:
                self.dim_reduction = get_dim_reduction_class(json_parameters["dimension_reduction_algorithm"],
                                                             json_parameters["n_components"])
                train = self.dim_reduction.fit_transform(train)
                val = self.dim_reduction.transform(val)
                print("Data has been reshaped into a {} array with {} features".format(train.shape[0], train.shape[1]))
            if json_parameters["clustering_algorithm"]:
                self.clustering = get_clustering_class(json_parameters["clustering_algorithm"])
                print("Training set clustering")
                self.clustering.cluster(train)
                print("Validation set clustering")
                self.clustering.cluster(val)
            if json_parameters["classifier_algorithm"]:
                self.classifier = get_clustering_class(json_parameters["classifier"])

                #TODO: Align on how we include labels in our data -- e.g. last column of train data?