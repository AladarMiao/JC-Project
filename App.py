from DataPreprocessor.DataPreprocessor import DataPreprocessor
import json
import sys
import warnings
from Constants import get_dim_reduction_class, get_clustering_class, get_classifier_class

warnings.filterwarnings("ignore")

class App:
    def __init__(self, json_filename):
        with open(json_filename) as f:
            self.json_parameters = json.load(f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python App.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    app = App(filename)
    labeled = app.json_parameters.get("labeled", False)
    preprocessor_parameters = app.json_parameters.get("preprocessor_parameters", {})
    dimension_reduction_parameters = app.json_parameters.get("dimension_reduction_parameters", {})
    clustering_parameters = app.json_parameters.get("clustering_parameters", {})
    classifier_parameters = app.json_parameters.get("classifier_parameters", {})
    DL_parameters = app.json_parameters.get("DL_parameters", {})
    train, val = None, None

    if preprocessor_parameters:
        preprocessor = DataPreprocessor(preprocessor_parameters.get("is_image_data"),
                                        csv_train_path=preprocessor_parameters.get("csv_train_path"),
                                        csv_validation_path=preprocessor_parameters.get("csv_validation_path"),
                                        images_train_path=preprocessor_parameters.get("images_train_path"),
                                        images_validation_path=preprocessor_parameters.get("images_validation_path"),
                                        labeled=labeled,
                                        images_train_labels=preprocessor_parameters.get("images_train_labels"),
                                        images_validation_labels=preprocessor_parameters.get("images_validation_labels"))
        train, val = preprocessor.return_train_data(), preprocessor.return_validation_data()
        train, val = preprocessor.impute_missing(preprocessor_parameters.get("impute_missing_values"))
    if dimension_reduction_parameters:
        if dimension_reduction_parameters.get("algo"):
            dim_reduction = get_dim_reduction_class(dimension_reduction_parameters.get("algo"),
                                                    dimension_reduction_parameters.get("principal_components"))
            train = dim_reduction.fit_transform(train)
            val = dim_reduction.fit_transform(val)
            print("Data has been reshaped into a {} array with {} features".format(train.shape[0], train.shape[1]))
    if clustering_parameters:
        if clustering_parameters.get("algo"):
            clustering = get_clustering_class(clustering_parameters.get("algo"),
                                              n_clusters=clustering_parameters.get("n_clusters"))
            print("Training set clustering")
            clustering.cluster(train)
            print("Validation set clustering")
            clustering.cluster(val)
    if classifier_parameters and labeled:
        if classifier_parameters.get("algo"):
            classifier = get_classifier_class(classifier_parameters.get("algo"))
            classifier.fit(train, classifier_parameters.get("labels"))
