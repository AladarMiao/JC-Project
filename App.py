from DataPreprocessor.DataPreprocessor import DataPreprocessor
import json

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
            data = json.load(f)

        #determines whether we are processing images or tabular data with our preprocessor
        if data["is_image"]:
            self.preprocessor = DataPreprocessor(data["is_image"])

        # Ask the user for preprocessing options
        self.preprocessor.drop_duplicates()
        self.preprocessor.impute_missing()

        # Show the user how the new table looks, and ask the user if he/she wants to download it
        self.preprocessor.display_table()

        data = self.preprocessor.return_data()
        print(data.shape)

        #Ask the user for dimension reduction options
        dim_reduction = input("Do you wish to perform dimension reduction? (y/n) ")
        if dim_reduction.lower()=="y":
            dimension_reduction_algorithm = input("""Which dimension reduction algorithm do you wish to use? We currently support the following:
                                        t-SNE, PCA """)
            n_components = int(input("""How many principal components are you looking for? """))
            self.dim_reduction = Util.getDimReductionClass(dimension_reduction_algorithm, n_components)
            data = self.dim_reduction.fit_transform(data)
            print("Data has been reshaped into a {} array with {} features".format(data.shape[0], data.shape[1]))

            # Ask the user if they want to download the reshaped data as a CSV file
            display(data)
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

        self.clustering = Util.getClusteringClass(clustering_algorithm, data)

        # Perform clustering
        self.clustering.cluster()

        # Measure performance
        silhouette_score = self.clustering.silhouette_score(data)

        # Print the score
        print("Silhouette score:", silhouette_score)