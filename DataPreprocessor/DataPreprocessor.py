import numpy as np
from PIL import Image
from Util import is_image
import pandas as pd
import datetime as dt
from IPython.display import display
from sklearn.impute import SimpleImputer
import os

class DataPreprocessor:
    def __init__(self, is_image, csv_train = None, csv_validation = None, images_train_path = None, images_validation_path=None):
        self.file_type = "image" if is_image else "csv"
        if not is_image:
            self.df_train = pd.read_csv(csv_train)
            self.df_validation = pd.read_csv(csv_validation)
        else:
            self.images_train = [os.path.join(images_train_path, f) for f in os.listdir(images_train_path) if is_image(f)]
            self.images_validation = [os.path.join(images_validation_path, f) for f in os.listdir(images_validation_path) if is_image(f)]

def drop_duplicates(self):
        remove_duplicates = input("Do you want to remove duplicates? Input T or F ") == "T"
        if remove_duplicates:
            self.df.drop_duplicates(inplace=True)

    def impute_missing(self, method='mean'):

        impute_strategy = input("How do you want to impute your missing values? e.g. \"median\", \"0\" " )
        imputer = SimpleImputer()
        if impute_strategy.isdigit():
            imputer = SimpleImputer(strategy='constant', fill_value=int(impute_strategy))
        else:
            imputer = SimpleImputer(strategy = impute_strategy)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)

    def display_table(self):
        display(self.df)
        to_csv = input("This is your new data file -- do you want to download it? Input T or F ") == "T"
        if to_csv:
            # Get the current timestamp
            timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            # Define the output filename
            # Split the filename into basename and extension
            basename, extension = os.path.splitext(self.filename)
            download_filename = f'{basename}_{timestamp}{extension}'

            # Download the dataframe to the output file
            self.df.to_csv(download_filename, index=False)

    def return_data(self):
        return self.df.iloc[:, :-1].values

    def return_labels(self):
        return self.df.iloc[:, -1].values

    def resize_images(self, path, new_width, new_height):
        """
        Resizes all images in a folder
        """
        # get all images in this path
        files = [os.path.join(path, f) for f in os.listdir(path) if is_image(f)]

        # create np array for the dataset
        dataset = np.empty((len(files), new_height, new_width), dtype=np.uint8)

        # Load the training images into the X_train array
        for i, filename in enumerate(files):
            image = Image.open(filename)
            # Resize the image
            new_image = image.resize((new_width, new_height))
            dataset[i] = np.asarray(new_image, dtype=np.float32) / 255.0 # Normalize the pixel values to between 0 and 1

        return dataset
    #
    # def rotate_images(self, path, degrees_clockwise):
    #
    # def flip_images(self):