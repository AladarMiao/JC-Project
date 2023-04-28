import numpy as np
from PIL import Image
import pandas as pd
from IPython.display import display
from sklearn.impute import SimpleImputer
import os

from Constants import TABULAR, IMAGE


def isImage(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))


class DataPreprocessor:
    def __init__(self, is_image_data, csv_train_path=None, csv_validation_path=None, images_train_path=None,
                 images_validation_path=None, labeled=None, images_train_labels=None, images_validation_labels=None):
        # TODO: add error handling for incorrect file types
        if not is_image_data:
            self.filetype = TABULAR
            self.df_train = pd.read_csv(csv_train_path)
            self.df_validation = pd.read_csv(csv_validation_path)
            if labeled:
                self.train_labels = self.df_train.iloc[:, -1].values
                self.validation_labels = self.df_validation.iloc[:, -1].values
                self.df_train = self.df_train.iloc[:, :-1]
                self.df_validation = self.df_validation.iloc[:, :-1]
        else:
            self.filetype = IMAGE
            self.images_train = [os.path.join(images_train_path, f) for f in os.listdir(images_train_path) if
                                 isImage(f)]
            self.images_validation = [os.path.join(images_validation_path, f) for f in
                                      os.listdir(images_validation_path) if isImage(f)]
            if labeled:
                self.train_labels = images_train_labels
                self.validation_labels = images_validation_labels

    def impute_missing(self, impute_strategy):
        imputer = SimpleImputer()
        if impute_strategy.isdigit():
            imputer = SimpleImputer(strategy='constant', fill_value=int(impute_strategy))
        else:
            imputer = SimpleImputer(strategy=impute_strategy)
        self.df_train = pd.DataFrame(imputer.fit_transform(self.df_train), columns=self.df_train.columns)
        self.df_validation = pd.DataFrame(imputer.transform(self.df_validation), columns=self.df_validation.columns)
        return self.df_train, self.df_validation

    def display_table(self):
        display(self.df_train)
        display(self.df_validation)

    def return_train_data(self):
        if self.filetype == IMAGE:
            return self.images_train
        return self.df_train

    def return_train_labels(self):
        return self.train_labels

    def return_validation_data(self):
        if self.filetype == IMAGE:
            return self.images_validation
        return self.df_validation

    def return_validation_labels(self):
        return self.validation_labels

    def resize_images(self, new_width, new_height):
        """
        Resizes all images in a folder
        """
        # create np array for the dataset
        training_set = np.empty((len(self.images_train), new_width, new_height))
        validation_set = np.empty((len(self.images_validation), new_width, new_height))
        # Load the training images into the X_train array
        for i, filename in enumerate(self.images_train):
            image = Image.open(filename)
            # Resize the image
            new_image = image.resize((new_width, new_height))
            training_set[i] = np.asarray(new_image,
                                         dtype=np.float32) / 255.0  # Normalize the pixel values to between 0 and 1

        for i, filename in enumerate(self.images_validation):
            image = Image.open(filename)
            # Resize the image
            new_image = image.resize((new_width, new_height))
            validation_set[i] = np.asarray(new_image,
                                           dtype=np.float32) / 255.0  # Normalize the pixel values to between 0 and 1
        self.images_train = training_set
        self.images_validation = validation_set

        return self.images_train, self.images_validation
    #
    # def rotate_images(self, path, degrees_clockwise):
    #
    # def flip_images(self):
