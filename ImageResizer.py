import os
import numpy as np
from PIL import Image
from Util import is_image

class ImageResizer:
    """
    A class that resizes all images in a folder to a given height and width and saves them to a new folder.
    """

    def __init__(self):
        """
        Initializes the ImageResizer with the path of the folder containing the images, the new width,
        and the new height for the images.
        """
        self.path = input("Enter the path of the folder containing the images: ")
        self.new_width = int(input("Enter the new width for the images: "))
        self.new_height = int(input("Enter the new height for the images: "))

    def resize_images(self):
        """
        Resizes all images in a folder
        """
        # count number of images in this path
        files = [os.path.join(self.path, f) for f in os.listdir(self.path) if is_image(f)]

        # create np array for the dataset
        dataset = np.empty((len(files), self.new_height, self.new_width), dtype=np.uint8)

        # Load the training images into the X_train array
        for i, filename in enumerate(files):
            image = Image.open(filename)
            # Resize the image
            new_image = image.resize((self.new_width, self.new_height))
            dataset[i] = np.asarray(new_image, dtype=np.float32) / 255.0 # Normalize the pixel values to between 0 and 1

        return dataset