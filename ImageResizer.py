import os
from PIL import Image
from Util import isImage

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
        self.new_folder = None
        self.num_images = 0

    def resize_images(self):
        """
        Resizes all images in a folder to a given height and width and saves them to a new folder.
        """

        # Create a new folder to save the resized images
        new_folder = os.path.join(self.path, 'resized')
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        self.new_folder = new_folder

        # Loop through all images in the folder
        for file_name in os.listdir(self.path):
            if isImage(file_name):
                # Open the image
                image = Image.open(os.path.join(self.path, file_name))

                # Resize the image
                new_image = image.resize((self.new_width, self.new_height))

                # Save the resized image to the new folder
                new_image.save(os.path.join(new_folder, file_name))

                # Increment the number of resized images
                self.num_images += 1