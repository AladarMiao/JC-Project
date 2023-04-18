from DLModels import ConvolutionalAutoencoder
from ImageResizer import ImageResizer
from Util import isImage
from PIL import Image
import tensorflow as tf
import os
import numpy as np

print(tf.config.list_physical_devices('GPU'))
nb_classes = 10

print("Resize Training Set")
train_resizer = ImageResizer()
train_resizer.resize_images()
width, height, train_path = train_resizer.new_width, train_resizer.new_height, train_resizer.new_folder

print("Resize Validation Set")
val_resizer = ImageResizer()
val_resizer.resize_images()
val_path = val_resizer.new_folder

# Get a list of all image filenames
train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if isImage(f)]
val_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if isImage(f)]

# Create empty NumPy arrays to store the training and test images
X_train = np.empty((len(train_files), height, width), dtype=np.float32)
X_val = np.empty((len(val_files), height, width), dtype=np.float32)

# Load the training images into the X_train array
for i, filename in enumerate(train_files):
    image = Image.open(filename)
    X_train[i] = np.asarray(image, dtype=np.float32) / 255.0 # Normalize the pixel values to between 0 and 1

# Load the test images into the X_test array
for i, filename in enumerate(val_files):
    image = Image.open(filename)
    X_val[i] = np.asarray(image, dtype=np.float32) / 255.0 # Normalize the pixel values to between 0 and 1

# Print the shapes of the X_train and X_test arrays
print('X_train shape:', X_train.shape)
print('X_test shape:', X_val.shape)

# y_train = np_utils.to_categorical(y_train, nb_classes)
# y_test = np_utils.to_categorical(y_test, nb_classes)

# Create a convolutional autoencoder object
autoencoder = ConvolutionalAutoencoder()

# Train the autoencoder
autoencoder.train_model(X_train, X_val)

# Plot the loss
autoencoder.plot_loss()

autoencoder.plot_decoded_imgs(X_val)
