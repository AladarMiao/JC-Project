from DLModels import ConvolutionalAutoencoder
from DataPreprocessor.DataPreprocessor import DataPreprocessor
import tensorflow as tf
import json

print(tf.config.list_physical_devices('GPU'))
nb_classes = 10

# Open the JSON file for reading
with open('sample.json') as f:
    # Load the contents of the file into a variable
    data = json.load(f)

data_preprocessor = DataPreprocessor(data["is_image"], images_train_path=data["image_train_path"],
                                     images_validation_path=data["image_validation_path"])

print("Resize Training and Validation Set")
X_train, X_val = data_preprocessor.resize_images(data["new_width"], data["new_height"])

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
