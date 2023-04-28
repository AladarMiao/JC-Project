from Constants import get_dl_class
from DataPreprocessor.DataPreprocessor import DataPreprocessor
import tensorflow as tf
import json

print(tf.config.list_physical_devices('GPU'))

# Open the JSON file for reading
with open('RunImageAE.json') as f:
    # Load the contents of the file into a variable
    data = json.load(f)

preprocessor_parameters = data["preprocessor_parameters"]
data_preprocessor = DataPreprocessor(preprocessor_parameters["is_image"],
                                     images_train_path=preprocessor_parameters["image_train_path"],
                                     images_validation_path=preprocessor_parameters["image_validation_path"])

print("Resize Training and Validation Set")
X_train, X_val = data_preprocessor.resize_images(preprocessor_parameters.get("new_width", 500),
                                                 preprocessor_parameters.get("new_height", 500))

# Print the shapes of the X_train and X_test arrays
print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)

dl_parameters = data["DL_parameters"]
# Create a convolutional autoencoder object
autoencoder = get_dl_class(dl_parameters["model_type"])

# Train the autoencoder
autoencoder.train_model(X_train,
                        X_val,
                        dl_parameters.get("batch_size", 128),
                        dl_parameters.get("epochs", 10),
                        dl_parameters.get("checkpoint_filepath", "model_checkpoint.h5"),
                        dl_parameters.get("pretrained_model_path", None))

# Plot the loss
autoencoder.plot_loss()

autoencoder.plot_decoded_imgs(X_val,
                              preprocessor_parameters.get("new_height", 500),
                              preprocessor_parameters.get("new_width", 500))
