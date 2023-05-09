import tensorflow as tf
import json

from Constants import get_clustering_class, get_dim_reduction_class
from DataPreprocessor.DataPreprocessor import DataPreprocessor, isImage

import os
import shutil
import keras
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from keras.utils import custom_object_scope

from ParameterAnalyzer.SHAPDeep import SHAPDeep

# def copy_images(source_folder, target_folder):
#     # Create the target folder if it doesn't exist
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#
#     # Loop through all files and directories in the source folder
#     for file_or_dirname in os.listdir(source_folder):
#         # Create the full path to the current item
#         current_item = os.path.join(source_folder, file_or_dirname)
#
#         # Check if the current item is a file
#         if os.path.isfile(current_item):
#             # Check if the file is an image (based on file extension)
#             if isImage(file_or_dirname.lower()):
#                 # Create the full path to the target file
#                 target_file = os.path.join(target_folder, file_or_dirname)
#                 # Copy the file to the target folder
#                 shutil.copyfile(current_item, target_file)
#         # Check if the current item is a directory
#         elif os.path.isdir(current_item):
#             # Recursively call this function on the subdirectory
#             copy_images(current_item, target_folder)
#
# # First, copy all images from the testing set's subdirectory to another folder
#
# copy_images('/mnt/c/Users/alada/Downloads/data/test', '/mnt/c/Users/alada/Downloads/AETestData')

# Open the JSON file for reading
with open('JsonInputs/AutoencoderTest.json') as f:
    # Load the contents of the file into a variable
    data = json.load(f)

preprocessor_parameters = data["preprocessor_parameters"]
data_preprocessor = DataPreprocessor(preprocessor_parameters["is_image"],
                                     image_train_path=preprocessor_parameters["image_train_path"],
                                     image_validation_path=preprocessor_parameters["image_validation_path"])

print("Resizing Test Set")
X_test, _ = data_preprocessor.resize_images(preprocessor_parameters.get("new_width", 500),
                                                 preprocessor_parameters.get("new_height", 500))

# Define the custom loss function
def custom_loss(y_true, y_pred, a1=0.86, a2=0.14):
    # Compute the MS-SSIM loss
    ms_ssim_loss = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
    ms_ssim_loss = tf.reduce_mean(ms_ssim_loss)

    # Compute the mean squared error between the two images and take the mean
    mse_loss = mean_squared_error(y_true, y_pred)
    mse_loss = tf.reduce_mean(mse_loss)

    # Combine the losses with the given weights
    loss = a1 * ms_ssim_loss + a2 * mse_loss
    return loss

# Load pre-trained model
with custom_object_scope({'custom_loss': custom_loss}):
    model = keras.models.load_model("/mnt/c/Users/alada/JC_code/hao_model.h5")

# Get the output of a specific layer
layer_name = "max_pooling2d_2"
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("max_pooling2d_2").output)


# Use the intermediate layer model to get the output of the specific layer
input_data = X_test # provide input data in the appropriate format
print("input_data.shape", input_data.shape)
output_data = intermediate_layer_model.predict(input_data)
print("output_data.shape", output_data.shape)

# Flatten the output_data to a 2D array
output_data = output_data.reshape(output_data.shape[0], -1)

clustering_parameters = data.get("clustering_parameters", {})
print("clustering_parameters",  clustering_parameters)

if clustering_parameters.get("algo"):
    clustering = get_clustering_class(clustering_parameters.get("algo"),
                                      n_clusters=clustering_parameters.get("num_clusters"))
    print("Clustering after passing through the Autoencoder")
    clustering.cluster(output_data)

shap_deep_analyzer = SHAPDeep(intermediate_layer_model, X_test) # TODO: running into errors, perhaps due to version conflicts
shap_deep_analyzer.plot_explainer(X_test)