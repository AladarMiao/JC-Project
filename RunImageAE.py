from DLModels import ConvolutionalAutoencoder
from ImageResizer import ImageResizer
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
nb_classes = 10

print("Resize Training Set")
train_resizer = ImageResizer()
X_train = train_resizer.resize_images()
width, height = train_resizer.new_width, train_resizer.new_height
print(width)
print(height)

print("Resize Validation Set")
val_resizer = ImageResizer()
X_val = val_resizer.resize_images()

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
