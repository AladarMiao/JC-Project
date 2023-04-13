from DLModels import ConvolutionalAutoencoder
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
nb_classes = 10

#Sample use case
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Create a convolutional autoencoder object
autoencoder = ConvolutionalAutoencoder()

# Train the autoencoder
autoencoder.train_model(X_train, X_test)

#Load pretrained
# autoencoder.load_model("my_model_20230412_153955.h5")

# Plot the loss
autoencoder.plot_loss()

autoencoder.plot_decoded_imgs(X_test)
