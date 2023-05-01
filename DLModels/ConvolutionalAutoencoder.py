from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv2D, MaxPooling2D, UpSampling2D, \
    Cropping1D, BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
import tensorflow as tf

from DLModels.DLModels import DLModel


class ConvolutionalAutoencoder(DLModel):

    def __init__(self):
        super().__init__()

    def define_model(self, height, width, a1=0.86, a2=0.14, reg=0.01):
        # Input layer
        x = Input(shape=(height, width, 1))

        # Encoder
        conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(x)
        bn1_1 = BatchNormalization()(conv1_1)
        pool1 = MaxPooling2D((2, 2), padding='same')(bn1_1)
        conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(pool1)
        bn1_2 = BatchNormalization()(conv1_2)
        pool2 = MaxPooling2D((2, 2), padding='same')(bn1_2)
        conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(pool2)
        bn1_3 = BatchNormalization()(conv1_3)
        h = MaxPooling2D((2, 2), padding='same')(bn1_3)

        # Decoder
        conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(h)
        bn2_1 = BatchNormalization()(conv2_1)
        up1 = UpSampling2D((2, 2))(bn2_1)
        conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(up1)
        bn2_2 = BatchNormalization()(conv2_2)
        up2 = UpSampling2D((2, 2))(bn2_2)
        conv2_3 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(up2)
        bn2_3 = BatchNormalization()(conv2_3)
        up3 = UpSampling2D((2, 2))(bn2_3)
        r = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(up3)

        # Create model
        autoencoder = Model(inputs=x, outputs=r)

        # Define custom loss function
        def custom_loss(y_true, y_pred):
            # Compute the MS-SSIM loss
            ms_ssim_loss = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
            ms_ssim_loss = tf.reduce_mean(ms_ssim_loss)

            # Round the loss to three decimal places, due to numerical instability issues
            ms_ssim_loss_rounded = tf.round(ms_ssim_loss * 1000) / 1000

            # Compute the mean squared error between the two images and take the mean
            mse_loss = mean_squared_error(y_true, y_pred)
            mse_loss = tf.reduce_mean(mse_loss)
            mse_loss_rounded = tf.round(mse_loss * 1000) / 1000

            # Combine the losses with the given weights
            loss = a1 * ms_ssim_loss_rounded + a2 * mse_loss_rounded
            return loss

        # Compile model with custom loss function
        autoencoder.compile(optimizer='adam', loss=custom_loss)
        self.model = autoencoder
        self.model.summary()

    def train_model(self, x_train, X_val, batch_size, epochs, checkpoint_filepath, pretrained_model_path):
        if pretrained_model_path:
            self.model = load_model(pretrained_model_path)
        else:
            self.height = x_train.shape[1]
            self.width = x_train.shape[2]
            self.define_model(self.height, self.width)
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')

        # Train the model with checkpoints
        self.history = self.model.fit(x_train, x_train, batch_size=batch_size,
                                      epochs=epochs, verbose=1, validation_data=(X_val, X_val))

        self.save_model()

    def plot_loss(self):
        if not self.history:
            print("Please train your model to get your model history")
            return
        print(self.history.history.keys())

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        # save the plot to a PNG file
        plt.savefig('plot_loss.png')

    def plot_decoded_imgs(self, X_test, height, width):
        decoded_imgs = self.model.predict(X_test)
        n = 20
        plt.figure(figsize=(20, 6))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(X_test[i].reshape(height, width))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(height, width))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
        plt.savefig('plot_decoded_imgs.png')
