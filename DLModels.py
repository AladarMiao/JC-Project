from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv2D, MaxPooling2D, UpSampling2D, Cropping1D
from keras.callbacks import ModelCheckpoint
import datetime as dt
import matplotlib.pyplot as plt

class DLModel:
    def __init__(self):
        self.model = None
        self.history = None

    def train(self):
        pass

    def define_model(self):
        pass

    def save_model(self):

        # Get the current time as a string
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Construct the filename with the timestamp
        filename = f'my_model_{timestamp}.h5'

        # Save the model to the file with the timestamped name
        self.model.save(filename)

        # Print message indicating the model has been saved
        print(f'Model saved to {filename}')

class AnomalyDetectionModel(DLModel):
    def __init__(self):
        super().__init__()

class ConvolutionalAutoencoder(AnomalyDetectionModel):

    def __init__(self):
        super().__init__()

    def define_model(self, height, width):
        #Feel free to adjust this accordingly
        x = Input(shape=(height, width, 1))

        # Encoder
        conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
        conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
        conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
        h = MaxPooling2D((2, 2), padding='same')(conv1_3)

        # Decoder
        conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
        up1 = UpSampling2D((2, 2))(conv2_1)
        conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
        up2 = UpSampling2D((2, 2))(conv2_2)
        conv2_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(up2)
        up3 = UpSampling2D((2, 2))(conv2_3)
        r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

        autoencoder = Model(inputs=x, outputs=r)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model = autoencoder
        self.model.summary()

    def train_model(self, X_train, X_test, checkpoint_filepath='model_checkpoint.h5'):
        self.height = X_train.shape[1]
        self.width = X_train.shape[2]
        self.define_model(self.height, self.width)
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Train the model with checkpoints
        self.history = self.model.fit(X_train, X_train, batch_size=128,
                                      epochs=10, verbose=1, validation_data=(X_test, X_test))

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

    def plot_decoded_imgs(self, X_test):
        decoded_imgs = self.model.predict(X_test)
        n = 10
        plt.figure(figsize=(20, 6))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i+1)
            plt.imshow(X_test[i].reshape(self.height, self.width))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            # display reconstruction
            ax = plt.subplot(3, n, i+n+1)
            plt.imshow(decoded_imgs[i].reshape(self.height, self.width))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

    def load_model(self, model_path):
        self.model = load_model(model_path)
        self.model.summary()
