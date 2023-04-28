from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

class ResNet18(DLModel):

    def __init__(self):
        super().__init__()

    def define_model(self, input_shape, num_classes):
        # Input layer
        x = Input(shape=input_shape)

        # conv1
        conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.0001))(x)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(bn1)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(act1)

        # conv2_x
        conv2_1 = self.residual_block(pool1, filters=64)
        conv2_2 = self.residual_block(conv2_1, filters=64)

        # conv3_x
        conv3_1 = self.residual_block(conv2_2, filters=128, strides=(2, 2))
        conv3_2 = self.residual_block(conv3_1, filters=128)

        # conv4_x
        conv4_1 = self.residual_block(conv3_2, filters=256, strides=(2, 2))
        conv4_2 = self.residual_block(conv4_1, filters=256)

        # conv5_x
        conv5_1 = self.residual_block(conv4_2, filters=512, strides=(2, 2))
        conv5_2 = self.residual_block(conv5_1, filters=512)

        # output
        avg_pool = GlobalAveragePooling2D()(conv5_2)
        output = Dense(units=num_classes, activation='softmax')(avg_pool)

        # Create model
        model = Model(inputs=x, outputs=output)

        # Compile model
        optimizer = Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model
        self.model.summary()

    def residual_block(self, x, filters, strides=(1, 1)):
        res = x

        # First convolution layer of the block
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(bn1)

        # Second convolution layer of the block
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(act1)
        bn2 = BatchNormalization()(conv2)

        # Shortcut connection
        if strides != (1, 1) or res.shape[3] != filters:
            res = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides
