import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator


def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    model = tf.keras.Sequential([
        #ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        tf.keras.layers.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3), data_format="channels_last"),
        #Conv2D with 32 7x7 filters and stride of 1
        tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name='conv0'),
        #BatchNormalization for axis 3
        tf.keras.layers.BatchNormalization(axis=3, name='bn0'),
        # ReLU
        tf.keras.layers.ReLU(
            max_value=None, negative_slope=0.0, threshold=0.0),
        # Max Pooling 2D with default parameters
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pool0'),
        # Flatten layer
        tf.keras.layers.Flatten(),
        # Dense layer with 1 unit for output & 'sigmoid' activation
        tf.keras.layers.Dense(1, activation='sigmoid', name='fc'),
    ])

    return model

def Exercise_1():
    happy_model = happyModel()
    # Print a summary for each layer
    for layer in summary(happy_model):
        print(layer)

    output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
                ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
                ['BatchNormalization', (None, 64, 64, 32), 128],
                ['ReLU', (None, 64, 64, 32), 0],
                ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
                ['Flatten', (None, 32768), 0],
                ['Dense', (None, 1), 32769, 'sigmoid']]

    comparator(summary(happy_model), output)


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    input_img = tf.keras.Input(shape=input_shape)
    # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='same')(input_img)
    # RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)
    # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same')(P1)
    # RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)
    # FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    # Dense layer
    # 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
    outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def Exercise_2():
    conv_model = convolutional_model((64, 64, 3))
    conv_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    conv_model.summary()

    output = [['InputLayer', [(None, 64, 64, 3)], 0],
              ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
              ['ReLU', (None, 64, 64, 8), 0],
              ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
              ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
              ['ReLU', (None, 8, 8, 16), 0],
              ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
              ['Flatten', (None, 64), 0],
              ['Dense', (None, 6), 390, 'softmax']]

    comparator(summary(conv_model), output)

#Train the Model
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)







#########################################################################################################################
def main() -> None:
    # Exercise_1()
    Exercise_2()
    # Exercise_3()
    # Exercise_4()
    # Exercise_5()
    # Exercise_6()
    # Exercise_7()
    return None
if __name__ == '__main__':
    main()

