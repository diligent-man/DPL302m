import scipy
import h5py
import math
import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

from tensorflow import keras as kr
from tensorflow.python.framework import ops

from cnn_utils import load_dataset, rand_mini_batches, OHE


################################################################################
def preprocessing(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, y_test: np.ndarray, filename: str) -> np.ndarray:
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)

    if filename == 'happy':
        y_train = y_train.T
        y_test = y_test.T
    elif filename == 'signs':
        y_train = OHE(y_train, num_of_class=6).T
        y_test = OHE(y_test, num_of_class=6).T

    return x_train, y_train, x_test, y_test


def dataset_inspection(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray) -> None:
    print ("number of training examples = " + str(x_train.shape[0]))
    print ("number of test examples = " + str(x_test.shape[0])); print()
    print ("X_train shape: " + str(x_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_test shape: " + str(x_test.shape))
    print ("Y_test shape: " + str(y_test.shape))

    '''
    training = 600
    test = 150
    
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)
    '''
    return None
################################################################################
def happyModel():
    """
    Arch: ZEROPAD2D -> ReLU(BatchNorm(CONV2D)) -> MAXPOOL -> FLATTEN -> DENSE
        -> 5 layers
    
    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 

    ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers

    kr.layers.ZeroPadding2D(padding, input_shape)
        padding:
            + int: applied symmetrically to height and width
            + tuple of 2 ints: interpreted as (height, width)
            + tuple of 2 tuples of 2 ints: interpreted as  ((top_pad, bottom_pad),
                                                            (left_pad, right_pad))

    tf.keras.layers.BatchNormalization(axis=-1,
                                      momentum=0.99,
                                      epsilon=0.001,
                                      center=True,
                                      scale=True,
                                      beta_initializer='zeros',
                                      gamma_initializer='ones',
                                      moving_mean_initializer='zeros',
                                      moving_variance_initializer='ones',
                                      beta_regularizer=None,
                                      gamma_regularizer=None, ...)
        where:
            momentum: exponentially weighted average


        formula: gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta, where
            gamma: learned scaling factor -> disbale by scale=False
            epsilon: for zero division
            beta: learned offset factor -> disable by center=False
    """
    model = tf.keras.Sequential([
        kr.layers.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),
        kr.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding='valid'),
        kr.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                      center=True, scale=True,
                                      beta_initializer='zeros', gamma_initializer='ones',
                                      moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        kr.layers.ReLU(),
        kr.layers.MaxPool2D(pool_size=(2,2), pool_function='max', strides=1),
        kr.layers.Flatten(data_format="channels_last"),
        kr.layers.Dense(units=1, activation='sigmoid', use_bias=True,
                        kernel_initializer='glorot_uniform', bias_initializer='zeros')
        # units=1 due to binary classification
    ])
    return model


def Exercise_1(x_train, y_train, x_test, y_test):

    happy_model = happyModel()
    happy_model.summary()

    happy_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc', 'mae'], jit_compile=True)
    happy_model.fit(x_train, y_train, epochs=5, batch_size=50)
    happy_model.evaluate(x_test, y_test, batch_size=50)

    predictions = happy_model.predict(x_train[0].reshape(1, 64, 64, 3))
    print(predictions)

################################################################################

def cnn_model(input_shape):
    """
    Arch: ReLU(CONV2D) -> MAXPOOL -> ReLU(CONV2D) -> MAXPOOL -> FLATTEN -> DENSE
        -> 6 layers

    Conv2D: Use 8 4x4 filters, stride 1, padding is "SAME" -> out shape = inp shape
    MaxPool2D: Use an 8x8 filter size and stride 8, padding is "SAME"

    Conv2D: Use 16 2x2 filters, stride 1, padding is "SAME"
    MaxPool2D: Use a 4x4 filter size and stride 4, padding is "SAME"
    Flatten the previous output.
    Fully-connected (Dense) layer: Apply a fully connected layer with 6 neurons and a softmax activation.


    Returns:
    model -- TF Keras model (object containing the information for the entire training process)

    """
    inputs = kr.Input(shape=input_shape)
    A1 = kr.layers.Conv2D(filters=32, kernel_size=(4,4), strides=1,
                          padding='same', activation='relu')(inputs)
    Pooling1 = kr.layers.MaxPooling2D(pool_size=(4,4),strides=4, padding='same')(A1)

    A2 = kr.layers.Conv2D(filters=32,kernel_size=(2,2), strides=1, padding='same',
                          activation='relu')(Pooling1)
    Pooling2 = kr.layers.MaxPooling2D(pool_size=(4,4), strides=4, padding='same')(A2)

    flatten = kr.layers.Flatten()(Pooling2)
    outputs = kr.layers.Dense(units=6, activation='softmax')(flatten)

    model = kr.Model(inputs=inputs, outputs=outputs)
    return model


def Exercise_2(x_train, y_train, x_test, y_test):
    signs_model = cnn_model(input_shape=x_train.shape[1:])
    signs_model.summary()

    signs_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc', 'mae'], jit_compile=True)
    signs_model.fit(x_train, y_train, epochs=30, batch_size=1)
    signs_model.evaluate(x_test, y_test, batch_size=1)
    return None


################################################################################
def main() -> None:
    """
    This assignment will build CNN for binary & multiclass classification
    Explain sequential & functional API in Keras

    Sequential API is less flexible than Functional API but straightforward

    Functional API can tackle with non-linear topology, shared layers -> a graph

    """
    # x_train, y_train, x_test, y_test, classes = load_dataset('train_happy.h5', 'test_happy.h5')
    # x_train, y_train, x_test, y_test = preprocessing(x_train, y_train, x_test, y_test, filename='happy')
    # dataset_inspection(x_train, y_train, x_test, y_test)
   
    # index = 12
    # plt.imshow(x_train[index])
    # plt.show()
    # Exercise_1(x_train, y_train, x_test, y_test)


    # x_train, y_train, x_test, y_test, classes = load_dataset('train_signs.h5', 'test_signs.h5')
    # x_train, y_train, x_test, y_test = preprocessing(x_train, y_train, x_test, y_test, filename='signs')
    # dataset_inspection(x_train, y_train, x_test, y_test)
    # Exercise_2(x_train, y_train, x_test, y_test)
    return None


if __name__ == '__main__':
    main()