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

def load_and_preprocess_happy_dataset():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

    X_train = X_train_orig / 255.0
    X_test = X_test_orig / 255.0

    # Chuyển đổi kết quả thành ma trận chuyển vị
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    return X_train, Y_train, X_test, Y_test

def create_happy_model():
    model = tf.keras.Sequential([
        tfl.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),
        tfl.Conv2D(filters=32, kernel_size=7, strides=(1, 1), input_shape=[64, 64, 3]),
        tfl.BatchNormalization(axis=3),
        tfl.ReLU(max_value=None, negative_slope=0.0, threshold=0.0),
        tfl.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
        tfl.Flatten(),
        tfl.Dense(1, activation="sigmoid")
    ])

    return model

def train_and_evaluate_happy_model(model, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=16):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    model.evaluate(X_test, Y_test)


X_train, Y_train, X_test, Y_test = load_and_preprocess_happy_dataset()
happy_model = create_happy_model()
train_and_evaluate_happy_model(happy_model, X_train, Y_train, X_test, Y_test)
