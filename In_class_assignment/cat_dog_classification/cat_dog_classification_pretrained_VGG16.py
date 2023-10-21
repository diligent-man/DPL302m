import os
import cv2 as cv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as kr
from matplotlib import pyplot as plt


def visualize_data_set(class_names: str, train_set) -> None:
    plt.figure(figsize=(10, 10))
    for images, labels in train_set.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    return None


def visualize_augmented_data(train_set, data_augmentation) -> None:
    for image, labels in train_set.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # tf.expand_dims: Add an outer "batch" dimension to a single element.
            # e.g: inp=(160, 160, 3) -> inp=(1, 160, 160, 3)
            augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')
    plt.show()
    return None


def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmenter = kr.Sequential()
    data_augmenter.add(kr.layers.RandomFlip(mode='horizontal'))
    data_augmenter.add(kr.layers.RandomRotation(factor=1, interpolation='bilinear'))
    return data_augmenter


def model(img_size, data_augmenter=data_augmenter()):
    input_shape = img_size
    # Inp layer
    inputs = kr.layers.Input(shape=input_shape)
    inputs = kr.applications.vgg16.preprocess_input(inputs) # tailor to vgg16 inp layer
    inputs = data_augmenter(inputs)

    # Freeze VGG16
    base_model = kr.applications.vgg16.VGG16()
    model = kr.Sequential()
    model.add(inputs)
    for layer in base_model.layers[:-2]:
        layer.trainable = False
        model.add(layer)






    # Add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    # x = kr.layers.
    # x = kr.layers.Dropout(0.2)(x)
    # x = kr.layers.Dense(4096 - 1024)(x)
    #
    #
    # x = kr.layers.Dense(4096 - 1024)(x)
    #
    #
    # x = kr.layers.Dropout(0.2)(x)
    # outputs = kr.layers.Dense(2)(x)
    # return kr.Model(inputs=inputs, outputs=outputs)

    model.summary()
    return model



def main() -> None:
    batch_size = 10; img_size = (224, 224); dir = "preprocessed_data/train"
    train_set = kr.preprocessing.image_dataset_from_directory(directory=dir, shuffle=True, labels="inferred",
                                                              batch_size=batch_size, image_size=img_size,
                                                              validation_split=0.1, subset='training',  seed=123,
                                                              )
    class_names = train_set.class_names
    train_set = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # visualize_data_set(class_names=class_names, train_set=train_set)

    data_augmentation = data_augmenter()
    # visualize_augmented_data(train_set, data_augmentation)

    model = model(img_size)


    return None

if __name__ == '__main__':
    main()
