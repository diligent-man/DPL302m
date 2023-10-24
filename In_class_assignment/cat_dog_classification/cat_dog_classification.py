import os
import cv2 as cv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def rescale(img, size=224):
    r, g, b = cv.split(img)
    r = cv.resize(src=r, dsize=(size, size), interpolation=cv.INTER_LANCZOS4)
    g = cv.resize(src=g, dsize=(size, size), interpolation=cv.INTER_LANCZOS4)
    b = cv.resize(src=b, dsize=(size, size), interpolation=cv.INTER_LANCZOS4)
    img = cv.merge([r, g, b])
    return img


def preprocessing() -> None:
    # preprocess train set
    train_path = "cat_dog_classification/train/" # 25000
    preprocessed_cat_train_path = "cat_dog_classification/preprocessed_data/train/cat/"
    preprocessed_dog_train_path = "cat_dog_classification/preprocessed_data/train/dog/"

    cat_counter = 0
    dog_counter = 0
    for filename in os.listdir(train_path):
        print(filename)
        img = cv.imread(train_path + filename, cv.COLOR_BGR2RGB)
        img = rescale(img)

        if "cat" in filename:
            name = "cat_" + str(cat_counter) + ".jpg"
            cv.imwrite(preprocessed_cat_train_path + name, img)
            cat_counter += 1
        elif "dog" in filename:
            name = "dog_" + str(dog_counter) + ".jpg"
            cv.imwrite(preprocessed_dog_train_path + name, img)
            dog_counter += 1

    # preprocess test set
    counter = 0
    test_path = "cat_dog_classification/test/"   # 12500
    preprocessed_test_path = "cat_dog_classification/preprocessed_data/test/"
    for filename in os.listdir(test_path):
        print(filename)
        img = cv.imread(test_path + filename, cv.COLOR_BGR2RGB)
        img = rescale(img)

        cv.imwrite(filename, img)
    return None

def main() -> None:
    # preprocessing()

    inputs = kr.utils.image_dataset_from_directory(
        directory=""
    )
    return None

if __name__ == '__main__':
    main()
