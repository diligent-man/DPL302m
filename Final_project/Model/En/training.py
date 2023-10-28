import os

from sklearn.utils import shuffle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataLoader():
    def __init__(self):
        self.__path = ""


    def set_path(self, path):
        self.__path = path


    def get_path(self):
        return self.__path


    def load_dataset(self) -> tuple:
        # In-memory loading
        metadata = os.listdir(self.__path)
        train_file, test_file= train_test_split(metadata, test_size=0.2, random_state=1234, shuffle=True)# list of filenames

        x_train = []
        y_train = []
        # Read data into mem
        for file in train_file:
            filename = self.__path + '/' + file
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    x_train.append(line.split('|')[0].strip())
                    y_train.append(line.split('|')[1].strip())

        x_test = []
        y_test = []
        for file in test_file:
            filename = self.__path + '/' + file
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    x_test.append(line.split('|')[0].strip())
                    y_test.append(line.split('|')[1].strip())
        return x_train, y_train, x_test, y_test



def main() -> None:
    path = '../../Wrong_word_generator/En_noised_data'
    
    data_loader = DataLoader()
    data_loader.set_path(path)
    x_train, y_train, x_test, y_test = data_loader.load_dataset()
    print(len(x_train), len(y_train), len(x_test), len(y_test))



    return None


if __name__ == '__main__':

    main()


