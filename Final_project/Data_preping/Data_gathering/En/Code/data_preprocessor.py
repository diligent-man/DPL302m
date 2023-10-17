import os
import re
import shutil

class TextPreprocessor:
    def __init__(self):
        self.__data_path = None
        self.__preprocessed_data_path = None

    def set_data_path(self, path):
        self.__data_path = path

    def get_data_path(self):
        return self.__data_path
    def set_preprocessed_data_path(self, path):
        self.__preprocessed_data_path = path

    def get_preprocessed_data_path(self):
        return self.__preprocessed_data_path

    @staticmethod
    def save_preprocessed_data(preprocessed_data_path: str, filename: str, data: str) -> None:
        with open(preprocessed_data_path + '/' + filename, 'w') as f:
            f.write(data + '\n')
        return None


    def split_into_sentence(self):
        for filename in os.listdir(self.__data_path):

            # if self.has_BOM(self.__data_path + '/' + filename):
            print(filename)




def BBC_preprocessing(parent_path: str, original_path: str) -> None:
    # parent_path: ../Data/BBC
    os.chdir(parent_path)
    category_ls  = sorted(os.listdir())
    preprocessed_data_dir = '../../Preprocessed_data/BBC'

    # split into singular sentence
    for category in category_ls:
        text_preprocessor = TextPreprocessor()
        text_preprocessor.set_data_path(parent_path + '/' + category)
        text_preprocessor.set_preprocessed_data_path(preprocessed_data_dir + '/BBC/' + category)


        # remove preprocessed data dir if it exists
        if category in os.listdir(preprocessed_data_dir):
            shutil.rmtree(path=preprocessed_data_dir + '/' + category)
        os.mkdir(path=preprocessed_data_dir + '/' + category, mode=0o777)

        # text_preprocessor.split_into_sentence()



    os.chdir(original_path)
    return None


def Ted_talk() -> None:
    return None


def Wiki() -> None:
    return None    


def main() -> None:
    original_path = os.getcwd()
    BBC_preprocessing(parent_path='../Data/BBC', original_path=original_path)    
    return None


if __name__ == '__main__':
    main()