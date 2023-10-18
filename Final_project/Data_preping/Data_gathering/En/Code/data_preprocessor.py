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
    def has_BOM(filename):
        # ref: https://codeverge.com/unicodeerror-utf-16-stream-does-not-start-with-bom
        with open(filename, 'rb') as f:
            initial_bytes = f.read(2)
        return initial_bytes in [b'\xFE\xFF', b'\xFF\xFE']

    @staticmethod
    def save_preprocessed_data(preprocessed_data_path: str, filename: str, data: str) -> None:
        with open(preprocessed_data_path + '/' + filename, 'w', encoding='utf-8') as f:
            f.write(data + '\n')
        return None

    def split_into_sentence(self):
        for filename in os.listdir(self.__data_path):
            with open(self.__data_path + '/' + filename, 'r', encoding='utf-8') as f:
                # Split by paragraph
                data = f.read().split('\n')
                paragraph = [paragraph.strip() for paragraph in data if len(paragraph) != 0]
                # Split by period
                paragraph = [sentence.split('. ') for sentence in paragraph]
                # Take only available sentence
                sentences = [sentence for sentences in paragraph for sentence in sentences if len(sentence) != 0]

            # save data to preprocessed_data folder
            data = '\n'.join(sentences)

            self.save_preprocessed_data(self.__preprocessed_data_path, filename, data)

    def remove_special_chars(self):
        for filename in sorted(os.listdir(self.__data_path)):
            with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                data = ''
                for line in f:
                    # remove special chars
                    line = re.sub(r'[^a-zA-Z ]', ' ', line.lower())
                    if line.startswith('-'):
                        line = line[1:]

                    # Loại bỏ khoảng trắng đầu dòng và giữ một khoảng trắng giữa các từ
                    line = ' '.join(line.strip().split())

                    # if len line < 5, ignore it
                    if len(line) >= 5:
                        print(line, len(line))
                        data = data + line + '\n'

                # save to file
                self.save_preprocessed_data(self.__preprocessed_data_path, filename, data)


def en_preprocessing() -> None:
    category_ls = ['Business', 'Entertainment', 'Health', 'Sport', 'Style',
                'Tech', 'Ted_talk', 'Travel', 'Weather',
                'Wiki','World']
    data_dir = 'D:/FPT/data/Data'
    preprocessed_data_dir = "D:/FPT/data/Preprocessed_data"

    # split into singular sentence
    for category in category_ls:
        text_preprocessor = TextPreprocessor()
        text_preprocessor.set_data_path(data_dir + '/' + category)
        text_preprocessor.set_preprocessed_data_path(preprocessed_data_dir + '/' + category)

        # remove preprocessed data dir if it exists
        if category in os.listdir(preprocessed_data_dir):
            shutil.rmtree(path=preprocessed_data_dir + '/' + category)
        os.mkdir(path=preprocessed_data_dir + '/' + category, mode=0o777)

        # perform splitting
        text_preprocessor.split_into_sentence()


    # remove punctuation marks
    for category in category_ls:
        text_preprocessor = TextPreprocessor()
        text_preprocessor.set_data_path(preprocessed_data_dir + '/' + category)
        text_preprocessor.set_preprocessed_data_path(preprocessed_data_dir + '/' + category)
        text_preprocessor.remove_special_chars()


def main() -> None:
    en_preprocessing()
    return None


if __name__ == '__main__':
    main()
