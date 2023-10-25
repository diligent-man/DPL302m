import os
import re
import shutil


class TextPreprocessor:
    def __init__(self):
        self.__data_path = ""
        self.__preprocessed_data_path = ""

    def set_data_path(self, path):
        self.__data_path = path

    def get_data_path(self):
        return self.__data_path

    def set_preprocessed_data_path(self, path):
        self.__preprocessed_data_path = path

    def get_preprocessed_data_path(self):
        return self.__preprocessed_data_path

    @staticmethod
    def __has_BOM(filename):
        # ref: https://codeverge.com/unicodeerror-utf-16-stream-does-not-start-with-bom
        with open(filename, 'rb') as f:
            initial_bytes = f.read(2)
        return initial_bytes in [b'\xFE\xFF', b'\xFF\xFE']

    @staticmethod
    def __save_preprocessed_data(preprocessed_data_path: str, filename: str, data: str) -> None:
        with open(preprocessed_data_path + '/' + filename, 'w', encoding='utf-8') as f:
            f.write(data)
        return None

    def split_into_sentence(self):
        for filename in sorted(os.listdir(self.__data_path)):
            if self.__has_BOM(self.__data_path + '/' + filename):
                with open(self.__data_path + '/' + filename, 'r', encoding='utf-16') as f:
                    # Split by paragraph
                    data = f.read().split('\n')
                    paragraph = [paragraph.strip() for paragraph in data if len(paragraph) != 0]
                    # Split by period
                    paragraph = [sentence.split('. ') for sentence in paragraph]
                    # Take only available sentence
                    sentences = [sentence for sentences in paragraph for sentence in sentences if len(sentence) != 0]
            else:
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
            self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)

    def remove_by_regex(self, regex_dict):
        for i in regex_dict:
            for filename in sorted(os.listdir(self.__data_path)):
                with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:            
                    data = []
                    for line in f:
                        line = re.sub(regex_dict[i][0], regex_dict[i][1], line)
                        # remove ws surrounding word
                        line = ' '.join([word.strip() for word in line.split()])
                        # if len line < 7, ignore it
                        if len(line.split()) <= 6:
                            continue
                        else:
                            data.append(line)
                    # save data
                    data = '\n'.join(data)
                    self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


def vn_preprocessing() -> None:
    category_ls = ['am_thuc', 'doi_song', 'du_lich', 'gia_dinh', 'the_gioi',
                   'giai_tri', 'giao_duc', 'khong_gian_song', 'loi_song',
                   'the_thao','thoi_su','thoi_trang']
    data_dir = '../Data'
    preprocessed_data_dir = '../Preprocessed_data'

    # split into singular sentence
    for category in category_ls:
        print(category)
        text_preprocessor = TextPreprocessor()
        text_preprocessor.set_data_path(data_dir + '/' + category)
        text_preprocessor.set_preprocessed_data_path(preprocessed_data_dir + '/' + category)

        # remove preprocessed data dir if it exists
        if category in os.listdir(preprocessed_data_dir):
            shutil.rmtree(path=preprocessed_data_dir + '/' + category)
        os.mkdir(path=preprocessed_data_dir + '/' + category, mode=0o777)

        # perform splitting
        text_preprocessor.split_into_sentence()
        text_preprocessor.set_data_path(text_preprocessor.get_preprocessed_data_path())

        regex_dict = {0: [r"[~!@#$%\^&\*()\_,，./<>\?;:：\"\[\]\{\}\\|“”0-9\+=]*", ""],  # punctuation_marks_and_numeral
                      1: [r"[-–]", ""], # hyphen & dash
                     }
        text_preprocessor.remove_by_regex(regex_dict)


def main() -> None:
    vn_preprocessing()
    return None


if __name__ == '__main__':
    main()
