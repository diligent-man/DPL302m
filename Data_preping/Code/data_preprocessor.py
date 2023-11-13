import os
import re
import string
import shutil
from nltk.corpus import stopwords


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
    def __expand_contractions(text, contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]

        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        return contractions_re.sub(replace, text)


    @staticmethod
    def __save_preprocessed_data(preprocessed_data_path, filename: str, data: str) -> None:
        with open(preprocessed_data_path + '/' + filename, 'w', encoding='utf-8') as f:
            f.write(data)
        return None


    def split_into_sentence(self) -> None:
        for filename in sorted(os.listdir(self.__data_path)):
            with open(self.__data_path + '/' + filename, 'r', encoding='utf-8') as f:
                # Split by paragraph
                data = f.read().split('\n')
                paragraph = [paragraph.strip() for paragraph in data if len(paragraph) != 0]
                # Split by period
                paragraph = [sentence.split('. ') for sentence in paragraph]
                # Take only available sentence
                sentences = [sentence for sentences in paragraph for sentence in sentences if len(sentence) != 0]
                # Remove non-breaking space characters (0xA0)
                sentences = [sentence.replace('\xa0', ' ') for sentence in sentences]
            # save data
            data = '\n'.join(sentences)
            self.__save_preprocessed_data(self.__preprocessed_data_path,  filename, data)
        return None


    def decapitalize(self) -> None:
        for filename in sorted(os.listdir(self.__data_path)):
            with open(self.__data_path + '/' + filename, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    data.append(line.lower())
                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None



    def expand_contraction(self) -> None:
        # Disambiguation b/t would & had
        #                    is & has
        # will be covered later on
        contractions_dict = {"i'll": "i will",
                             "you'll": "you will",
                             "we'll": "we will",
                             "they'll": "they will",
                             "he'll": "he will",
                             "she'll": "she will",
                             "it'll": "it will",

                             "i've": "i have",
                             "you've": "you have",
                             "they've": "they have",
                             "we've": "we have",
                             "might've": "might have",
                             "may've": "may have",
                             "must've": "must have",
                             "could've": "could have",
                             "would've": "would have",
                             "should've": " should have",
                             "who've": "who have",
                             "what've": "what have",
                             "where've": "where have",
                             "when've": "when have",
                             "why've": "why have",

                             "mayn't": "may not",
                             "mightn't": "might not",
                             "mustn't": "must not",
                             "can't": "can not",
                             "couldn't": "could not",
                             "shouldn't": "should not",
                             "wouldn't": "would not",
                             "needn't": "need not",
                             "shan't": "shall not",
                             "don't": "do not",
                             "doesn't": "does not",

                             "who'd": "who would",
                             "what'd": "what would",
                             "where'd": "where would",
                             "when'd": "when would",
                             "why'd": "why would",
                             "that'd": "that would",
                             "there'd": "there would",
                             "i'd": "i would",
                             "you'd": "you would",
                             "we'd": "we would",
                             "they'd": "they would",
                             "he'd": "he would",
                             "she'd": "she would",
                             "it'd": "it would",

                             "what's": "what is",
                             "where's": "where is",
                             "when's": "when is",
                             "why's": "why is",
                             "that's": "that is",
                             "there's": "there is",

                             "i'm": "i am",
                             "he's": "he is",
                             "she's": "she is",
                             "it's": "it is",
                             "we're": "we are",
                             "you're": "you are",
                             "they're": "they are",
                             "these're": "these are",
                             "those're": "those are",
                             }
        for filename in sorted(os.listdir(self.__data_path)):
            with open(self.__data_path + '/' + filename, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    expanded_words = [self.__expand_contractions(word, contractions_dict) for word in line[:-1].split(" ")]
                    expanded_line = " ".join(expanded_words)
                    data.append(expanded_line.strip())
                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


    def remove_by_regex(self, regex_dict) -> None:
        for i in regex_dict:
            for filename in sorted(os.listdir(self.__data_path)):
                with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                    data = []
                    for line in f:
                        line = re.sub(regex_dict[i][0], regex_dict[i][1], line)
                        # remove ws surrounding word
                        line = " ".join([word.strip() for word in line.split(" ")])
                        data.append(line)
                    # save data
                    data = '\n'.join(data)
                    self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


    def remove_stop_word(self) -> None:
        for filename in sorted(os.listdir(self.__data_path)):
            with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                data = []
                stop_words = set(stopwords.words('english'))
                for line in f:
                    filtered_sentence = [word.strip() for word in line.split(" ") if not word in stop_words]
                    data.append(" ".join(filtered_sentence).strip())

                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


    def verify_sequence_length(self) -> None:
        for filename in sorted(os.listdir(self.__data_path)):
            with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    line = line[:-1]
                    # Ignore len(line) < 7
                    if len(line.split(' ')) > 7:
                        if len(line.split(' ')) > 40:
                            line = line.split(' ')[:40]
                            data.append(" ".join(line).strip())
                        else:
                            data.append(line.strip())
                    else:
                        continue
                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


    def verify_word_length(self) -> None:
        for filename in sorted(os.listdir(self.__data_path)):
            with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    line = line[:-1]
                    line = [word for word in line.split(" ") if len(word) > 0]
                    data.append(" ".join(line))
                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


def en_preprocessing() -> None:
    os.mkdir(path="../Preprocessed_data", mode=0o777)
    os.mkdir(path="../Preprocessed_data/BBC", mode=0o777)
    # os.mkdir(path="../Preprocessed_data/GPT", mode=0o777)
    os.mkdir(path="../Preprocessed_data/Ted_talk", mode=0o777)

    metadata = {"category_ls_1": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Tech', 'Travel', 'Weather', 'World'],
                # "category_ls_2": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Travel', 'Weather'],
                "category_ls_3": ['Ted_talk'],
                "data_dir_1": ['../Data/BBC'],
                # "data_dir_2": ['../Data/GPT'],
                "data_dir_3": ['../Data'],
                "preprocessed_data_dir_1": ["../Preprocessed_data/BBC"],
                # "preprocessed_data_dir_2": ["../Preprocessed_data/GPT"],
                "preprocessed_data_dir_3": ["../Preprocessed_data"]
                }

    # split into singular sentence
    index_1 = index_2 = index_3 = 0
    for i in range(len(metadata["category_ls_1"]) + len(metadata["category_ls_3"])):
        text_preprocessor = TextPreprocessor()

        # choosing what category will be processed
        if i < len(metadata["category_ls_1"]):
            flag = "1"
            category = metadata["category_ls_" + flag][index_1]
            index_1 += 1

        # elif i < len(metadata["category_ls_1"]) + len(metadata["category_ls_2"]):
        #     flag = "2"
        #     category = metadata["category_ls_" + flag][index_2]
        #     index_2 += 1

        elif i < len(metadata["category_ls_1"]) + len(metadata["category_ls_3"]):
            flag = "3"
            category = metadata["category_ls_" + flag][index_3]
            index_3 += 1

        data_dir = metadata["data_dir_" + flag][0]
        preprocessed_data_dir = metadata["preprocessed_data_dir_" + flag][0]
        text_preprocessor.set_data_path(data_dir + '/' + category)
        text_preprocessor.set_preprocessed_data_path(preprocessed_data_dir + '/' + category)
        print(text_preprocessor.get_data_path())

        # remove preprocessed data dir if it exists
        if category in os.listdir(preprocessed_data_dir):
            shutil.rmtree(path=preprocessed_data_dir + '/' + category)
        os.mkdir(path=preprocessed_data_dir + '/' + category, mode=0o777)

        text_preprocessor.split_into_sentence()
        text_preprocessor.set_data_path(text_preprocessor.get_preprocessed_data_path())

        text_preprocessor.decapitalize()
        regex_dict = {0: [r"[^a-z0-9-–'’\s]", ""],  # remove non-Latin chars
                      1: [r"’", "'"],
                      2: [r"'s", ""],  # possession
                      3: [r"[–-]", " "],  # dash -> hyphen
                      4: [r"[^a-z]-[^a-z]*", " "],  # remove adrift hyphen
                      5: [r"[0-9]+th", ""],  # remove century
                      6: [r"[0-9]+s", ""],
                      7: [r"[0-9]", ""],
                      8: [r"isbn", ""]
                      }
        text_preprocessor.remove_by_regex(regex_dict)
        # text_preprocessor.expand_contraction()
        text_preprocessor.remove_stop_word()

        regex_dict = {0: [r"'", ""], 1: [r"\s{2,}", " "]}
        text_preprocessor.remove_by_regex(regex_dict)
        text_preprocessor.verify_word_length()

        # run EDA.py before run verify_sequence_length
        text_preprocessor.verify_sequence_length()
        



def main() -> None:
    en_preprocessing()
    return None


if __name__ == '__main__':
    main()