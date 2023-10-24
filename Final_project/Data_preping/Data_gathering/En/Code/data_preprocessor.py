import os
import re
import string
import shutil
import contractions
import textsearch


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


    def eliminate_contraction(self) -> None:
        contractions_dict = {"how'll":"how will",
                             "mayn't":"may not",
                             "might've":"might have",
                             "mightn't":"might not",
                             "mustn't":"must not",
                             "needn't":"need not",
                             "shan't":"shall not",
                             "so've":"so have",
                             "that'd":"that would",
                             "there'd":"there would",
                             "to've":"to have",
                             "what'll":"what will",
                             "when've":"when have",
                             "where've":"where have",
                             "why've":"why have",
                             "will've":"will have"}
        
        for filename in sorted(os.listdir(self.__data_path)):
            with open(self.__data_path + '/' + filename, 'r', encoding='utf-8') as f:
                expanded_lines = []
                for line in f:
                    expanded_words = [self.__expand_contractions(word, contractions_dict) for word in line.split()]
                    expanded_line = ' '.join(expanded_words)
                    expanded_lines.append(expanded_line)
                # save data
                expanded_lines = '\n'.join(expanded_lines)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, expanded_lines)
            return None
                                    
    
    def remove_emoji(self) -> None:
        emoji_pattern = re.compile("["
                                   u"U0001F600-U0001F64F"  # emoticons
                                   u"U0001F300-U0001F5FF"  # symbols & pictographs
                                   u"U0001F680-U0001F6FF"  # transport & map symbols
                                   u"U00002702-U000027B0"
                                   u"U000024C2-U0001F251"
                                   "]+", flags=re.UNICODE)
            
        for filename in sorted(os.listdir(self.__data_path)):
            with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    data.append(emoji_pattern.sub(r'', line))
                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
        return None


    def remove_by_regex(self, regex_dict):
        for i in regex_dict:
            for filename in sorted(os.listdir(self.__data_path)):
                with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:            
                    data = []
                    for line in f:
                        line = re.sub(regex_dict[i][0], regex_dict[i][1], line)

                        # remove ws at the beginning and the end of sentence
                        line = ' '.join([word.strip() for word in line.split()])
                        # if len line < 7, ignore it
                        if len(line.split()) < 7:
                            continue
                        else:
                            data.append(line)
                    # save data
                    data = '\n'.join(data)
                    self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
            self.set_data_path(self.get_preprocessed_data_path())
        return None


    


def en_preprocessing() -> None:
    metadata = {"category_ls_1": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Tech', 'Travel', 'Weather','World'],
                "category_ls_2": ['Ted_talk', 'Wiki'],
                "data_dir_1": ['../Data/BBC'],
                "data_dir_2": ['../Data'],
                "preprocessed_data_dir_1":["../Preprocessed_data/BBC"],
                "preprocessed_data_dir_2":["../Preprocessed_data"]
                     }

    # split into singular sentence
    index_1 = index_2 = 0
    for i in range(len(metadata["category_ls_1"]) + len(metadata["category_ls_2"])):
        text_preprocessor = TextPreprocessor()

        # choosing what category will be processed
        if i < len(metadata["category_ls_1"]):
            flag = "1"
            category = metadata["category_ls_" + flag][index_1]
            index_1 += 1
        else:
            flag = "2"
            category = metadata["category_ls_" + flag][index_2]
            index_2 += 1
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
        text_preprocessor.eliminate_contraction()
        text_preprocessor.remove_emoji()

        regex_dict = {0: [r"[~!@#$%\^&\*()\_,./<>\?;:\"\[\]\{\}\\\|“”\u2122\u00A90-9]*", ""],  # punctuation_marks_and_numeral
                     1: [r"[-–]", " "], # hyphen & dash
                     2: [r"[\u4E00-\u9FFF]", " "], # Chinese hieroglyphs
                     3: [r"[\u1D00-\u1D7F\u1D80-\u1DBF\u2070-\u209F\u0300-\u036F\u0255\u01D4]", ""]
                     }   
        text_preprocessor.remove_by_regex(regex_dict)
        


def main() -> None:
    en_preprocessing()
    return None


if __name__ == '__main__':
    main()
