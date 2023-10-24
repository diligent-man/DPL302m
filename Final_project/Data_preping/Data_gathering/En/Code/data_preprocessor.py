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
                        if len(line.split()) < 7:
                            continue
                        else:
                            data.append(line)
                    # save data
                    data = '\n'.join(data)
                    self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)
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
        text_preprocessor.set_data_path(text_preprocessor.get_preprocessed_data_path())

        text_preprocessor.eliminate_contraction()
        text_preprocessor.set_data_path(text_preprocessor.get_preprocessed_data_path())

        regex_dict = {0: [r"[~!@#$%\^&\*()\_,，./<>\?;:：\"\[\]\{\}\\|“”\u2122\u00A90-9\u300A\u300B]*", ""],  # punctuation_marks_and_numeral
                      1: [r"[–]", ""], 
                      2: [r"[\u4E00-\u9FFF]", ""], # Chinese hieroglyphs
                      3: [r"[\u00E0\u00E1\u1EA3\u00E3\u1EA1\u00E2\u1EA7\u1EA5\u1EAD\u1EAB\u1EAF\u0103\u1EB1\u1EAF\u1EB5\u1EB3\u1EB7\u00E8\u00E9\u1EBB\u1EBD\u1EB9\u00EA\u1EC1\u1EBF\u1EC3\u1EC5\u1EC7\u0111\u00EC\u00ED\u1EC9\u0129\u1ECB\u00F2\u00F3\u1ECF\u00F5\u1ECD\u00F4\u1ED3\u1ED1\u1ED5\u1ED7\u1ED9\u01A1\u1EDD\u1EDB\u1EDF\u1EE1\u1EE3\u00F9\u00FA\u1EE7\u0169\u1EE5\u1EEB\u1EE9\u1EED\u1EEF\u1EF1\u1EF3\u00FD\u1EF7\u1EF9\u1EF5\u00C0\u00C1\u00C2\u00C3\u00C4\u00C7\u00C8\u00C9\u00CA\u00CB\u00CE\u00CF\u00D4\u0152\u00D9\u00DA\u00DB\u00DC\u0178\u00E0\u00E1\u00E2\u00E3\u00E4\u00E7\u00E8\u00E9\u00EA\u00EB\u00EE\u00EF\u00F4\u0153\u00F9\u00FA\u00FB\u00FC\u00FD\u00FF]", ""], # Vietnamese chars
                      4: [r"[\u0255\u01D4\u01CE]", ""] # IPA
                     }
        text_preprocessor.remove_by_regex(regex_dict)
        


def main() -> None:
    en_preprocessing()
    return None


if __name__ == '__main__':
    main()
