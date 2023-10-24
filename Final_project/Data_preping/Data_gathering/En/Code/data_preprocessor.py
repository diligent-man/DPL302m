import os
import re
import shutil
import contractions


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

            # save data
            data = '\n'.join(sentences)
            self.__save_preprocessed_data(self.__preprocessed_data_path,  filename, data)


    def eliminate_contraction(self):
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
        
        for filename in os.listdir(self.__data_path):
            with open(self.__data_path + '/' + filename, 'r', encoding='utf-8') as f:
                expanded_lines = []
                for line in f:
                    expanded_words = [self.__expand_contractions(word, contractions_dict) for word in line.split()]
                    expanded_line = ' '.join(expanded_words)
                    expanded_lines.append(expanded_line)
            
            # save data
            expanded_lines = '\n'.join(expanded_lines)
            self.__save_preprocessed_data(self.__preprocessed_data_path, filename, expanded_lines)
                                    

    def remove_non_alphabet_chars(self):
        for filename in sorted(os.listdir(self.__data_path)):
            print(filename, self.get_data_path())
            with open(os.path.join(self.__data_path, filename), 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    # remove non-alphabet chars
                    line = re.sub(r"[^a-zA-Z \']", ' ', line)
                    if line.startswith('-'):
                        line = line[1:]

                    # remove ws at the beginning and the end of sentence
                    line = ' '.join(line.strip().split())

                    # if len line < 5, ignore it
                    if len(line.split()) < 5:
                        continue
                    else:
                        data.append(line)
            
                # save data
                data = '\n'.join(data)
                self.__save_preprocessed_data(self.__preprocessed_data_path, filename, data)


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

        # remove preprocessed data dir if it exists
        if category in os.listdir(preprocessed_data_dir):
            shutil.rmtree(path=preprocessed_data_dir + '/' + category)
        os.mkdir(path=preprocessed_data_dir + '/' + category, mode=0o777)

        # perform splitting
        text_preprocessor.split_into_sentence()

        # eliminate contraction
        text_preprocessor.eliminate_contraction()

        # remove punctuation marks
        text_preprocessor.remove_non_alphabet_chars()
    
        


def main() -> None:
    en_preprocessing()
    return None


if __name__ == '__main__':
    main()
