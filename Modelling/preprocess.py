import os
import re
import sys
import spacy
import pickle
import pandas as pd

import pathlib # used when running in window
from torchtext.legacy import data
from generate_input import generate_input
from iterator import batch_size_fn, CustomIterator

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


# check env
colab = False
if colab:
    # gg colab env
    SOURCE_path = "/content/drive/MyDrive/Modelling/weights/SOURCE.pkl"
    TARGET_path = "/content/drive/MyDrive/Modelling/weights/TARGET.pkl"
    temp_path = "/content/drive/MyDrive/Modelling/temp.csv"
else:
    # local env
    SOURCE_path = "weights/SOURCE.pkl"
    TARGET_path = "weights/TARGET.pkl"
    temp_path = "temp.csv"


class CustomTokenizer(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenize(self, sentence):
        sentence = re.sub(r"[^0-9a-zA-Z' -]+", "", sentence)
        sentence = sentence.lower()
        # return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
        return [self.nlp.tokenizer(char).text for char in sentence]


def preprocess(sentence):
    sentence = re.sub(r"[^0-9a-zA-Z' -]+", "", sentence)
    sentence = sentence.lower()
    return sentence


def create_files(option):
    # tokenize and create target & source
    source = CustomTokenizer(option.source_lang)
    target = CustomTokenizer(option.target_lang)

    TARGET = data.Field(lower=True, tokenize=target.tokenize, init_token='<sos>', eos_token='<eos>')
    SOURCE = data.Field(lower=True, tokenize=source.tokenize)

    if option.load_weights is True:
        try:
            print("Loading presaved SOURCE and TARGET files...")
            SOURCE = pickle.load(open(file=SOURCE_path, mode='rb'))
            TARGET = pickle.load(open(file=TARGET_path, mode='rb'))
            print("Finished.")
        except:
            print("Error: Cannot open SOURCE.pkl and TARGET.pkl files.")
            quit()
    return SOURCE, TARGET


def create_data(option, SOURCE, TARGET, repeat=0):
    if repeat == 0:
        print("Generating new dataset...")
    else:
        print("Generating new dataset for the next epoch...")

    option.source_data = generate_input(option)

    if option.data_file is not None and repeat == 0:
        try:
            option.target_data = open(option.data_file).read().strip().split('\n')
        except:
            print("Error: '" + option.data_file + "' file not found.")
            quit()

    tmp_source = CustomTokenizer(option.source_lang)
    tmp_target = CustomTokenizer(option.target_lang)

    raw_data = {'source': [line for line in option.source_data], 'target': [line for line in option.target_data]}
    df = pd.DataFrame(raw_data, columns=["source", "target"])

    mask = (df['source'].str.count(' ') < option.max_strlen) & (df['target'].str.count(' ') < option.max_strlen)
    df = df.loc[mask]

    if os.path.exists(temp_path):
        os.remove(temp_path)

    df.to_csv("temp.csv", index=False)

    data_fields = [('source', SOURCE), ('target', TARGET)]
    train = data.TabularDataset('temp.csv', format='csv', fields=data_fields)

    train_iter = CustomIterator(
        train,
        batch_size=option.batch_size,
        device=option.cuda_device,
        repeat=False,
        sort_key=lambda x: (len(x.source), len(x.target)),
        batch_size_fn=batch_size_fn,
        train=True,
        shuffle=True
    )

    SOURCE.build_vocab(train)
    TARGET.build_vocab(train)

    if option.checkpoint > 0:
        if os.path.exists(SOURCE_path):
            os.remove(SOURCE_path)

        if os.path.exists(TARGET_path):
            os.remove(TARGET_path)

        try:
            print("\tSaving SOURCE and TARGET files...")
            pickle.dump(SOURCE, open(SOURCE_path, 'wb'))
            pickle.dump(TARGET, open(TARGET_path, 'wb'))
            print("\tFinished saving SOURCE and TARGET files.")
        except:
            print("Error: Saving data to weights/<filename>.pkl is not successful.")
            quit()

    option.source_pad = SOURCE.vocab.stoi['<pad>']
    option.target_pad = TARGET.vocab.stoi['<pad>']

    option.train_len = get_length(train_iter)
    print("Finished.")
    return train_iter


def get_length(train):
    for i, b in enumerate(train):
        pass
    return i