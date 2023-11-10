import os
import re
import pandas as pd
import spacy
from torchtext.legacy import data
from generate_input import *
from iterator import *
import pickle


class CustomTokenizer(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenize(self, sentence):
        sentence = re.sub(r"[^0-9a-zA-Z' -]+","", sentence)
        sentence = sentence.lower()
        # return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
        return [self.nlp.tokenizer(char).text for char in sentence]


def preprocess(sentence):
    sentence = re.sub(r"[^0-9a-zA-Z' -]+","", sentence)
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
            SOURCE = pickle.load(open(file='weights/SOURCE.pkl', mode='rb'))
            TARGET = pickle.load(open(file='weights/TARGET.pkl', mode='rb'))
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

    raw_data = {'src' : [line for line in option.source_data], 'trg': [line for line in option.target_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < option.max_strlen) & (df['trg'].str.count(' ') < option.max_strlen)
    df = df.loc[mask]

    if os.path.exists('temp.csv'):
        os.remove('temp.csv')

    df.to_csv("temp.csv", index=False)

    data_fields = [('src', SOURCE), ('trg', TARGET)]
    train = data.TabularDataset('./temp.csv', format='csv', fields=data_fields)

    train_iter = CustomIterator(
        train,
        batch_size=option.batch_size,
        device=option.cuda_device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=True,
        shuffle=True
        )

    SOURCE.build_vocab(train)
    TARGET.build_vocab(train)

    if option.checkpoint > 0:
        if os.path.exists('weights/SOURCE.pkl'):
            os.remove('weights/SOURCE.pkl')

        if os.path.exists('weights/TARGET.pkl'):
            os.remove('weights/TARGET.pkl')

        try:
            print("\tSaving SOURCE and TARGET files...")
            pickle.dump(SOURCE, open('weights/SOURCE.pkl', 'wb'))
            pickle.dump(TARGET, open('weights/TARGET.pkl', 'wb'))
            print("\tFinished saving SOURCE and TARGET files.")
        except:
            print("Error: Saving data to weights/<filename>.pkl is not successful.")
            quit()
    
    option.src_pad = SOURCE.vocab.stoi['<pad>']
    option.trg_pad = TARGET.vocab.stoi['<pad>']

    option.train_len = get_length(train_iter)
    print("Finished.")
    return train_iter

def get_length(train):
    for i, b in enumerate(train):
        pass
    return i
