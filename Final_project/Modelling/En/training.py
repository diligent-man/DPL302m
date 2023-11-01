import sys
import os
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


from transformer import Transformer
from transformers import BertTokenizer
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split
from Pretrained_models.character_bert import CharacterBertModel
from Pretrained_models.character_cnn import CharacterIndexer


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
            filename = self.__path + file
            with open(filename, mode='r', encoding='utf-8') as f:
                print(filename)
                for line in f:
                    x_train.append(line.split('|')[0].strip())
                    y_train.append(line.split('|')[1].strip())

        x_test = []
        y_test = []
        for file in test_file:
            filename = self.__path + file
            with open(filename, mode='r', encoding='utf-8') as f:
                print(filename)
                for line in f:
                    x_test.append(line.split('|')[0].strip())
                    y_test.append(line.split('|')[1].strip())
        return x_train, y_train, x_test, y_test




def tokenize_sequence(sequence, model, tokenizer):    
    tokenized_sequence = tokenizer.basic_tokenizer.tokenize(sequence)
    tokenized_sequence = ['[CLS]', *tokenized_sequence, '[SEP]']
    
    indexer = CharacterIndexer()
    padded_tokenized_sequence = indexer.as_padded_tensor([tokenized_sequence])  # This is a batch with a single token sequence x
    

    # Feed batch to CharacterBERT & get the embeddings
    embeddings_for_batch, _ = model(padded_tokenized_sequence)
    embeddings_for_sequence = embeddings_for_batch[0]  
    return embeddings_for_sequence


import pickle
def save_pickle(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def main() -> None:
    path = '../../Wrong_word_generator/En_noised_data/'

    data_loader = DataLoader()
    data_loader.set_path(path)
    x_train, y_train, x_test, y_test = data_loader.load_dataset()
    

    print(len(x_train), len(x_test))
    tokenizer_opt = []
    for sentence in y_train:
        print(sentence)
        tokenizer_opt.append(tfds.features.text.SubwordTextEncoder.build_from_corpus(sentence.numpy(), target_vocab_size=2**13))
    print(tokenizer_opt)

    save_pickle('tokenizer/tokenizer_opt.pkl', tokenizer_opt)


    # model = CharacterBertModel.from_pretrained('../Pretrained_models/general_character_bert/')
    # tokenizer = BertTokenizer.from_pretrained('../Pretrained_models/bert-base-uncased/')   
    
    # for i in range(100):
    #     print(x_test[i])
    #     x_test[i] = tokenize_sequence(x_test[i], model, tokenizer)
    # print(x_test[:5])
    # tokenized_texts = tf.map_fn(, x_test, swap_memory=True, parallel_iterations=50)
    # print(type(tokenized_texts))
    # print(tokenized_texts)


    
    # x_train, y_train, x_test, y_test = tokenize(x_train, y_train, x_test, y_test)
    


    


    
    

    # # Convert token sequence into character indices
    # indexer = CharacterIndexer()
    # batch = []  # This is a batch with a single token sequence x
    # batch_ids = indexer.as_padded_tensor(batch)

    # # Load some pre-trained CharacterBERT
    

    # # Feed batch to CharacterBERT & get the embeddings
    # embeddings_for_batch, _ = model(batch_ids)
    # embeddings_for_x = embeddings_for_batch[0]
    # print('These are the embeddings produces by CharacterBERT (last transformer layer)')
    return None


if __name__ == '__main__':

    main()


