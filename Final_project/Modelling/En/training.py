import sys
import os
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf



from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.model_selection import train_test_split
from Pretrained_models.character_bert import CharacterBertModel
from Pretrained_models.character_cnn import CharacterIndexer
from transformer import Transformer, CustomSchedule, create_masks, loss_function

np_config.enable_numpy_behavior()


#################################################
class DataGovernor():
    def __init__(self):
        self.__path = ""
        self.__X = None


    def set_path(self, path):
        self.__path = path


    def get_path(self):
        return self.__path


    def set_X(self, X):
        self.__X = X


    def get_X(self):
        return self.__X


    def load_dataset(self) -> tuple:
        # Only load index into mem
        num_of_wrong_sentence = 10  # ref wrong_word_generator.py
        with open(self.__path, 'r') as f:
            num_of_line_in_corpus = len(f.readlines())
        # indexes = [*range(num_of_line_in_corpus * num_of_wrong_sentence)]  # num_of_lines
        indexes = [*range(100)] # num_of_lines
        train_indexes, test_indexes = train_test_split(indexes, test_size=0.2, random_state=12345, shuffle=True) # list of indexes
        return np.array(train_indexes), np.array(test_indexes)


    def randomize_mini_batches(self, batch_size: int, seed=12345):
        dataset = self.__X
        num_of_examples = dataset.shape[0]
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(num_of_examples))
        shuffled_dataset = dataset[permutation]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(num_of_examples / batch_size)  # number of mini batches of size batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batches.append(shuffled_dataset[k * batch_size: k * batch_size + batch_size])

        # Handling the end case (last mini-batch < batch_size)
        if num_of_examples % batch_size != 0:
            mini_batches.append(shuffled_dataset[num_complete_minibatches * batch_size : num_of_examples])
        return mini_batches


##############################################################################################
def index_char(batch, shift):
    indexer = CharacterIndexer()    
    indexed_batch = []
    # Tokenize
    batch = [sequence[:-1].split(' ') for sequence in batch]
    # Add [CLS], [PAD] and [SEP]
    batch = [["[CLS]", *sequence] + (MAX_LENGTH - len(sequence)) * ["[PAD]"] + ["[SEP]"] for sequence in batch]

    # case 1: shift is None -> inp
    if shift is None:
        pass
    # case 2: shift is True -> target_input
    elif shift == True:
        # needn't add PAD for removed entry ????
        batch = [sequence[:-1] for sequence in batch]
    # case 3: shift is True -> target_ouput
    elif shift == False:
        # needn't add PAD for removed entry ????
        batch = [sequence[1:] for sequence in batch]

    # Convert token sequence into character indices
    batch = indexer.as_padded_tensor(batch, as_tensor="tf")
    return np.array(batch)


def split_train_test(indexes: list):
    training_files = sorted(os.listdir(training_file_path))
    inp = []
    out = []
    for index in indexes:
        for i in range(len(training_files)):
            starter = int(training_files[i][:-4].split('_')[0])
            ender = int(training_files[i][:-4].split('_')[1])

            if index > starter and index < ender:
                with open(training_file_path + training_files[i], 'r') as f:
                    line = f.readlines()[ender-index].split('|')
                    inp.append(line[1])  # incorrect sentence
                    out.append(line[0])  # correct sentence
    return np.array(inp), np.array(out)


def train_step(indexes: list):
    inp, out = split_train_test(indexes)
    inp = index_char(batch=inp, shift=None)
    target_input = index_char(batch=out, shift=True)
    target_output = index_char(batch=out, shift=False)
    del out
    print(inp.shape, target_input.shape, target_output.shape)

    # shifted_right_out = np.array([" ".join(sequence.split(' ')[1:]) for sequence in out]) # uses for prediction

    # inp = tf.map_fn(fn=index_char(shift=None), elems=inp, fn_output_signature=tf.int32)  # (batch_szie, max_length, 50)
    # target_input = tf.map_fn(fn=index_char(shift=True), elems=out, fn_output_signature=tf.int32)  # (batch_szie, max_length, 50)
    # target_output = tf.map_fn(fn=index_char(shift=False), elems=out, fn_output_signature=tf.int32)  # (batch_szie, max_length, 50)s
    # pritn(inp.shape, target_input.shape, target_output.shape)
    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp.reshape(BATCH_SIZE * (MAX_LENGTH + 2), 50), out.reshape(BATCH_SIZE * (MAX_LENGTH + 2), 50))
    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, out)

    # print(enc_padding_mask.shape, combined_mask.shape, dec_padding_mask.shape)

    # with tf.GradientTape() as tape:
    #     predictions, _ = model(inp=inp, out=out, training=True,
    #                            enc_padding_mask=enc_padding_mask,
    #                            look_ahead_mask=combined_mask,
    #                            dec_padding_mask=dec_padding_mask)
    #     loss = loss_function(shifted_right_out, predictions)

    # gradients = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # train_loss(loss)
    # train_accuracy(shifted_right_out, predictions)


embedding_model = CharacterBertModel.from_pretrained('../Pretrained_models/general_character_bert/')
model = Transformer(d_model=768, num_heads=8, num_layers=4, dff=1024,
                    input_vocab_size=1000, target_vocab_size=1000,
                    pe_input=1000, pe_target=1000,
                    dropout=0.1)

corpus_path = '../../Wrong_word_generator/en_corpus.txt'
training_file_path = '../../Wrong_word_generator/noised_en/'
BATCH_SIZE = 2
MAX_LENGTH = 40
learning_rate = CustomSchedule(768)
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


def main() -> None:
    data_governor = DataGovernor()
    data_governor.set_path(corpus_path)
    train_indexes, test_indexes = data_governor.load_dataset()
    print(f"""Train indexes shape: {train_indexes.shape[0]}
Test indexes shape: {test_indexes.shape[0]}""")
    
    # Each mini-batch contains 8 indexes for each training iteration
    data_governor.set_X(train_indexes)
    mini_batches = data_governor.randomize_mini_batches(batch_size=BATCH_SIZE)  # (num_of_batches, indexes_in_each_batch)
    num_of_mini_batches = len(mini_batches)
    print(f"Number of batches: {num_of_mini_batches}")

    # Training
    for epoch in range(1):
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> non_diacritic, tar -> diacritic
        for batch, indexes in tqdm(enumerate(mini_batches), total=num_of_mini_batches, dynamic_ncols=True):
            train_step(indexes)
            print(f"""Epoch: {epoch}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}\n""")

        # if (epoch + 1) % 5 == 0:
        # ckpt_save_path = ckpt_manager.save()
        # print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
        #                                                      ckpt_save_path))

        # print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
        #                                             train_loss.result(), 
        #                                             train_accuracy.result()))

    #     print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    return None


if __name__ == '__main__':
    main()