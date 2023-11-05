import sys
import os
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf


from tqdm import tqdm
from matplotlib import pyplot as plt
from transformer import Transformer
from transformers import BertTokenizer
from tensorflow.python.ops.numpy_ops import np_config

from sklearn.model_selection import train_test_split
from Pretrained_models.character_bert import CharacterBertModel
from Pretrained_models.character_cnn import CharacterIndexer

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
        # indexes = [*range(len(os.listdir(self.__path)))] # num_of_lines
        indexes = [*range(10000)] # num_of_lines
        train, test = train_test_split(indexes, test_size=0.2, random_state=12345, shuffle=True) # list of indexes
        return np.array(train), np.array(test)
    

    def randomize_mini_batches(self, batch_size: int, seed=12345):
        dataset = self.__X
        num_of_examples = dataset.shape[0]
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(num_of_examples))
        shuffled_dataset = dataset[permutation]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(num_of_examples / batch_size) # number of mini batches of size batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batches.append(shuffled_dataset[k * batch_size : k * batch_size + batch_size])
        
        # Handling the end case (last mini-batch < batch_size)
        if num_of_examples % batch_size != 0:
            mini_batches.append(shuffled_dataset[num_complete_minibatches * batch_size : num_of_examples])
        return mini_batches


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


##############################################################################################
def loss_function(y, y_hat):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss(y, y_hat)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inp, out):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(out)[1])
    dec_target_padding_mask = create_padding_mask(out)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


def index_char(sequence, MAX_LENGTH=50):
    indexer = CharacterIndexer()
    sequence = sequence.numpy().decode('utf-8')

    # Tokenize
    sequence = tokenizer.basic_tokenizer.tokenize(sequence)

    # Add [CLS], [PAD] and [SEP]
    sequence = ["[CLS]", *sequence] + (MAX_LENGTH - len(sequence) - 2) * ["[PAD]"] + ["[SEP]"]
    
    # Convert token sequence into character indices
    sequence = indexer.as_padded_tensor([sequence])
    sequence = np.squeeze(sequence.detach().numpy(), axis=0)
    # sequence = sequence.reshape(MAX_LENGTH * sequence.shape[1]) # (MAX_LENGTH * index_dim)
    return sequence


def split_inp_out(dataset):
    inp = []  # contains incorrect sentences
    out = []  # contains correct sentences
    for line in dataset:
        inp.append(line.decode('utf-8').split('|')[1])
        out.append(line.decode('utf-8').split('|')[0])
    return np.array(inp), np.array(out)


def train_step(indexes):
    filenames = [directory + '/' + str(index) + ".txt" for index in indexes]
    dataset = list(tf.data.TextLineDataset(filenames=filenames).as_numpy_iterator())
    inp, out = split_inp_out(dataset)
    shifted_right_out = np.array([" ".join(sequence.split(' ')[1:]) for sequence in out]) # uses for prediction

    inp = tf.map_fn(fn=index_char, elems=inp, dtype=tf.int32)
    out = tf.map_fn(fn=index_char, elems=out, dtype=tf.int32)
    shifted_right_out = tf.map_fn(fn=index_char, elems=shifted_right_out, dtype=tf.int32)

    inp = inp.reshape(-1, 50)  # (mini_batch_size * MAX_LENGTH, 50)
    out = out.reshape(-1, 50)  # (mini_batch_size * MAX_LENGTH, 50)
    shifted_right_out = shifted_right_out.reshape(-1, 50)  # (mini_batch_size * MAX_LENGTH, 50)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, out)

    with tf.GradientTape() as tape:
        predictions, _ = model(inp=inp, out=out, training=True,
                               enc_padding_mask=enc_padding_mask,
                               look_ahead_mask=combined_mask,
                               dec_padding_mask=dec_padding_mask)
        loss = loss_function(shifted_right_out, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(shifted_right_out, predictions)



tokenizer = BertTokenizer.from_pretrained('../Pretrained_models/bert-base-uncased/')
embedding_model = CharacterBertModel.from_pretrained('../Pretrained_models/general_character_bert/')
model = Transformer(d_model=768, num_heads=8, num_layers=4, dff=1024,
                    input_vocab_size=1000, target_vocab_size=1000,
                    pe_input=1000, pe_target=1000,
                    dropout=0.1)

directory = '../../Wrong_word_generator/noised_en'
learning_rate = CustomSchedule(768)
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
def main() -> None:
    data_governor = DataGovernor()
    data_governor.set_path(directory)
    train_indexes, test_indexdes = data_governor.load_dataset()

    # Each mini-batch contains 8 indexes for each training iteration
    data_governor.set_X(train_indexes)
    mini_batches = data_governor.randomize_mini_batches(batch_size=1)  # (num_of_batches, indexes_in_each_batch)
    num_of_mini_batches = len(mini_batches)
    print(f"Number of batches: {num_of_mini_batches}")
    
    # Training
    for epoch in range(3):
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> non_diacritic, tar -> diacritic
        for (batch, indexes) in tqdm(enumerate(mini_batches), desc="Processing", total=num_of_mini_batches, colour='CYAN', dynamic_ncols=True, ncols=2):
            train_step(indexes)
            print()
            print(f'''Epoch: {epoch}, Batch: {batch}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}\n''')
          
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


