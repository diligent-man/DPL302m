import sys
import os
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from transformers import AutoTokenizer
from wrong_word_generator import add_noise
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config

from Pretrained_models.utils.character_cnn import CharacterIndexer
# from Pretrained_models.character_bert import CharacterBertModel

from transformer import Transformer, CustomSchedule, loss_function

np_config.enable_numpy_behavior()
output_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# BertTokenizer.from_pretrained('../Pretrained_models/bert-base-uncased/')
# character_embedding_model = CharacterBertModel.from_pretrained('../Pretrained_models/general_character_bert/')


#################################################
def randomize_mini_batches(dataset: list, batch_size: int, seed=12345):
    dataset = np.array(dataset)
    num_of_examples = len(dataset)
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
def padding(batch):
    # Add [CLS], [PAD] and [SEP]
    batch = [[*sequence] + (MAX_LENGTH - len(sequence)) * ["[PAD]"] for sequence in batch]
    return batch


def truncating(batch):
    # Tokenize
    batch = [seq.split(' ') for seq in batch]    
    # truncating
    batch = [seq[:MAX_LENGTH] for seq in batch]
    # rejoining
    batch = [(" ".join(seq)).strip() for seq in batch]
    return batch


def train_step(inp: list):
    # Padding
    inp = truncating(inp)
    out = [add_noise(seq, language="en") for seq in inp]

    # create target_inp, target_out
    target_inp = ["[CLS] " + seq for seq in out]
    target_out = [seq + " [SEP]" for seq in out]
    del out

    # print(np.array(target_inp).shape, np.array(target_out).shape)
    # print(target_inp, target_out)
    # print(np.array(target_inp).shape, np.array(target_out).shape)

    # # indexing out
    # indexed_target_inp = [output_tokenizer(seq)["input_ids"] for seq in target_inp]
    # print(np.array(indexed_target_inp))



    # shift out

    # inp will be embedded at encoder
    # out = [output_tokenizer(sentence)["input_ids"] for sentence in out]
    # print(out)
    # target_inp = index_char(batch=out, shift=True)
    # target_out = index_char(batch=out, shift=False)
    # del out

    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, target_inp)
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


model = Transformer(d_model=768, num_heads=8, num_layers=4, dff=1024,
                    input_vocab_size=1000, target_vocab_size=1000,
                    pe_input=1000, pe_target=1000,
                    dropout=0.1)

BATCH_SIZE = 2
MAX_LENGTH = 40
learning_rate = CustomSchedule(768)
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
def main() -> None:
    # read dataset
    dataset = []
    with open("train.txt", 'r') as f:
        for line in f:
            dataset.append(line[:-1])

    # mini_batch splitting
    mini_batches = randomize_mini_batches(dataset=dataset, batch_size=BATCH_SIZE)
    num_mini_batches = len(mini_batches)
    print("Num of mini_batches:", num_mini_batches)
    
    # Training
    for epoch in range(1):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for index, mini_batch in tqdm(enumerate(mini_batches), total=num_mini_batches, dynamic_ncols=True):
            train_step(mini_batch)
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