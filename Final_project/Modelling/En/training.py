import sys
import os
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from transformers import AutoTokenizer
from wrong_word_generator import add_noise, make_misspellings
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
from transformer import Transformer, CustomSchedule, loss_function

np_config.enable_numpy_behavior()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
CLS = tokenizer("[CLS]", add_special_tokens=False)["input_ids"]  # [101]
SEP = tokenizer("[SEP]", add_special_tokens=False)["input_ids"]  # [102]

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

    for i in range(len(mini_batches)):
        mini_batches[i] = [str(seq) for seq in mini_batches[i]]
    return mini_batches


##############################################################################################
def truncating(batch):
    # Tokenize
    batch = [seq.split(' ') for seq in batch]
    # truncating
    batch = [seq[:MAX_LENGTH-1] for seq in batch]
    # rejoining
    batch = [(" ".join(seq)).strip() for seq in batch]
    return batch


def train_step(inp: list):
    # Add noise to create output
    out = [make_misspellings(seq) for seq in inp]
    out = truncating(out)

    # Tokenize
    inp = tokenizer(inp, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="np")["input_ids"]  # (batch_size, MAX_LENGTH)
    out = tokenizer(out, padding="max_length", max_length=MAX_LENGTH-1, truncation=True, add_special_tokens=False, return_tensors="np")["input_ids"]  # (batch_size, MAX_LENGTH)

    # Create target_inp, target_out
    target_inp = tf.Variable([tf.concat([CLS, indexed_seq], axis=0) for indexed_seq in out])
    target_out = tf.Variable([tf.concat([indexed_seq, SEP], axis=0) for indexed_seq in out])

    with tf.GradientTape() as tape:
        predictions, _ = model(inp=inp, target_inp=target_inp, training=True)
        # print(predictions)
        loss = loss_function(target_out, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target_out, predictions)


model = Transformer(d_model=768, num_heads=8, num_layers=4, dff=1024,
                    target_vocab_size=30522, pe_input=1000, pe_target=1000, dropout=0.1)

BATCH_SIZE = 2
MAX_LENGTH = 40
learning_rate = CustomSchedule(768)
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

checkpoint_path = "checkpoints/"
ckpt = tf.train.Checkpoint(transformer=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

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
            start = time.time()
            train_step(mini_batch)
            print(f"""Epoch: {epoch}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}\n""")

            if (index + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for iteration {} at {}'.format(index, ckpt_save_path))

            print('Time taken for 1 iteration: {} secs\n'.format(time.time() - start))
    return None


if __name__ == '__main__':
    main()