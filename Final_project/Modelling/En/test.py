import tensorflow as tf


class Name_generator:
    def __init__(self,char_to_ind_map, ind_to_char_map, vocab_size, max_seq_len):
        self.charId = char_to_ind_map 
        self.idChar = ind_to_char_map
        self.max_name_len = max_seq_len
        self.vocab_size = vocab_size

    def generate_name(self, model, start_chars):
        chars = list(start_chars)
        c = 0
        i = 0
        while c != '#':
            ids = [self.charId[char] for char in chars]
            ids_padded = tf.keras.utils.pad_sequences([ids], value = 0.0, padding = 'post', maxlen = self.max_name_len)
            probs = model(ids_padded, training = False)[0][i].numpy()
            probs = probs/sum(probs)
            d = np.random.choice(len(vocab), p = probs)
            if d != 0:
                c = self.idChar[d]
                chars.append(c)
                i = i + 1 
        print(''.join(chars).replace('#',''))

        
    def __call__(self, model, epoch):
        print('\n Names generated after epoch {}:\n'.format(epoch + 1))
        for _ in range(5):
            index = np.random.randint(1,self.vocab_size-1)
            self.generate_name(model,self.idChar[index])




# Preparing data, "name" is a list of nicknames to train the model on 
names = ['Nguyen Duc Trong', "Nguyen Van A", "Nguyen Van B"]
vocab = ['[pad]'] + sorted(set(''.join(names)))
charId = dict([el, i] for i, el in enumerate(vocab))
idChar = dict([i, el] for i, el in enumerate(vocab))

print(vocab)
print(charId)
print(idChar)



# def char_to_id(name):
#     name = [*name]
#     name = [charId[item] for item in name]
#     return name


# vectorized_names = list(map(char_to_id, names))
# max_name_len = max(list(map(lambda x: len(x), vectorized_names)))
# vectorized_names_padded = tf.keras.utils.pad_sequences(vectorized_names, padding = 'post', maxlen = max_name_len)
# # caching the vectorized input sequences in order to build the model
# x = tf.convert_to_tensor(vectorized_names_padded[:,:-1])
# # tf.data.Dataset instance for training
# batch_size = 64
# dataset = tf.data.Dataset.from_tensor_slices(vectorized_names_padded).shuffle(buffer_size = 1000).batch(batch_size)
# # turning elements of dataset into pairs of input and output sequences
# def prepare_input_output(sequence):
#     x = sequence[:, :-1]
#     y = sequence[:, 1:]
#     return x, y
# dataset = dataset.map(prepare_input_output)
# # creating an instance of Transformer model with hyperparameters
# TF = Transformer(len(vocab),max_name_len-1,8,256,256,256,512,0.2,2)
# optimizer = tf.keras.optimizers.Adam()
# TF(x[0:1,:]) # building the model
# TF.summary()
