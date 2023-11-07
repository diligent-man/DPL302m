import os

from torch import dropout
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf


# Supplementary functions for encoder/ decoder layers
def positional_encoding(pos, d_model):
    def get_angles(pos, i, d_model):
        angle_dropouts = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_dropouts
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :], d_model) # shape (position, d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32) # shape: (position, d_model)



def scaled_dot_product_attention(q, k, v, mask=None):
      """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights
      """
      matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

      # scale matmul_qk
      dk = tf.cast(tf.shape(k)[-1], tf.float32) # dk is same as d_model
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

      # add the mask to the scaled tensor.
      if mask is not None:
        scaled_attention_logits += (mask * -1e9)

      # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
      # indices having mask will be 0 when passed through softmax cuz e^-inf ~ 0
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

      output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
      return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(units=dff, activation='relu'),  # (batch_size, seq_len, dff), dff stands for ????
      tf.keras.layers.Dense(units=d_model)  # (batch_size, seq_len, d_model)
  ])


###############################################################################################################################
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
           Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights




class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)


    def __call__(self, x, training: bool, mask):
        attn_output, _ = self.multi_head_attention(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.feed_forward(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)


    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2



class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_layers, dff,
                 input_vocab_size, maximum_position_encoding, dropout=0.1):
        """
        d_model: number of expected features/ length in the encoder/decoder inputs
        num_heads: # of multi-headed attentions in each block
        num_layers: # of encoder & decoder blocks -> can sepadropoutly split for encoder & decoder (modify later)
        dff: dimension of the feedforward network model
        dropout: dropout

        input_vocab_size: # of output words
        target_vocab_size: # of input words
        pe_input: maximum positional encoding  of encoder block -> how many p we would have
        pe_target: maximum positional encoding of decoder block -> how many p we would have
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model) # ???
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]


    def __call__(self, x, training, mask):
        """
        x:
        training:
        mask:
        """
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
          x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)



class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_layers, dff,
                 target_vocab_size, maximum_positional_encoding, dropout=0.1):
        """
        d_model: number of expected features/ length in the encoder/decoder inputs
        num_heads: # of multi-headed attentions in each block
        num_layers: # of encoder & decoder blocks -> can sepadropoutly split for encoder & decoder (modify later)
        dff: dimension of the feedforward network model
        dropout: dropout

        input_vocab_size: # of output words
        target_vocab_size: # of input words
        pe_input: maximum positional encoding  of encoder block -> how many p we would have
        pe_target: maximum positional encoding of decoder block -> how many p we would have
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model) # ???

        self.pos_encoding = positional_encoding(maximum_positional_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]


    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        x:
        enc_output:
        training:
        look_ahead_mask:
        padding_mask:
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, d_model, num_heads, num_layers, dff,
                 input_vocab_size, target_vocab_size,
                 pe_input, pe_target, dropout=0.1):
        """
        d_model: number of expected features/ length in the encoder/decoder inputs
        num_heads: # of multi-headed attentions in each block
        num_layers: # of encoder & decoder blocks -> can sepadropoutly split for encoder & decoder (modify later)
        dff: dimension of the feedforward network model
        dropout: dropout

        input_vocab_size: # of output words
        target_vocab_size: # of input words
        pe_input: maximum positional encoding  of encoder block -> how many p we would have
        pe_target: maximum positional encoding of decoder block -> how many p we would have
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, num_layers, dff, input_vocab_size, pe_input, dropout)
        self.decoder = Decoder(d_model, num_heads, num_layers, dff, target_vocab_size, pe_target, dropout)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)


    def __call__(self, inp, out, training, enc_padding_mask,
                 look_ahead_mask, dec_padding_mask):
        # print('enc_padding_mask: ', enc_padding_mask)
        enc_output = self.encoder(inp, training, enc_padding_mask)  # inp shape: # (batch_size, inp_seq_len, d_model)
        print("Enc_output", enc_output.shape)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(out, enc_output, training, look_ahead_mask, dec_padding_mask)
        print("Dec_out", dec_output.shape)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


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


def create_look_ahead_mask(batch_size, seq_len):
    # A look-ahead mask is required to prevent the decoder from attending to succeeding words,
    # such that the prediction for a particular word can only depend on known outputs
    # for the words that come before it.

    # Band matrix: https://en.wikipedia.org/wiki/Band_matrix#:~:text=In%20mathematics%2C%20particularly%20matrix%20theory,more%20diagonals%20on%20either%20side.
    # k1: lower bandwidth; k2: upper bandwidth
    # lower bandwith: the distances b/w lower subdiagonal line and main diagonal
    # upper bandwidth: the distances b/w upper subdiagonal line and main diagonal
    # All entris stay outside k1 and k2 will be marked as 0
    
    """
           My    name      is     Trong
    My    index    0        0      0
    name  index   index     0      0
    is    index   index   index    0
    Trong index   index   index   index

    Interpretation - consider row by row
    My: solely attend to itself
    name: attend to itself & My as well
    is: attend to itself also My and name
    name: same deduction

    0 will marked as 1, which will be 0 after put through softmax
    """
    mask = 1 - tf.linalg.band_part(input=tf.ones((batch_size, seq_len, seq_len)), num_lower=-1, num_upper=0) # lower triangular matrix <==> currents word only attend to the previous one
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    # {PAD] = 0 = TRUE
    mask = tf.math.equal(seq, 0)
    mask = tf.cast(mask, tf.float32)
    # add extra dimensions to add the paddingto the attention logits.
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inp, out):
    # inp: (batch_size, seq_len, 50)
    #  out: (batch_size, seq_len, 50)
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)  # (batch_size, 1, 1, seq_len, 50)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)  # (batch_size, 1, 1, seq_len, 50)

    # Used in the 1st attention block in the decoder.
    dec_target_padding_mask = create_padding_mask(out)  # (batch_size, 1, 1, seq_len, 50)
    look_ahead_mask = create_look_ahead_mask(out.shape[1])
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    print(enc_padding_mask.shape, dec_padding_mask.shape, combined_mask.shape)
    return enc_padding_mask, combined_mask, dec_padding_mask


def loss_function(y, y_hat):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss(y, y_hat)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



# def main() -> None:
#     model = Transformer(d_model=512, num_heads=8, num_layers=4, dff=1024,
#                  input_vocab_size=1000, target_vocab_size=1000,
#                  pe_input=1000, pe_target=1000)

#     temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
#     temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
#     print(temp_input)


#     fn_out, _ = model(temp_input, temp_target, training=False, 
#                                enc_padding_mask=None, 
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)

#     print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
#     return None


# if __name__ == '__main__':
#     main()