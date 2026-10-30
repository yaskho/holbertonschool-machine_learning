#!/usr/bin/env python3
"""
Module containing Transformer model and all its building blocks.
"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculates positional encoding for a sequence.

    Args:
        max_seq_len: maximum sequence length
        dm: model depth/dimensionality

    Returns:
        numpy array of shape (1, max_seq_len, dm)
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention layer.
    """

    def __init__(self, dm, h):
        """
        Class constructor.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.q_dense = tf.keras.layers.Dense(dm)
        self.k_dense = tf.keras.layers.Dense(dm)
        self.v_dense = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Call method for multi-head attention.
        """
        batch_size = tf.shape(Q)[0]

        q = self.q_dense(Q)
        k = self.k_dense(K)
        v = self.v_dense(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)
        return output, attention_weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block layer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=None, mask=None):
        """
        Call method for EncoderBlock.
        """
        attn_output, _ = self.mha(x, x, x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """
    Decoder block layer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training=None, look_ahead_mask=None,
             padding_mask=None):
        """
        Call method for DecoderBlock.
        """
        attn1, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1, enc_output, enc_output, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3


class Encoder(tf.keras.layers.Layer):
    """
    Encoder module for Transformer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab_size, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, dm)
        self.pos_encoding = positional_encoding(max_seq_len, dm)
        self.enc_layers = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=None, mask=None):
        """
        Call method for Encoder.
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder module for Transformer.
    """

    def __init__(self, N, dm, h, hidden, target_vocab_size, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, dm)
        self.pos_encoding = positional_encoding(max_seq_len, dm)
        self.dec_layers = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training=None, look_ahead_mask=None,
             padding_mask=None):
        """
        Call method for Decoder.
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.dec_layers[i](
                x, enc_output, training=training,
                look_ahead_mask=look_ahead_mask, padding_mask=padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """
    Transformer network architecture.
    """

    def __init__(self, N, dm, h, hidden, input_vocab_size, target_vocab_size,
                 max_len, drop_rate=0.1):
        """
        Class constructor.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab_size, max_len, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab_size, max_len, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, target, training=None, encoder_mask=None,
             look_ahead_mask=None, decoder_mask=None):
        """
        Call method for Transformer.
        """
        enc_output = self.encoder(
            inputs, training=training, mask=encoder_mask
        )
        dec_output = self.decoder(
            target, enc_output, training=training,
            look_ahead_mask=look_ahead_mask, padding_mask=decoder_mask
        )
        final_output = self.linear(dec_output)

        return final_output
