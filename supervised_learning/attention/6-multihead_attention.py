#!/usr/bin/env python3
"""
Multi-Head Attention Layer using TensorFlow Keras
"""
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class MultiHeadAttention to perform multi-head attention
    """
    def __init__(self, dm, h):
        """
        Class constructor

        Args:
            dm (int): Dimensionality of the model
            h (int): Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)

        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension of x into (h, depth).
        Transpose the result to shape: (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Executes multi-head attention over the inputs Q, K, V

        Args:
            Q: Tensor of shape (batch, seq_len_q, dk)
            K: Tensor of shape (batch, seq_len_v, dk)
            V: Tensor of shape (batch, seq_len_v, dv)
            mask: Always None or a mask tensor

        Returns:
            output: Tensor containing scaled dot-product attention
                    with last dimensions (..., seq_len_q, dm)
            weights: Tensor containing attention weights
                    with last dimensions (..., h, seq_len_q, seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        # Pass Q, K, V through linear projections
        q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        k = self.Wk(K)  # (batch_size, seq_len_v, dm)
        v = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # Split projections into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, h, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, h, seq_len_v, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, h, seq_len_v, depth)

        # Apply scaled dot product attention on split heads
        # output shape: (batch_size, h, seq_len_q, depth)
        # weights shape: (batch_size, h, seq_len_q, seq_len_v)
        scaled_attention, weights = sdp_attention(q, k, v, mask)

        # Transpose back: (batch_size, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenate heads: (batch_size, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))

        # Pass concatenated output through final linear projection
        output = self.linear(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, weights
