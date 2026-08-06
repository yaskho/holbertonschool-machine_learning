#!/usr/bin/env python3
"""Module containing the RNNDecoder class for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class for decoding sequences in machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor for RNNDecoder.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Passes target sequence word, hidden state, and encoder output.

        Args:
            x (tf.Tensor): Tensor of shape (batch, 1) containing previous
                word index in the target sequence.
            s_prev (tf.Tensor): Tensor of shape (batch, units) containing
                previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape (batch, input_seq_len,
                units) containing outputs of the encoder.

        Returns:
            y (tf.Tensor): Tensor of shape (batch, vocab) containing output
                word predictions.
            s (tf.Tensor): Tensor of shape (batch, units) containing new
                decoder hidden state.
        """
        context, weights = self.attention(s_prev, hidden_states)
        x_embed = self.embedding(x)
        context_expanded = tf.expand_dims(context, 1)
        x_concat = tf.concat([context_expanded, x_embed], axis=-1)

        output, s = self.gru(x_concat, initial_state=s_prev)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, s
