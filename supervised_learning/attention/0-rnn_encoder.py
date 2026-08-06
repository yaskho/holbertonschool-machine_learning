#!/usr/bin/env python3
"""Module containing the RNNEncoder class for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class for encoding sequences in machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor for RNNEncoder.

        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Initializes the hidden state for the RNN cell to a tensor of zeros.

        Returns:
            tf.Tensor: Tensor of shape (batch, units) containing zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Passes input tensor and initial hidden state through the encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len)
                containing word indices within the vocabulary.
            initial (tf.Tensor): Initial hidden state tensor of shape
                (batch, units).

        Returns:
            outputs (tf.Tensor): Tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder.
            hidden (tf.Tensor): Tensor of shape (batch, units) containing
                the last hidden state of the encoder.
        """
        x_embed = self.embedding(x)
        outputs, hidden = self.gru(x_embed, initial_state=initial)
        return outputs, hidden
