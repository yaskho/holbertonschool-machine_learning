#!/usr/bin/env python3
"""Module containing the SelfAttention class for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates Bahdanau attention for machine translation."""

    def __init__(self, units):
        """Class constructor for SelfAttention.

        Args:
            units (int): Number of hidden units in the alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates attention context vector and weights.

        Args:
            s_prev (tf.Tensor): Tensor of shape (batch, units) containing
                the previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape (batch, input_seq_len,
                units) containing the outputs of the encoder.

        Returns:
            context (tf.Tensor): Tensor of shape (batch, units) containing
                the context vector for the decoder.
            weights (tf.Tensor): Tensor of shape (batch, input_seq_len, 1)
                containing the attention weights.
        """
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) +
                                  self.U(hidden_states)))

        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
