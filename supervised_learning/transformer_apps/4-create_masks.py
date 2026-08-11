#!/usr/bin/env python3
"""Module for creating Transformer attention masks."""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Create the masks used by the Transformer.

    Args:
        inputs: Tensor of shape (batch_size, seq_len_in).
        target: Tensor of shape (batch_size, seq_len_out).

    Returns:
        encoder_mask: Padding mask for the encoder.
        combined_mask: Padding and look-ahead mask for decoder self-attention.
        decoder_mask: Padding mask for decoder cross-attention.
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    target_padding_mask = tf.cast(
        tf.math.equal(target, 0), tf.float32
    )
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    seq_len = tf.shape(target)[1]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len)),
        -1,
        0
    )

    combined_mask = tf.maximum(
        target_padding_mask,
        look_ahead_mask
    )

    combined_mask = combined_mask[:, tf.newaxis, :, :]

    return encoder_mask, combined_mask, decoder_mask
