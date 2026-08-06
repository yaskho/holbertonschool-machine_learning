#!/usr/bin/env python3
"""Module to calculate scaled dot product attention."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention.

    Args:
        Q (tf.Tensor): Query matrix with shape (..., seq_len_q, dk).
        K (tf.Tensor): Key matrix with shape (..., seq_len_v, dk).
        V (tf.Tensor): Value matrix with shape (..., seq_len_v, dv).
        mask (tf.Tensor, optional): Mask tensor broadcastable to
            (..., seq_len_q, seq_len_v). Defaults to None.

    Returns:
        output (tf.Tensor): Scaled dot product attention tensor with
            shape (..., seq_len_q, dv).
        weights (tf.Tensor): Attention weights tensor with shape
            (..., seq_len_q, seq_len_v).
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
