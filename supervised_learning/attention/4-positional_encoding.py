#!/usr/bin/env python3
"""Module containing the positional_encoding function."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model depth (d_model).

    Returns:
        np.ndarray: Array of shape (max_seq_len, dm) containing
            positional encoding vectors.
    """
    PE = np.zeros((max_seq_len, dm))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    div_term = 10000 ** (2 * (i // 2) / dm)

    PE[:, 0::2] = np.sin(pos / div_term[:, 0::2])
    PE[:, 1::2] = np.cos(pos / div_term[:, 1::2])

    return PE
