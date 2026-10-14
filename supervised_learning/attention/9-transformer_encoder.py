#!/usr/bin/env python3
"""
Transformer Encoder Module
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class Encoder to create the encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): Number of blocks in the encoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layer
            input_vocab (int): Size of the input vocabulary
            max_seq_len (int): Maximum sequence length possible
            drop_rate (float): Dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Executes the encoder computation graph

        Args:
            x: Tensor of shape (batch, input_seq_len) containing input tokens
            training: Boolean determining if model is training
            mask: Mask to be applied for multi-head attention

        Returns:
            Tensor of shape (batch, input_seq_len, dm) containing encoder output
        """
        seq_len = tf.shape(x)[1]

        # Pass input through embedding layer
        x = self.embedding(x)

        # Scale embeddings by sqrt(dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding for sequence length
        x += self.positional_encoding[:seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through N EncoderBlocks
        for block in self.blocks:
            x = block(x, training, mask)

        return x
