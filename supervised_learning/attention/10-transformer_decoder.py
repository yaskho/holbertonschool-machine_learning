#!/usr/bin/env python3
"""
Transformer Decoder Module
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class Decoder to create the decoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): Number of blocks in the decoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layer
            target_vocab (int): Size of the target vocabulary
            max_seq_len (int): Maximum sequence length possible
            drop_rate (float): Dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Executes the decoder computation graph

        Args:
            x: Tensor of shape (batch, target_seq_len) containing target inputs
            encoder_output: Tensor of shape (batch, input_seq_len, dm)
                            containing output of the encoder
            training: Boolean determining if model is training
            look_ahead_mask: Mask for the first multi-head attention layer
            padding_mask: Mask for the second multi-head attention layer

        Returns:
            Tensor of shape (batch, target_seq_len, dm) containing decoder output
        """
        seq_len = tf.shape(x)[1]

        # 1. Target embedding lookups
        x = self.embedding(x)

        # 2. Scale embeddings by sqrt(dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 3. Add positional encodings for the sequence length
        x += self.positional_encoding[:seq_len]

        # 4. Apply initial dropout
        x = self.dropout(x, training=training)

        # 5. Pass sequentially through N DecoderBlock layers
        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)

        return x
