#!/usr/bin/env python3
"""Transformer model for Portuguese to English translation."""

import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """Generates positional encodings."""

    def __init__(self, dm, max_len):
        """Initialize the positional encoding."""
        super().__init__()

        position = tf.cast(
            tf.range(max_len)[:, tf.newaxis],
            tf.float32
        )
        div_term = tf.exp(
            tf.range(0, dm, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / dm)
        )

        pe = tf.zeros((max_len, dm), dtype=tf.float32)

        sin_values = tf.sin(position * div_term)
        cos_values = tf.cos(position * div_term)

        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.range(0, dm, 2)[:, tf.newaxis],
            tf.transpose(sin_values)
        )

        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.range(1, dm, 2)[:, tf.newaxis],
            tf.transpose(cos_values)
        )

        self.pos_encoding = pe[tf.newaxis, :, :]

    def call(self, x):
        """Add positional encoding to input."""
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head self-attention layer."""

    def __init__(self, dm, h):
        """Initialize the attention layer."""
        super().__init__()

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)
        self.wo = tf.keras.layers.Dense(dm)

    def call(self, q, k, v, mask=None):
        """Perform multi-head attention."""
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = tf.reshape(
            q, (batch_size, -1, self.h, self.depth)
        )
        k = tf.reshape(
            k, (batch_size, -1, self.h, self.depth)
        )
        v = tf.reshape(
            v, (batch_size, -1, self.h, self.depth)
        )

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scores += mask * -1e9

        weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(weights, v)
        output = tf.transpose(output, [0, 2, 1, 3])

        output = tf.reshape(output, (batch_size, -1, self.dm))

        return self.wo(output)


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(self, dm, h, hidden):
        """Initialize the encoder block."""
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self, x, mask=None):
        """Run the encoder block."""
        attention = self.mha(x, x, x, mask)
        x = self.norm1(x + attention)

        ffn = self.ffn(x)
        return self.norm2(x + ffn)


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""

    def __init__(self, dm, h, hidden):
        """Initialize the decoder block."""
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.norm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self, x, encoder_output, target_mask=None,
             encoder_mask=None):
        """Run the decoder block."""
        attention1 = self.mha1(
            x, x, x, target_mask
        )
        x = self.norm1(x + attention1)

        attention2 = self.mha2(
            x, encoder_output, encoder_output, encoder_mask
        )
        x = self.norm2(x + attention2)

        ffn = self.ffn(x)
        return self.norm3(x + ffn)


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_len):
        """Initialize the encoder."""
        super().__init__()

        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_vocab, dm
        )
        self.positional_encoding = PositionalEncoding(
            dm, max_len
        )

        self.blocks = [
            EncoderBlock(dm, h, hidden)
            for _ in range(N)
        ]

    def call(self, x, mask=None):
        """Run the encoder."""
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_len):
        """Initialize the decoder."""
        super().__init__()

        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            target_vocab, dm
        )
        self.positional_encoding = PositionalEncoding(
            dm, max_len
        )

        self.blocks = [
            DecoderBlock(dm, h, hidden)
            for _ in range(N)
        ]

    def call(self, x, encoder_output, target_mask=None,
             encoder_mask=None):
        """Run the decoder."""
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                target_mask,
                encoder_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Transformer model for machine translation."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_len):
        """Initialize the Transformer."""
        super().__init__()

        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_len
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_len
        )

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, encoder_mask=None,
             combined_mask=None, decoder_mask=None):
        """Run the Transformer."""
        encoder_output = self.encoder(
            inputs, encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            combined_mask,
            decoder_mask
        )

        return self.linear(decoder_output)
