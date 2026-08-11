#!/usr/bin/env python3
"""Dataset module for Portuguese to English machine translation."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and prepares the Portuguese-English translation dataset."""

    def __init__(self, batch_size, max_len):
        """Initialize and prepare the training and validation datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(
            self.tf_encode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.data_valid = self.data_valid.map(
            self.tf_encode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )

        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(
            20000
        )
        self.data_train = self.data_train.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64)
            )
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )

        self.data_valid = self.data_valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64)
            )
        )

    def tokenize_dataset(self, data):
        """
        Create subword tokenizers for Portuguese and English.

        Args:
            data: tf.data.Dataset containing (pt, en) pairs.

        Returns:
            The Portuguese and English tokenizers.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_data = data.map(lambda pt, en: pt).as_numpy_iterator()
        en_data = data.map(lambda pt, en: en).as_numpy_iterator()

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            (pt.decode('utf-8') for pt in pt_data),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            (en.decode('utf-8') for en in en_data),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a Portuguese-English translation pair.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.

        Returns:
            A tuple containing Portuguese and English token lists.
        """
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en, add_special_tokens=False
        )

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.

        Returns:
            Portuguese and English token tensors.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
