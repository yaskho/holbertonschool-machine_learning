#!/usr/bin/env python3
"""
Module containing the Dataset class to load and prep a dataset
for Portuguese to English Machine Translation, including TensorFlow encoding.
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and prepares the Portuguese-to-English translation dataset,
    initializes sub-word tokenizers, encodes text, and maps to TF dataset.
    """

    def __init__(self):
        """
        Class constructor initializing datasets, tokenizers, and mapped splits.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates and trains sub-word tokenizers for Portuguese and English text.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence tuples.

        Returns:
            tokenizer_pt: Portuguese sub-word tokenizer
            tokenizer_en: English sub-word tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            """Yields Portuguese text in batches."""
            batch = []
            for pt, _ in data:
                batch.append(pt.numpy().decode('utf-8'))
                if len(batch) == 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def en_iterator():
            """Yields English text in batches."""
            batch = []
            for _, en in data:
                batch.append(en.numpy().decode('utf-8'))
                if len(batch) == 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        vocab_size = 2 ** 13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=vocab_size
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation sentence pair into tokens with start and
        end of sentence markers.

        Args:
            pt: tf.Tensor containing Portuguese sentence string
            en: tf.Tensor containing English sentence string

        Returns:
            pt_tokens: list of Portuguese token IDs
            en_tokens: list of English token IDs
        """
        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_ids = self.tokenizer_pt.encode(pt_str, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_str, add_special_tokens=False)

        pt_tokens = [pt_vocab_size] + pt_ids + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_ids + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method.

        Args:
            pt: tf.Tensor containing Portuguese sentence string
            en: tf.Tensor containing English sentence string

        Returns:
            pt_tokens: tf.Tensor of Portuguese token IDs with shape (None,)
            en_tokens: tf.Tensor of English token IDs with shape (None,)
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
