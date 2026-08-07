#!/usr/bin/env python3
"""
Module containing the Dataset class to load and prep a dataset
for Portuguese to English Machine Translation.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and prepares the Portuguese-to-English translation dataset,
    and initializes training sub-word tokenizers.
    """

    def __init__(self):
        """
        Class constructor initializing datasets and tokenizers.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates and trains sub-word tokenizers for Portuguese and English text.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence tuples.

        Returns:
            tokenizer_pt: Portuguese sub-word tokenizer
            tokenizer_en: English sub-word tokenizer
        """
        # Load pre-trained fast tokenizers
        base_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        # Generators for training tokenizers from dataset
        def pt_iterator():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Train new tokenizers with max vocab size 2^13 (8192)
        vocab_size = 2 ** 13

        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=vocab_size
        )
        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
