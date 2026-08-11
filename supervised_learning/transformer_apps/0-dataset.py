#!/usr/bin/env python3
"""Dataset module for machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads and prepares the Portuguese-English dataset."""

    def __init__(self):
        """Class constructor."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates and trains sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence pairs

        Returns:
            tokenizer_pt, tokenizer_en
        """

        def pt_iterator():
            """Iterator over Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            """Iterator over English sentences."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        base_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        base_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        tokenizer_pt = base_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = base_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
