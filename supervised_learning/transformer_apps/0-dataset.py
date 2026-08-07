#!/usr/bin/env python3
"""
Module for loading and preparing Portuguese-to-English translation dataset.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and preps a dataset for Portuguese-to-English machine translation.
    """

    def __init__(self):
        """
        Initializes dataset splits and Portuguese/English tokenizers.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for Portuguese and English text.

        Args:
            data: tf.data.Dataset of (pt, en) sentence tuples

        Returns:
            tokenizer_pt, tokenizer_en: trained sub-word tokenizers
        """
        pt_model = 'neuralmind/bert-base-portuguese-cased'
        en_model = 'bert-base-uncased'

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(pt_model)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(en_model)

        def pt_iterator():
            """Yields Portuguese strings safely from dataset."""
            dataset_iter = (data.as_numpy_iterator()
                            if hasattr(data, 'as_numpy_iterator')
                            else data)
            for pt, _ in dataset_iter:
                if isinstance(pt, bytes):
                    yield pt.decode('utf-8')
                elif hasattr(pt, 'numpy'):
                    yield pt.numpy().decode('utf-8')
                else:
                    yield str(pt)

        def en_iterator():
            """Yields English strings safely from dataset."""
            dataset_iter = (data.as_numpy_iterator()
                            if hasattr(data, 'as_numpy_iterator')
                            else data)
            for _, en in dataset_iter:
                if isinstance(en, bytes):
                    yield en.decode('utf-8')
                elif hasattr(en, 'numpy'):
                    yield en.numpy().decode('utf-8')
                else:
                    yield str(en)

        vocab_size = 2 ** 13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=vocab_size
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
