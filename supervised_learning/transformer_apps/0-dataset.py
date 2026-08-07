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
        base_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_texts = [pt.numpy().decode('utf-8') for pt, _ in data]
        en_texts = [en.numpy().decode('utf-8') for _, en in data]

        vocab_size = 2 ** 13

        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            pt_texts, vocab_size=vocab_size
        )
        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            en_texts, vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
