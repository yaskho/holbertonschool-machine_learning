#!/usr/bin/env python3
"""Dataset module for Machine Translation."""

from setup import load_pt2en
import transformers


class Dataset:
    """Loads and preps a dataset for machine translation."""

    def __init__(self):
        """Class constructor."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset.

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                tuples of tf.Tensors containing Portuguese and English text.

        Returns:
            tokenizer_pt, tokenizer_en: The trained Portuguese and English
            tokenizers.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_generator():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_generator():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_generator(),
            vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_generator(),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
