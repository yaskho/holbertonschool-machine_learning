#!/usr/bin/env python3
"""Dataset class for Portuguese-English machine translation."""

from setup import load_pt2en
import transformers


class Dataset:
    """Loads the dataset and trains sub-word tokenizers."""

    def __init__(self):
        """Initialize the training/validation datasets and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English sub-word tokenizers.

        Args:
            data: tf.data.Dataset of (pt, en) sentence pairs.

        Returns:
            tokenizer_pt, tokenizer_en
        """
        pt_base = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        en_base = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def pt_iterator():
            """Yield Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def en_iterator():
            """Yield English sentences."""
            for _, en in data:
                yield en.numpy().decode("utf-8")

        tokenizer_pt = pt_base.train_new_from_iterator(
            pt_iterator(), vocab_size=2 ** 13
        )
        tokenizer_en = en_base.train_new_from_iterator(
            en_iterator(), vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
