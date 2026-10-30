#!/usr/bin/env python3
"""
Dataset module for machine translation task
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """
    Dataset class that loads and prepares Portuguese to English translation
    datasets using HuggingFace subword tokenizers.
    """

    def __init__(self):
        """
        Initializes dataset instances and tokenizers
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for Portuguese and English from dataset

        Args:
            data: tf.data.Dataset yielding (pt, en) tuple pairs

        Returns:
            tokenizer_pt: Portuguese subword tokenizer
            tokenizer_en: English subword tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            (pt.numpy().decode('utf-8') for pt, en in data),
            vocab_size=2**13
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            (en.numpy().decode('utf-8') for pt, en in data),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a Portuguese and English translation pair into tokens

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens: list containing Portuguese tokens with start/end tokens
            en_tokens: list containing English tokens with start/end tokens
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = (
            [self.tokenizer_pt.vocab_size]
            + self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
            + [self.tokenizer_pt.vocab_size + 1]
        )

        en_tokens = (
            [self.tokenizer_en.vocab_size]
            + self.tokenizer_en.encode(en_text, add_special_tokens=False)
            + [self.tokenizer_en.vocab_size + 1]
        )

        return pt_tokens, en_tokens
