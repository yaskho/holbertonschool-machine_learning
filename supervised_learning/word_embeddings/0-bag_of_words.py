#!/usr/bin/env python3
"""Module that creates a bag of words embedding matrix"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences are used

    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) with the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    tokenized = []
    for sentence in sentences:
        sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
        tokenized.append(re.findall(r"\w+", sentence))

    if vocab is None:
        vocab = sorted(set(word for words in tokenized for word in words))

    index = {word: i for i, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in index:
                embeddings[i, index[word]] += 1

    return embeddings, np.array(vocab)
