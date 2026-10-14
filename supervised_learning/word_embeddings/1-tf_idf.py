#!/usr/bin/env python3
"""Module that creates a TF-IDF embedding"""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding

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

    s = len(sentences)
    f = len(vocab)
    index = {word: i for i, word in enumerate(vocab)}

    tf = np.zeros((s, f))
    for i, words in enumerate(tokenized):
        for word in words:
            if word in index:
                tf[i, index[word]] += 1

    df = np.sum(tf > 0, axis=0)
    idf = np.log((1 + s) / (1 + df)) + 1

    embeddings = tf * idf
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, np.array(vocab)
