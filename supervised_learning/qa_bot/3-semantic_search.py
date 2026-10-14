#!/usr/bin/env python3
"""Semantic Search module using TensorFlow Hub Universal Sentence Encoder."""

import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of reference documents.

    Args:
        corpus_path (str): Path to the folder containing reference documents.
        sentence (str): Query sentence to perform semantic search with.

    Returns:
        str: Text content of the document most similar to sentence.
    """
    model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    documents = []
    # Read all files in the corpus path
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())

    if not documents:
        return None

    # Generate embeddings for query sentence and corpus documents
    embeddings = model([sentence] + documents)

    # Compute inner product / cosine similarity between query and documents
    query_embedding = embeddings[0:1]
    doc_embeddings = embeddings[1:]

    similarities = np.inner(query_embedding, doc_embeddings)[0]
    best_doc_idx = np.argmax(similarities)

    return documents[best_doc_idx]
