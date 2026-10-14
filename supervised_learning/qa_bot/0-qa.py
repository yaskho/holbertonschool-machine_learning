#!/usr/bin/env python3
"""Question Answering module using BERT and TensorFlow Hub."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Find a snippet of text in a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): Reference document text containing the answer.

    Returns:
        str: Extracted answer snippet, or None if no answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    # Tokenize input question and reference context
    inputs = tokenizer(question, reference, return_tensors='tf')
    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']

    # Predict start and end logits using TF-Hub QA model
    outputs = model([input_word_ids, input_mask, input_type_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]

    # Find token indices with maximum probability
    start_idx = tf.argmax(start_logits, axis=-1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=-1).numpy()[0]

    # Return None if indices are invalid or point to [CLS] token (no answer)
    if start_idx > end_idx or start_idx == 0:
        return None

    # Convert token IDs to token strings and extract answer span
    input_ids = input_word_ids.numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_tokens = tokens[start_idx:end_idx + 1]

    if not answer_tokens:
        return None

    answer = ' '.join(answer_tokens)
    if not answer or answer == '[CLS]':
        return None

    return answer
