#!/usr/bin/env python3
"""Multi-reference Question Answering Bot module."""

question_answer_single = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """Answer user questions using a corpus of reference documents.

    Args:
        corpus_path (str): Path to directory containing reference documents.
    """
    exit_commands = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        try:
            user_input = input('Q: ')
        except (KeyboardInterrupt, EOFError):
            print('A: Goodbye')
            break

        if user_input.strip().lower() in exit_commands:
            print('A: Goodbye')
            break

        # Find most relevant reference document using semantic search
        reference = semantic_search(corpus_path, user_input)

        if reference is None:
            print('A: Sorry, I do not understand your question.')
            continue

        # Extract answer snippet from retrieved reference text
        answer = question_answer_single(user_input, reference)

        if answer is None or answer.strip() == '':
            print('A: Sorry, I do not understand your question.')
        else:
            print('A:', answer)
