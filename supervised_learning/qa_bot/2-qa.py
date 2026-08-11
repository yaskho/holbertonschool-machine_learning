#!/usr/bin/env python3
"""Interactive Question Answering loop module on a reference text."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer user questions continuously from a reference document.

    Args:
        reference (str): Reference document text containing information.
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

        answer = question_answer(user_input, reference)

        if answer is None or answer.strip() == '':
            print('A: Sorry, I do not understand your question.')
        else:
            print('A:', answer)
