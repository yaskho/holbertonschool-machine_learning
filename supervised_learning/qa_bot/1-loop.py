#!/usr/bin/env python3
"""Interactive QA loop module."""


def qa_loop():
    """Handle interactive user Q&A loop until an exit command is entered."""
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

        print('A:')


if __name__ == '__main__':
    qa_loop()
