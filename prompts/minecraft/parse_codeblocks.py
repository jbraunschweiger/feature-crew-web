#!/usr/bin/env python3
# parse_codeblocks.py

import os
import re
import argparse

def clean_path_line(line: str) -> str:
    """
    Remove leading/trailing whitespace, asterisks and backticks,
    then strip any leading slash so we stay inside the project.
    """
    s = line.strip()
    # remove markdown “strong” markers (**…**)
    if s.startswith('**') and s.endswith('**'):
        s = s[2:-2].strip()
    # remove backticks (`…`)
    if s.startswith('`') and s.endswith('`'):
        s = s[1:-1].strip()
    # now strip any leading slash
    return s.lstrip(os.sep)

def main():
    parser = argparse.ArgumentParser(
        description='Extract Markdown codeblocks from a TXT file and write each to its preceding file path.'
    )
    parser.add_argument('input_file',
                        help='Path to the TXT file containing paths + ```code``` blocks')
    args = parser.parse_args()

    # after cleaning, must look like a relative path with an extension
    path_re = re.compile(r'^[\w\-/\.]+$')
    fence_re = re.compile(r'^```')  # start/end of a fenced code block

    file_path = None
    in_block = False
    buffer = []

    with open(args.input_file, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n')

            if not in_block:
                # try to interpret this line as a path declaration
                cleaned = clean_path_line(line)
                if path_re.match(cleaned) and '.' in cleaned:
                    file_path = cleaned
                    continue

                # next non-path line—if it’s a fence, start capturing
                if file_path and fence_re.match(line.strip()):
                    in_block = True
                    buffer = []
                    continue

            else:
                # closing fence: flush buffer to file
                if fence_re.match(line):
                    dir_name = os.path.dirname(file_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    with open(file_path, 'w') as out:
                        out.write('\n'.join(buffer))
                    print(f'Wrote: {file_path}')
                    # reset
                    file_path = None
                    in_block = False
                    buffer = []
                else:
                    buffer.append(line)

if __name__ == '__main__':
    main()