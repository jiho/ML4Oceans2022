#!/bin/env python
#
# Remove cells which contain the string ## ANSWER and produce
# a new .ipynb without them = the student version.
#
# (c) 2022 Jean-Olivier Irisson, GNU General Public License v3

import sys
import nbformat

def remove_answers(file):
    # read the file
    nb = nbformat.read(file, as_version=4)

    for cell in nb.cells:
        if cell['source'].find('## ANSWER') != -1:
            # remove the content
            cell['source'] = ''
            # and the outputs
            cell.outputs = []

    # write as a new file
    nbformat.write(nb, file.replace('.ipynb', '_student.ipynb'))

if __name__ == '__main__':
    file = sys.argv[1]
    remove_answers(file)
    print('converted', file)
