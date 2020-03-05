#!/usr/bin/env python

# to count tag combinations:
#   ./printtags.py ~/data/pat/train.conllu | sort | uniq -c | sort -n

import argparse
from conll import iter_conll

parser = argparse.ArgumentParser()
parser.add_argument('datafile')
parser.add_argument('--no-deprel', dest='deprel', action='store_false', help="don't print dependencies")
parser.add_argument('--no-pos', dest='pos', action='store_false', help="don't print relative positions")
args = parser.parse_args()

for sentence in iter_conll(args.datafile, verbose=False):
    for entry in sentence:
        if entry.id > 0:
            result = []
            if args.deprel:
                result.append(entry.deprel)
            if args.pos:
                result.append(str(entry.pos))
            print(' '.join(result))
