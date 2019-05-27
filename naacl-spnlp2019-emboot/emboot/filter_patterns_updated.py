#!/usr/bin/env python
#### ~/anaconda/bin/python filter_patterns_updated.py 5 data/ontonotes_pattern_vocabulary.txt data/ontonotes_pattern_vocabulary_pruned.txt data/ontonotes_training_data.txt data/ontonotes_training_data_pruned.txt data/ontonotes_entity_vocabulary_pruned.txt


import io
import sys
from vocabulary import Vocabulary


n = int(sys.argv[1]) ## NOTE: This filters patterns which are <= n in the set of patterns in the data or retains patterns which are > n in the dataset
patternVocabFile = str(sys.argv[2])
patternVocabPrunedFile = str(sys.argv[3])
dataFile = str(sys.argv[4])
dataPrunedFile = str(sys.argv[5])
entityVocabPrunedFile = str(sys.argv[6])

patterns = Vocabulary.from_file(patternVocabFile) ####'conll_pattern_vocabulary.txt')
patterns.prepare(min_count=n)
patterns.to_file(patternVocabPrunedFile)  ####'conll_pattern_vocabulary_pruned.txt')

entities = Vocabulary()

#####with io.open('conll_training_data.txt') as f, io.open('conll_training_data_pruned.txt', 'w') as out:
with io.open(dataFile) as f, io.open(dataPrunedFile, 'w') as out:
    i = 0
    for line in f:
        fields = line.split('\t')
        label = fields[0]
        entity = fields[1]
        pats = [p for p in fields[2:] if patterns.contains(p)]
        print("I = " + str(i))
        i+=1
        if pats:
            entities.add(entity)
            fields = [label] + [entity] + pats
            out.write('\t'.join(fields) + '\n')

entities.prepare()
####entities.to_file('conll_entity_vocabulary_pruned.txt')
entities.to_file(entityVocabPrunedFile)
