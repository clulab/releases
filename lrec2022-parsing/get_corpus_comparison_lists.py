#!/usr/bin/env python
# coding: utf-8

"""Gets corpus comparison lists and applies conversions to make new converted conllu files."""

import functions as f


# Select language
LANGUAGE = "english"
LANG = "en"
TRAIN_DIR = "train_data/"+LANG+"/"

# Select number of sentences for each corpus
# Choose from 250, 500, 1000, 2000, 4000
NUM_SENTS = 250

# Select number of similar words to create new word pairs
TOP_N = 10

# Select threshold for filtering similar words; base model does not use a filter!
# THRESHOLD = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0,0]
THRESHOLD = 0.0

# Select seed for training data (1-3)
SEED = 1

# Load vectors

VECTORS, VECTOR_TYPE = f.load_vectors(LANGUAGE)
BERT_VECTORS = "vectors/"+LANGUAGE+"_corpus=A_sents="+str(NUM_SENTS)+"_seed="+str(SEED)+".magnitude"

# Get word-word-relation triples for each corpus

CORPUS_A = TRAIN_DIR+LANGUAGE+"_train_corpus=A_sents="+str(NUM_SENTS)+"_seed="+str(SEED)+".txt"
CORPUS_B = TRAIN_DIR+LANGUAGE+"_train_corpus=B_sents="+str(NUM_SENTS)+"_seed="+str(SEED)+".txt"

list_A, sentences_A = f.process_training_data(CORPUS_A)
list_B, sentences_B = f.process_training_data(CORPUS_B)

# Compare triples from each corpus to find mismatched relations

mismatches_A = {}
mismatches_B = {}
for pair_B in list_B.items():
    # If the (head, dependent) pair is in both corpus A and B
    if pair_B in list_A.items():
        # get the relations for that pair in B and in A
        relations_B = list_B[pair_B]
        relations_A = list_A[pair_B]
        # get the relations in B NOT in A and relations in A NOT in B
        not_in_A = [x for x in relations_B if x not in set(relations_A)]
        not_in_B = [x for x in relations_A if x not in set(relations_B)]
        # if there are relations not in A/B,
        # add entry to mismatches_A or mismatches_B for that pair-relation combo
        if len(not_in_A) != 0:
            mismatches_B[pair_B] = not_in_A
        if len(not_in_B) != 0:
            mismatches_A[pair_B] = not_in_B

print(f"{len(mismatches_A)} pairs with a relation in corpus A but not in corpus B")
print(f"{len(mismatches_B)} pairs with a relation in corpus B but not in corpus A")

# Generate Converted-Simple conversion dictionaries and create new conllu files
conversion_B_simple = f.get_conversions_simple(mismatches_B,
                                               sentences_B,
                                               list_B,
                                               list_A)
CONVERTED_CORPUS_B_SIMPLE = CORPUS_B[:-4]+"_converted_simple.conllu"
f.apply_conversions(CORPUS_B, CONVERTED_CORPUS_B_SIMPLE, conversion_B_simple)

# Generate Converted-GloVe conversion dictionaries and create new conllu files
conversion_B_pretrained = f.get_conversions_pretrained(mismatches_B,
                                                       sentences_B,
                                                       list_B,
                                                       list_A,
                                                       VECTORS,
                                                       TOP_N,
                                                       THRESHOLD)
CONVERTED_CORPUS_B_PRETRAINED = CORPUS_B[:-4]+"_converted_pretrained="+VECTOR_TYPE+".conllu"
f.apply_conversions(CORPUS_B, CONVERTED_CORPUS_B_PRETRAINED, conversion_B_pretrained)

# Generate Converted-BERT conversion dictionaries and create new conllu files
conversion_B_BERT = f.get_conversions_pretrained(mismatches_B,
                                                 sentences_B,
                                                 list_B,
                                                 list_A,
                                                 BERT_VECTORS,
                                                 TOP_N,
                                                 THRESHOLD)
CONVERTED_CORPUS_B_BERT = CORPUS_B[:-4]+"_converted_BERT.conllu"
f.apply_conversions(CORPUS_B, CONVERTED_CORPUS_B_BERT, conversion_B_BERT)
