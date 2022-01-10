#!/usr/bin/env python
# coding: utf-8

"""Use this to generate train or dev files."""

import random

# Select language
LANGUAGE = "english"
LANG = "en"
TRAIN_DIR = "train_data/"+LANG+"/"
DEV_DIR = "dev_data/"+LANG+"/"

# Select number of sentences for each corpus
num_sents = [125, 250, 500, 1000, 2000]


def load_corpus(language):
    """
    Loads corpus files.
    """
    print(f"Loading {language} corpora...")
    # GUM (Georgetown University Multilayer) corpus from UD website
    corpus_a_dir = "data/ud-en-gum/"
    train_a = corpus_a_dir+"en_gum-ud-train.conllu"
    dev_a = corpus_a_dir+"en_gum-ud-dev.conllu"

    # wsj corpus with different conventions
    corpus_b_dir = "data/wsj-DIFF-CONVENTIONS/"
    train_b = corpus_b_dir+"train.conllu"
    dev_b = corpus_b_dir+"dev.conllu"

    print("Corpora loaded!")
    return train_a, dev_a, train_b, dev_b

corpus_a_train, corpus_a_dev, corpus_b_train, corpus_b_dev = load_corpus(LANGUAGE)


def create_train_partitions(ud_file, a_b):
    """
    Processes UD training data into randomly sampled training partitions of a fixed size.
    """
    with open(ud_file) as infile:
        ud_lines = infile.readlines()
    # get list of sentences
    sentences = []
    sentence = []
    for line in ud_lines:
        if line[0] == "#":
            continue
        elif len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = []
        else:
            split = line.split("\t")
            sentence.append(split)

    for size in num_sents:
        i = 1
        while i <=3:
            print(f"Creating train partition {i} of {size} sentences for {LANGUAGE} corpus {a_b}.")
            seed_filename = TRAIN_DIR+LANG+"_corpus="+a_b+"_sents="+str(size)+"_seed="+str(i)+".txt"
            random.shuffle(sentences)
            with open(seed_filename, "w") as outfile:
                for line in sentences[:size]:
                    for entry in line:
                        outfile.write("\t".join(entry))
                    outfile.write("\n")
            i += 1

create_train_partitions(corpus_a_train, "A")
create_train_partitions(corpus_b_train, "B")


def create_dev_partitions(ud_file, a_b):
    """
    Processes UD dev data into randomly sampled dev partitions of a fixed size.
    """
    with open(ud_file) as infile:
        ud_lines = infile.readlines()
    # get list of sentences
    sentences = []
    sentence = []
    for line in ud_lines:
        if line[0] == "#":
            continue
        elif len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = []
        else:
            split = line.split("\t")
            sentence.append(split)

    for size in num_sents:
        i = 1
        while i <=3:
            print(f"Creating dev partition {i} of {size} sentences for {LANGUAGE} corpus {a_b}.")
            seed_filename = DEV_DIR+LANG+"_corpus="+a_b+"_sents="+str(size)+"_seed="+str(i)+"-ud-dev.conllu"
            random.shuffle(sentences)
            with open(seed_filename, "w") as outfile:
                for line in sentences[:size]:
                    for entry in line:
                        outfile.write("\t".join(entry))
                    outfile.write("\n")
            i += 1

create_dev_partitions(corpus_a_dev, "A")
create_dev_partitions(corpus_b_dev, "B")
