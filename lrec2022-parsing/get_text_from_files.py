#!/usr/bin/env python
# coding: utf-8

"""This script gets the text from a conllu-style file to use for generating BERT embeddings"""

LANGUAGE = "english"
LANG = "en"
TRAIN_DIR = "train_data/"+LANG+"/"
TEXT_DIR = "text_files/"+LANG+"/"
seeds = [1, 2, 3]
sentences = [125, 250, 500, 1000, 2000]

# loop over each file
for sents in sentences:
    for seed in seeds:
        INPUT_FILE = LANG+"_corpus=A_sents="+str(sents)+"_seed="+str(seed)+".txt"
        TEXT_FILE = TEXT_DIR+LANGUAGE+"_corpus=A_sents="+str(sents)+"_seed="+str(seed)+"_text.txt"
        with open(TRAIN_DIR+INPUT_FILE, "r") as infile:
            text = infile.read()
        lines = text.strip().split("\n\n")
        all_texts = []
        for line in lines:
            sent = []
            for item in line.strip().split("\n"):
                sent.append(item.split("\t")[1])
            all_texts.append(" ".join(sent))
        for item in all_texts:
            print(item)
        with open(TEXT_FILE, "w") as outfile:
            for line in all_texts:
                outfile.write(line)
                outfile.write("\n")
