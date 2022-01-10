#!/usr/bin/env python
# coding: utf-8

"""This is for combining converted training files."""

LANGUAGE = "english"
partitions = [125, 250, 500, 1000, 2000]
seeds = [1, 2, 3]

DIRECTORY = "train_data/en/"

for partition in partitions:
    print(f"{partition*2} sentences")
    for seed in seeds:
        print(f"Seed {seed}")

        # basic combined A + B file
        INFILE1 = (DIRECTORY+LANGUAGE+"_train_corpus=A_sents="+
                   str(partition)+"_seed="+str(seed)+".txt")
        INFILE2 = (DIRECTORY+LANGUAGE+"_train_corpus=B_sents="+
                   str(partition)+"_seed="+str(seed)+".txt")
        OUTFILE1 = (DIRECTORY+LANGUAGE+"_train_A_B_sents="+
                    str(partition*2)+"_seed="+str(seed)+".conllu")

        with open(OUTFILE1, "w") as output:
            with open(INFILE1, "r") as input1:
                text = input1.read().strip()
                output.write(text)
            output.write("\n\n")
            with open(INFILE2, "r") as input2:
                text = input2.read().strip()
                output.write(text)

        # combined file A + B-converted
        INFILE3 = (DIRECTORY+LANGUAGE+"_train_corpus=A_sents="+
                   str(partition)+"_seed="+str(seed)+".txt")
        INFILE4 = (DIRECTORY+LANGUAGE+"_train_corpus=B_sents="+
                   str(partition)+"_seed="+str(seed)+"_converted_simple.conllu")
        INFILE4B = (DIRECTORY+LANGUAGE+"_train_corpus=B_sents="+
                    str(partition)+"_seed="+str(seed)+"_converted_pretrained=GloVe.conllu")
        INFILE4C = (DIRECTORY+LANGUAGE+"_train_corpus=B_sents="+
                    str(partition)+"_seed="+str(seed)+"_converted_BERT.conllu")
        OUTFILE2 = (DIRECTORY+LANGUAGE+"_train_A_B-converted-simple_sents="+
                    str(partition*2)+"_seed="+str(seed)+".conllu")
        OUTFILE2B = (DIRECTORY+LANGUAGE+"_train_A_B-converted-pretrained=GloVe_sents="+
                     str(partition*2)+"_seed="+str(seed)+".conllu")
        OUTFILE2C = (DIRECTORY+LANGUAGE+"_train_A_B-converted-BERT_sents="+
                     str(partition*2)+"_seed="+str(seed)+".conllu")

        with open(OUTFILE2, "w") as output:
            with open(INFILE3, "r") as input1:
                text = input1.read().strip()
                output.write(text)
            output.write("\n\n")
            with open(INFILE4, "r") as input2:
                text = input2.read().strip()
                output.write(text)

        with open(OUTFILE2B, "w") as output:
            with open(INFILE3, "r") as input1:
                text = input1.read().strip()
                output.write(text)
            output.write("\n\n")
            with open(INFILE4B, "r") as input2:
                text = input2.read().strip()
                output.write(text)

        with open(OUTFILE2C, "w") as output:
            with open(INFILE3, "r") as input1:
                text = input1.read().strip()
                output.write(text)
            output.write("\n\n")
            with open(INFILE4C, "r") as input2:
                text = input2.read().strip()
                output.write(text)
