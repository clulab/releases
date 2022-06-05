#!/usr/bin/env python
# coding: utf-8

# Generates "sentence_chars_to_BERTtokens_indexes.tsv"

# Supposed to be used only in the BERT mode of path extraction

# Supposed to be run by the script "call_create_BERTtoken_mappings"
# It activates the appropriate virtual environment for this file and
# then runs this file

from ast import literal_eval
import sys
from transformers import BertTokenizerFast
import torch

sentence_chars_to_BERTtokens_indexes = {}
sentenceids_to_sentences = {}

def main():
    
    global sentence_chars_to_BERTtokens_indexes
    global sentenceids_to_sentences
    
    if len(sys.argv) != 2:
        print("Please provide the BERT model name as a command line argument:")
        print("Usage: python3 create_BERTtoken_mappings.py {BERT model name}")
        exit()
        
    model_name = sys.argv[1]

    files_dir = "../data/"
    sentenceids_to_sentences = {}
    sentenceids_to_sentences_file = files_dir + "sentenceids_to_sentences.tsv"
    f = open(sentenceids_to_sentences_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        sen_id = literal_eval(fields[0])
        sentence = literal_eval(fields[1])
        sentenceids_to_sentences[sen_id] = sentence
    f.close()
    
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    
    total_sentences = len(sentenceids_to_sentences)
    progressMileStone = 0.05
    counter = 0
    
    print("Creating a dictionary for a mapping from sentence ids to their BERT tokens indexes...")
    sentence_chars_to_BERTtokens_indexes = {}
    
    for sen_id , sentence in sentenceids_to_sentences.items():
        
        if ((counter/total_sentences) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05
            
        inputs = tokenizer(sentence, return_tensors="pt")
        
        chars_to_BERTtokens_indexes = {}
        
        for i in range(len(sentence)):
            BERT_token_idx = inputs.char_to_token(i)
            if BERT_token_idx is None:
                BERT_token_idx = -1
            chars_to_BERTtokens_indexes[i] = BERT_token_idx
            
        sentence_chars_to_BERTtokens_indexes[sen_id] = chars_to_BERTtokens_indexes
            
        counter += 1
        
    print("100%")
        
    
    print("\nWriting sentence_chars_to_BERTtokens_indexes.tsv to disk...")
    sentence_chars_to_BERTtokens_indexes_file = files_dir + "sentence_chars_to_BERTtokens_indexes.tsv"
    f = open(sentence_chars_to_BERTtokens_indexes_file, mode="w", encoding="utf_8")
    
    counter = 0
    progressMileStone = 0.05
    
    for sen_id , chars_to_BERTtokens_indexes in sentence_chars_to_BERTtokens_indexes.items():
        
        if ((counter/total_sentences) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05

        second_col_str = "{"
        for char_idx , BERT_token_idx in chars_to_BERTtokens_indexes.items():
            second_col_str += str(char_idx) + ":" + str(BERT_token_idx) + ","
        if (len(second_col_str) > 1):
            second_col_str = second_col_str[:-1]
        second_col_str += "}"
        
        f.write(str(sen_id) +
                "\t" +
                second_col_str +
                "\n"
               )
        
        counter += 1
    
    f.close()
    print("100%")


if __name__ == "__main__":
    main()
