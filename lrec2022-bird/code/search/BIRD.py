#!/usr/bin/env python
# coding: utf-8

'''
Reads several tsv files which are the results of path extraction 
phase. Then for a given path, finds the top 40 most similar paths 
according to the BIRD method.

Please note that the BERT model uses CUDA for faster execution.

Usage: 
python3 BIRD.py {BERT model name} {input query path} {mode: weighted/unweighted}
'''

from ast import literal_eval
import math
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizerFast, BertModel
import torch

pathids_to_paths = {}
paths_features = {}
paths_slotfreq = {}
xfeatures_totalfreqs = {}
yfeatures_totalfreqs = {}
xfeatures_paths = {}
yfeatures_paths = {}
xfeatures_totalfreqs_sum = 0
yfeatures_totalfreqs_sum = 0
sentenceids_to_sentences = {}
paths_sentences = {}
is_data_loaded = False

if len(sys.argv) != 4:
    print("Usage: python3 BIRD.py {BERT model name} {input query path} {mode: weighted/unweighted}")
    exit()

model_name = sys.argv[1]
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, return_dict=True)
model = model.to('cuda:0')
model.eval()

# mode: weighted/unweighted
mode = sys.argv[3]
if ((mode != "weighted") and (mode != "unweighted")):
    print('mode must be "weighted" or "unweighted"')
    exit()

BERT_HIDDEN_STATE_SIZE = model.config.hidden_size
BERT_HIDDEN_STATE_SIZE

def load_data():
    global pathids_to_paths
    global paths_features
    global paths_slotfreq
    global xfeatures_totalfreqs
    global yfeatures_totalfreqs
    global xfeatures_paths
    global yfeatures_paths
    global xfeatures_totalfreqs_sum
    global yfeatures_totalfreqs_sum
    global sentenceids_to_sentences
    global paths_sentences

    # read 'pathids_to_paths.tsv' and load it into pathids_to_paths
    # dictionary. This dictionary will contain a mapping from path ids
    # to paths.
    files_dir = "../data/"
    pathids_to_paths_file = files_dir + "pathids_to_paths.tsv"
    f = open(pathids_to_paths_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        path = literal_eval(fields[1])
        pathids_to_paths[path_id] = get_path_textual_string(path)
    f.close()

    # read 'paths_features.tsv' and load it into paths_features
    # dictionary. This dictionary will contain a mapping from 
    # path ids to a tuple (slotX and slotY). slotX is a dictionary
    # that contains the slot X features of a paths (slot-filler
    # words and their frequencies). Similarly, slotY contains
    # the slot Y features of a path.
    paths_features_file = files_dir + "paths_features.tsv"
    f = open(paths_features_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        slotX = literal_eval(fields[1])
        slotY = literal_eval(fields[2])
        paths_features[path_id] = (slotX,slotY)
    f.close()

    # read 'xfeatures_paths.tsv' and load it into xfeatures_paths
    # dictionary. This dictionary will contain a mapping from a word
    # to the list of the path ids of all the paths that the word is
    # a slot X slot-filler. This dictionary is required for speeding up
    # calculation of similary scores.
    xfeatures_paths_file = files_dir + "xfeatures_paths.tsv"
    f = open(xfeatures_paths_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        path_ids = literal_eval(fields[1])
        xfeatures_paths[word] = path_ids
    f.close()
    
    # read 'yfeatures_paths.tsv' and load it into yfeatures_paths
    # dictionary. This dictionary will contain a mapping from a word
    # to the list of the path ids of all the paths that the word is
    # a slot Y slot-filler. This dictionary is required for speeding up
    # calculation of similary scores.
    yfeatures_paths_file = files_dir + "yfeatures_paths.tsv"
    f = open(yfeatures_paths_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        path_ids = literal_eval(fields[1])
        yfeatures_paths[word] = path_ids
    f.close()
    
    # read 'paths_slotfreq.tsv' and load it into paths_slotfreq
    # dictionary. This dictionary will contain a mapping from
    # a path id to tuple (slotX_freq , slotY_freq). slotX_freq
    # is the sum of all of the word frequencies for slot X of the
    # path. Similarly, slotY_freq is the sum of all word frequencies
    # for slot Y of the path. The paths_slotfreq dictionary is required
    # for speeding up calculation of the similarity scores.
    paths_slotfreq_file = files_dir + "paths_slotfreq.tsv"
    f = open(paths_slotfreq_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        t = literal_eval(fields[1])
        paths_slotfreq[path_id] = t
    f.close()
    
    # read 'xfeatures_totalfreqs.tsv' and load it into xfeatures_totalfreqs
    # dictionary. This dictionary will contain a mapping from words
    # to the total number of frequencies the word was ever used in slot X 
    # of any path in the corpus. This dictionary is required for speeding up
    # calculation of the similarity scores.
    xfeatures_totalfreqs_file = files_dir + "xfeatures_totalfreqs.tsv"
    f = open(xfeatures_totalfreqs_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        freq = literal_eval(fields[1])
        xfeatures_totalfreqs[word] = freq
    f.close()

    # read 'yfeatures_totalfreqs.tsv' and load it into yfeatures_totalfreqs
    # dictionary. This dictionary will contain a mapping from words
    # to the total number of frequencies the word was ever used in slot Y 
    # of any path in the corpus. This dictionary is required for speeding up
    # calculation of the similarity scores.    
    yfeatures_totalfreqs_file = files_dir + "yfeatures_totalfreqs.tsv"
    f = open(yfeatures_totalfreqs_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        freq = literal_eval(fields[1])
        yfeatures_totalfreqs[word] = freq
    f.close()
    
    xfeatures_totalfreqs_sum = sum(xfeatures_totalfreqs.values())
    yfeatures_totalfreqs_sum = sum(yfeatures_totalfreqs.values())
    
    
    sentenceids_to_sentences_file = files_dir + "sentenceids_to_sentences.tsv"
    f = open(sentenceids_to_sentences_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        sen_id = literal_eval(fields[0])
        sentence = literal_eval(fields[1])
        sentenceids_to_sentences[sen_id] = sentence
    f.close()
    
    
    paths_sentences_file = files_dir + "paths_sentences.tsv"
    f = open(paths_sentences_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        sentences_info = literal_eval(fields[1])
        paths_sentences[path_id] = sentences_info
    f.close()


def main():
    global pathids_to_paths
    global paths_features
    global paths_slotfreq
    global xfeatures_totalfreqs
    global yfeatures_totalfreqs
    global xfeatures_paths
    global yfeatures_paths
    global xfeatures_totalfreqs_sum
    global yfeatures_totalfreqs_sum
    global sentenceids_to_sentences
    global paths_sentences
    global is_data_loaded
    
    # load data from tsv files
    if is_data_loaded == False:
        load_data()
        is_data_loaded = True

    # read an input path from command-line
    input_path_string = sys.argv[2]
    p = None
    for path_id, path_str in pathids_to_paths.items():
        if path_str == input_path_string:
            p = path_id
            break
    if p is None:
        print("The given path not was found in the paths database")
        exit_program()
    
    # find all the candidate paths
    print("\nFinding the candidate path...")
    candidate_paths = get_candidate_paths(p, paths_features, xfeatures_paths, yfeatures_paths)
    
    print("\nNumber of candidate paths:", len(candidate_paths))

    # filter the candidate paths if the number of their common 
    # features with the input path is less than a fixed percent 
    # (filtering_threshold_precentage) of the total number of 
    # features for the input path and the candidate path.
    print("\nFiltering the candidate paths...")
    filtered_candidate_paths = []
    filtering_threshold_precentage = 0.01
    (slotX , slotY) = paths_features[p]
    i = 0
    progressMileStone = 0.05
    total_candidate_paths = len(candidate_paths)
    for c in candidate_paths:
        if ((i/total_candidate_paths) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05
    
        common_features_count = 0
        
        for word in slotX:
            if (c in xfeatures_paths[word]):
                common_features_count += 1

        for word in slotY:
            if (c in yfeatures_paths[word]):
                common_features_count += 1

        total_features_count = len(slotX) + len(slotY) + len(paths_features[c][0]) + len(paths_features[c][1])

        if ((common_features_count * 2.0) / total_features_count) >= filtering_threshold_precentage:
            filtered_candidate_paths.append(c)
        
        i += 1
    print("100%")
    
    print("\nNumber of remaining candidate paths:", len(filtered_candidate_paths))
    
    # calculate similarity scores between the input path and the candidate paths
    print("\nCalculating similarity scores between the candidate paths and the input path...")
    (embedding_p_slotX , embedding_p_slotY) = Embedding(p)
    results = []
    i = 0
    mileStone = 0.01
    progressMileStone = mileStone
    total_filtered_candidate_paths = len(filtered_candidate_paths)
    for c in filtered_candidate_paths:
        if ((i/total_filtered_candidate_paths) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += mileStone
    
        similarity_score = S(embedding_p_slotX , embedding_p_slotY , c)
        if (similarity_score>0):
            results.append((c , similarity_score))
            
        i += 1
    print("100%")
        
    if (len(results)==0):
        print("\nNo results found in the corpus.")
        exit()
    
    # sort the resutls based on similarity score
    results.sort(key = lambda x: x[1], reverse=True)

    # print the top k results
    print_top_k = 40
    print("\nThe top " + str(print_top_k) + " results are:")
    for i in range(min(print_top_k,len(results))):
        pid = results[i][0]
        print(pathids_to_paths[pid])
        print("   Example sentences:")
        # also print two sentences of the path
        for i in [0 , -1]:
            print("   " , sentenceids_to_sentences[paths_sentences[pid][i][-1]])
        print()


# returns the textual representation of a given path
def get_path_textual_string(path):
    
    ret_val = ""
    last_printed_element = ""
    first_element_to_print = ""
    
    for t in path:
        if (t[3] == ">"):
            first_element_to_print = t[0]
            if (first_element_to_print == last_printed_element):
                first_element_to_print = ""
                       
            ret_val += first_element_to_print + "->" + t[2] + "->" + t[1]
            last_printed_element = t[1]
        else:
            first_element_to_print = t[1]
            if (first_element_to_print == last_printed_element):
                first_element_to_print = ""
                    
            ret_val += first_element_to_print + "<-" + t[2] + "<-" + t[0]
            last_printed_element = t[0]
            
    return ret_val[1:-1]


# calculates similarity between two given paths by calculating
# the arithmetic mean of the similarity scores of the slots
def S(embedding_p_slotX , embedding_p_slotY , c):
    
    (embedding_c_slotX , embedding_c_slotY) = Embedding(c)
    
    sim_slotX = cosine_similarity(embedding_p_slotX , embedding_c_slotX)
    
    sim_slotY = cosine_similarity(embedding_p_slotY , embedding_c_slotY)
    
    return (sim_slotX + sim_slotY) / 2.0 


def Embedding(p_id):
    
    global paths_sentences
    global sentenceids_to_sentences
    
    sentences = []
    slotX_start_char_indexes = []
    slotX_end_char_indexes = []
    slotY_start_char_indexes = []
    slotY_end_char_indexes = []
    slotX_words = []
    slotY_words = []

    for (slotX_start_char_idx, slotX_end_char_idx, slotX_word, slotY_start_char_idx, slotY_end_char_idx, slotY_word, sen_id) in paths_sentences[p_id]:
        
        sentences.append(sentenceids_to_sentences[sen_id])
        slotX_start_char_indexes.append(slotX_start_char_idx)
        slotX_end_char_indexes.append(slotX_end_char_idx)
        slotY_start_char_indexes.append(slotY_start_char_idx)
        slotY_end_char_indexes.append(slotY_end_char_idx)
        slotX_words.append(slotX_word)
        slotY_words.append(slotY_word)

    (slotX_embedding , slotY_embedding) = get_BERT_embeddings(p_id, sentences, slotX_words, slotY_words,
                                                              slotX_start_char_indexes, slotX_end_char_indexes, 
                                                              slotY_start_char_indexes, slotY_end_char_indexes)
    
    return (slotX_embedding.reshape(1, -1) , slotY_embedding.reshape(1, -1))


def get_BERT_embeddings(p_id, sentences, slotX_words, slotY_words, 
                        slotX_start_char_indexes, slotX_end_char_indexes, 
                        slotY_start_char_indexes, slotY_end_char_indexes):

    global tokenizer
    global model
    global BERT_HIDDEN_STATE_SIZE
    global mode
    
    slotX_sentences_embeddings = np.zeros((len(sentences),BERT_HIDDEN_STATE_SIZE))
    slotY_sentences_embeddings = np.zeros((len(sentences),BERT_HIDDEN_STATE_SIZE))
    
    slotX_weights = np.zeros(len(sentences))
    slotY_weights = np.zeros(len(sentences))
    
    batch_size = 64
    iterations = math.ceil(len(sentences) / batch_size) 
    
    for i in range(iterations):
        batch_start_index =  (i) * batch_size
        
        if (i == (iterations-1)):
            batch_end_index = len(sentences)
        else:
            batch_end_index =  (i+1) * batch_size
    
        inputs = tokenizer(sentences[batch_start_index:batch_end_index], padding=True, truncation=True, return_tensors="pt").to('cuda:0')
    
        with torch.no_grad():
            outputs = model(**inputs , output_hidden_states=False)

        last_hidden_states = outputs.last_hidden_state.cpu().numpy()
        
        # get slotX and slotY vectors for each sentence
        for j in range(batch_start_index,batch_end_index):
            
            k = j-batch_start_index
            
            slotX_start_token_idx = inputs.char_to_token( k , slotX_start_char_indexes[j] )
            slotX_end_token_idx = inputs.char_to_token( k , slotX_end_char_indexes[j]-1 ) + 1
            slotY_start_token_idx = inputs.char_to_token( k , slotY_start_char_indexes[j] )
            slotY_end_token_idx = inputs.char_to_token( k , slotY_end_char_indexes[j]-1 ) + 1
            
            slotX_states = last_hidden_states[ k , slotX_start_token_idx : slotX_end_token_idx , : ]
            slotX_states_avg = np.average( slotX_states , axis=0 )
            slotX_sentences_embeddings[j] = slotX_states_avg

            slotY_states = last_hidden_states[ k , slotY_start_token_idx : slotY_end_token_idx , : ]
            slotY_states_avg = np.average( slotY_states , axis=0 )
            slotY_sentences_embeddings[j] = slotY_states_avg
            
            if (mode == "weighted"):
                slotX_weights[j] = mi(p_id, 0, slotX_words[j])
                slotY_weights[j] = mi(p_id, 1, slotY_words[j])

    
    if (mode == "unweighted"):
        slotX_embedding = np.average(slotX_sentences_embeddings , axis=0)
        slotY_embedding = np.average(slotY_sentences_embeddings , axis=0)
    else:  #mode = "weighted"
        slotX_embedding = np.average(slotX_sentences_embeddings , axis=0 , weights=slotX_weights)
        slotY_embedding = np.average(slotY_sentences_embeddings , axis=0 , weights=slotY_weights)
    
    return (slotX_embedding , slotY_embedding)


# calculates the mutual information between the given slot of the 
# given path and its filler word according to equation (1) of DIRT paper
def mi(p_id, slot_idx, word):

    frequency_count_psw = paths_features[p_id][slot_idx][word]
    
    if slot_idx==0:
        frequency_count_NsN = xfeatures_totalfreqs_sum
    else:
        frequency_count_NsN = yfeatures_totalfreqs_sum
    
    frequency_count_psN = paths_slotfreq[p_id][slot_idx]
    
    if slot_idx==0:
        frequency_count_Nsw = xfeatures_totalfreqs[word]
    else:
        frequency_count_Nsw = yfeatures_totalfreqs[word]

    return math.log((frequency_count_psw * frequency_count_NsN)
                    /
                    (frequency_count_psN * frequency_count_Nsw))


# returns the list of candidate paths for a given path
def get_candidate_paths(p, paths_features, xfeatures_paths, yfeatures_paths):
    
    combined_path_ids = []
    
    (slotX , slotY) = paths_features[p]

    for word in slotX:
        combined_path_ids = combined_path_ids + xfeatures_paths[word]

    for word in slotY:
        combined_path_ids = combined_path_ids + yfeatures_paths[word]

    ret_val = list(set(combined_path_ids))
    
    ret_val.remove(p)
    
    return ret_val


def exit_program():
    exit()


if __name__ == "__main__":
    main()
