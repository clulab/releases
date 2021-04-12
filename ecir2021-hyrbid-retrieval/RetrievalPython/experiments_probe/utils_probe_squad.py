import sys
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # used for compute cosine similarity for sparse matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression


parent_folder_path = str(Path('.').absolute().parent)
utils_folder_path = parent_folder_path+'/utils'
models_folder_path = parent_folder_path+'/models'
data_folder_path = parent_folder_path + '/data'

sys.path+=[parent_folder_path, utils_folder_path, models_folder_path, data_folder_path]

import numpy as np

import pickle
import torch
import time
import torch.nn as nn
import sklearn.preprocessing
import random
import datetime
import utils_dataset_squad


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def get_vocabulary(instances_train, knowledge_base, vocab_save_path, tfidf_vectorizer_save_path):
    stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                       "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                       "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                       "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                       "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                       "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                       "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
                       "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
                       "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                       "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                       "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    if os.path.exists(vocab_save_path):
        with open(vocab_save_path, "rb") as handle:
            vocab_dict = pickle.load(handle)
    else:
        wnl_lemmatizer = WordNetLemmatizer()
        vocab_count_dict = {}
        vocab_freq_dict = {}
        vocab_index_dict = {}
        vocab_dict = {}

        for instance in instances_train:
            query_lemmas = [wnl_lemmatizer.lemmatize(t) for t in word_tokenize(instance["query"][0].lower())]
            #vocab_keys.extend(query_lemmas)
            for query_lemma in query_lemmas:
                if query_lemma not in vocab_count_dict:
                    vocab_count_dict[query_lemma]=1
                vocab_count_dict[query_lemma]+=1

        for fact in knowledge_base:
            fact_lemmas = [wnl_lemmatizer.lemmatize(t) for t in word_tokenize(fact.lower())]
            for fact_lemma in fact_lemmas:
                if fact_lemma not in vocab_count_dict:
                    vocab_count_dict[fact_lemma]=1
                vocab_count_dict[fact_lemma]+=1

        # Remove stop words and special tokens.
        if "``" in vocab_count_dict:
            del vocab_count_dict["``"]
        if "\'\'" in vocab_count_dict:
            del vocab_count_dict["\'\'"]
        for stop_word in stop_words_list:
            if stop_word in vocab_count_dict:
                del vocab_count_dict[stop_word]

        vocab_count_dict = {k:vocab_count_dict[k] for k in sorted(vocab_count_dict, key=vocab_count_dict.get, reverse=True)}

        #vocab_keys = sorted(list(set(vocab_keys)-set(stop_words_list)))
        for value, vocab_key in enumerate(vocab_count_dict.keys()):
            if value<10000:
                vocab_index_dict[vocab_key] = value

        total_lemma_count = sum(vocab_count_dict.values())
        for vocab_key in vocab_count_dict.keys():
            vocab_freq_dict[vocab_key] = vocab_count_dict[vocab_key]**0.75/total_lemma_count

        vocab_dict["index_dict"] = vocab_index_dict
        vocab_dict["freq_dict"] = vocab_freq_dict

        with open(vocab_save_path, "wb") as handle:
            pickle.dump(vocab_dict, handle)

    if os.path.exists(tfidf_vectorizer_save_path):
        with open(tfidf_vectorizer_save_path, "rb") as handle:
            tfidf_vectorizer = pickle.load(handle)
    else:

        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list, tokenizer=LemmaTokenizer(), min_df = 5)
        doc_matrix = tfidf_vectorizer.fit_transform(knowledge_base)
        with open(tfidf_vectorizer_save_path, "wb") as handle:
            pickle.dump(tfidf_vectorizer, handle)

    print("target vocab dict loading finished, size ", len(vocab_dict["index_dict"]))
    print("`` in vocab:", "``" in vocab_dict["index_dict"], "\t \'\' in vocab:", "\'\'" in vocab_dict["index_dict"])
    print("vocab max:", max(vocab_dict["index_dict"].values()))
    return vocab_dict["index_dict"], tfidf_vectorizer

def get_negative_lemmas(input_lemmas_list, vocab_dict):
    input_lemmas_no_dup = set(input_lemmas_list)
    negative_lemmas_pool = set(vocab_dict.keys()) - input_lemmas_no_dup

    negative_list = random.sample(list(negative_lemmas_pool), len(input_lemmas_no_dup))

    return negative_list

def get_shuffled_vocab_dict(vocab_dict):
    original_keys = list(vocab_dict.keys())
    new_keys = [random.sample(original_keys,1)[0] for i in range(len(original_keys))]

    return {original_keys[i]: new_keys[i] for i in range(len(original_keys))}


# convert the raw instances to probe instances, including add tf-idf embedding, bert embedding, random embedding, training labels and random labels.
def instance_raw_to_probe(instances_list, kb,  useqa_embd_path, vocab_dict, vocab_dict_shuffled, tfidf_vectorizer, save_path):
    # three types of input embeddings:
    #   trained bert embedding
    #   tf-idf embedding
    #   random embedding
    # three types of labels:
    #   lemma indices of question and gold science fact;
    #   lemma indices of question and gold science fact of another question;
    #   lemma indices of question and gold science fact where the lemmas are replaced by a random function.

    with open(useqa_embd_path, "rb") as handle:
        useqa_embds = np.load(useqa_embd_path)


    wnl_lemmatizer = WordNetLemmatizer()

    instances_probe_list = list([])
    with torch.no_grad():
        for i, instance in enumerate(instances_list):
            instance_dict_probe = {}

            # Generate input embeddings
            query_useqa_embd = useqa_embds[i]
            instance_dict_probe["query_useqa_embd"] = torch.tensor(query_useqa_embd, dtype = torch.float32)
            instance_dict_probe["query_random_embd"] = torch.tensor(np.random.rand(512), dtype = torch.float32)
            instance_dict_probe["query_tfidf_embd"] = torch.tensor(tfidf_vectorizer.transform([instance["query"]]).todense(), dtype = torch.float32).squeeze()

            # Generate training labels for the probe task
            lemmas_query = [wnl_lemmatizer.lemmatize(t) for t in word_tokenize(instance["query"].lower())]
            tokens_fact = kb[instance["facts"]]
            lemmas_fact = [wnl_lemmatizer.lemmatize(t) for t in word_tokenize(tokens_fact.lower())]
            lemmas_negative = get_negative_lemmas(lemmas_query+lemmas_fact, vocab_dict)

            instance_dict_probe["lemmas_query"] = lemmas_query
            instance_dict_probe["lemmas_fact"] = lemmas_fact
            instance_dict_probe["lemmas_negative"] = lemmas_negative

            lemma_query_indices = [vocab_dict[lemma] for lemma in lemmas_query if lemma in vocab_dict]
            lemma_fact_indices = [vocab_dict[lemma] for lemma in lemmas_fact if lemma in vocab_dict]
            lemma_negative_indices = [vocab_dict[lemma] for lemma in lemmas_negative if lemma in vocab_dict]

            instance_dict_probe["lemma_query_indices_gold"] = list(set(lemma_query_indices))
            instance_dict_probe["lemma_fact_indices_gold"] = list(set(lemma_fact_indices))
            instance_dict_probe["lemma_negative_indices_gold"] = list(set(lemma_negative_indices))

            lemmas_query_token_remap = [vocab_dict_shuffled[lemma] for lemma in lemmas_query if lemma in vocab_dict_shuffled]+lemmas_query
            lemmas_fact_token_remap = [vocab_dict_shuffled[lemma] for lemma in lemmas_fact if lemma in vocab_dict_shuffled]+lemmas_fact
            lemmas_negative_token_remap = get_negative_lemmas(lemmas_query_token_remap+lemmas_fact_token_remap, vocab_dict)

            instance_dict_probe["lemma_query_indices_token_remap"] = list(set([vocab_dict[lemma] for lemma in lemmas_query_token_remap if lemma in vocab_dict]))
            instance_dict_probe["lemma_fact_indices_token_remap"] = list(set([vocab_dict[lemma] for lemma in lemmas_fact_token_remap if lemma in vocab_dict]))
            instance_dict_probe["lemma_negative_indices_token_remap"] = list(set([vocab_dict[lemma] for lemma in lemmas_negative_token_remap if lemma in vocab_dict]))

            # Generate question id
            instance_dict_probe["id"] = instance["id"]

            a = (len(instance_dict_probe["lemma_query_indices_gold"]) != 0)
            b = (len(instance_dict_probe["lemma_fact_indices_gold"]) != 0)
            c = (len(instance_dict_probe["lemma_negative_indices_gold"]) != 0)

            if a and b and c:

                instances_probe_list.append(instance_dict_probe)

        resample_control_indices = list(range(len(instances_probe_list)))
        random.shuffle(resample_control_indices)
        for i, instance in enumerate(instances_probe_list):
            instance["lemma_query_indices_ques_shuffle"] = instances_probe_list[resample_control_indices[i]]["lemma_query_indices_gold"]
            instance["lemma_fact_indices_ques_shuffle"] = instances_probe_list[resample_control_indices[i]]["lemma_fact_indices_gold"]
            instance["lemma_negative_indices_ques_shuffle"] = instances_probe_list[resample_control_indices[i]]["lemma_negative_indices_gold"]

    return instances_probe_list

def get_probe_dataset(train_list, dev_list,  kb, useqa_embd_paths, vocab_dict, tfidf_vectorizer, save_path, dataset_name):
    # generate data for 5 random seeds;
    # generate data for train, test and dev.
    # assert (len(model_paths) == 5)

    if os.path.exists(save_path+dataset_name):
        with open(save_path+dataset_name, "rb") as handle:
            instances_list_all_seeds = pickle.load(handle)

    else:
        instances_list_all_seeds = list([])

        # TODO: change this back to 5 before formal experiments.
        for random_seed in range(5):
            print("Generating probe dataset of seed ", random_seed, " ...")
            # Set the random seed to ensure reproducibility
            np.random.seed(random_seed)  # the scope of numpy random seed includes different functions
            random.seed(random_seed) # the scope of random seed includes different functions

            # Shuffle the mapping of the vocab dictionary
            vocab_dict_shuffled = get_shuffled_vocab_dict(vocab_dict)

            # Start building the probe dataset
            instances_list_one_seed = {}
            instances_list_one_seed["train"] = instance_raw_to_probe(train_list, kb, useqa_embd_paths["train"], vocab_dict, vocab_dict_shuffled, tfidf_vectorizer, save_path)
            instances_list_one_seed["dev"] = instance_raw_to_probe(dev_list, kb, useqa_embd_paths["dev"],  vocab_dict, vocab_dict_shuffled, tfidf_vectorizer, save_path)
            instances_list_one_seed["remapping_dict"] = vocab_dict_shuffled
            instances_list_all_seeds.append(instances_list_one_seed)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(save_path+dataset_name, "wb") as handle:
            pickle.dump(instances_list_all_seeds, handle)

    return instances_list_all_seeds

def build_vocab_occurrence_dict(instances_list, sci_kb, lemmatizer):
    vocab_count_dict = {}

    for instance in instances_list:
        lemmas_query = [lemmatizer.lemmatize(t) for t in word_tokenize(instance["query"][0].lower())]
        tokens_fact = sci_kb[instance["facts"][0]]
        lemmas_fact = [lemmatizer.lemmatize(t) for t in word_tokenize(tokens_fact.lower())]
        lemmas_all = lemmas_query+lemmas_fact

        for lemma in lemmas_all:
            if lemma not in vocab_count_dict:
                vocab_count_dict[lemma] = 1
            vocab_count_dict[lemma] += 1

    return vocab_count_dict

# This function is used to get the statistics of the training labels.
def get_vocab_statistics():
    if not os.path.exists("data/"):
        os.mkdir("data/")

    # train_list, dev_list, kb = utils_dataset_openbook.construct_retrieval_dataset_openbook()
    #
    # wnl_lemmatizer = WordNetLemmatizer()
    #
    # vocab_dict_all,_ = get_vocabulary(train_list, sci_kb, "data/vocab_dict.pickle", "data/tfidf_vectorizer.pickle")
    #
    # vocab_count_train = build_vocab_occurrence_dict(train_list, sci_kb, wnl_lemmatizer)
    # vocab_count_dev = build_vocab_occurrence_dict(dev_list, sci_kb, wnl_lemmatizer)
    # vocab_count_test = build_vocab_occurrence_dict(test_list, sci_kb, wnl_lemmatizer)
    #
    # total_lemmas_train = sum(list(vocab_count_train.values()))
    # total_unique_lemmas_train = len(vocab_count_train)
    #
    # total_lemmas_dev = sum(list(vocab_count_dev.values()))
    # total_unique_lemmas_dev = len(vocab_count_dev)
    #
    # total_lemmas_test = sum(list(vocab_count_test.values()))
    # total_unique_lemmas_test = len(vocab_count_test)
    #
    # dev_novel_lemmas = sum([vocab_count_dev[lemma] for lemma in vocab_count_dev if lemma not in vocab_count_train])
    # dev_novel_lemmas_unique = sum([1 for lemma in vocab_count_dev if lemma not in vocab_count_train])
    #
    # test_novel_lemmas = sum([vocab_count_test[lemma] for lemma in vocab_count_test if lemma not in vocab_count_train])
    # test_novel_lemmas_unique = sum([1 for lemma in vocab_count_test if lemma not in vocab_count_train])
    #
    # print("dev novel lemmas:", dev_novel_lemmas, "\ttotal:", sum(list(vocab_count_dev.values())))
    # print("dev novel lemmas unique:", dev_novel_lemmas_unique, "\ttotal:", len(vocab_count_dev))
    # print("test novel lemmas:", test_novel_lemmas, "\ttotal:", sum(list(vocab_count_test.values())))
    # print("test novel lemmas unique:", test_novel_lemmas_unique, "\ttotal:", len(vocab_count_test))

    #print(vocab_count_dict)

    # occurrence_count = np.array([dict_item[1] for dict_item in vocab_count_dict.items()])
    #
    # for dict_item in  vocab_count_dict.items():
    #     if dict_item[1]>90:
    #         print(dict_item)
    #         input("AAA")
    #
    # print(np.histogram(occurrence_count, bins=np.arange(100)))
    #
    # print(len(vocab_count_dict))

    return 0

#get_vocab_statistics()



