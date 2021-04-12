import json
from random import sample

import sys
from pathlib import Path
import pickle
import numpy as np

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import linear_kernel # used for compute cosine similarity for sparse matrix
import os
from sklearn.metrics.pairwise import cosine_similarity


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def scores_to_rankings(input_numpy_array):
    output_ranking_list = list(reversed(np.argsort(input_numpy_array).tolist()))  # rankings of facts, from most relevant
    return output_ranking_list

def list_to_id_dict(instances_list):
    instances_dict = {}
    for instance in instances_list:
        sample_id = instance["id"]
        instances_dict[sample_id] = instance

    return instances_dict

def random_negative_from_kb(target_fact_num_list, kb_as_list, num_of_negative_facts):
    candidate_indexes = list(range(len(kb_as_list)))
    candidate_indexes_new = [x for x in candidate_indexes if x not in target_fact_num_list]
    selected_indexes = sample(candidate_indexes_new,num_of_negative_facts)

    return selected_indexes


def get_knowledge_base(kb_path: str):
    kb_data = list([])
    with open(kb_path, 'r') as the_file:
        kb_data = [line.strip() for line in the_file.readlines()]

    return kb_data


# Load questions as list of json files
def load_questions_json(question_path: str):
    questions_list = list([])
    with open(question_path, 'r', encoding='utf-8') as dataset:
        for i, line in enumerate(dataset):
            item = json.loads(line.strip())
            questions_list.append(item)

    return questions_list

def construct_dataset(train_path: str, dev_path: str, test_path: str, fact_path: str) -> (list, list, list):
    # This function is used to generate instances list for train, dev and test.
    def file_to_list(file_path: str, sci_facts: list) -> list:
        choice_to_id = {"A": 0, "B": 1, "C": 2, "D": 3}
        json_list = load_questions_json(file_path)

        instances_list = list([])
        for item in json_list:
            instance = {}
            instance["id"] = item["id"]
            for choice_id in range(4):
                if choice_id == choice_to_id[item['answerKey']]:
                    instance["text"] = item["question"]["stem"] + " " + item["question"]["choices"][choice_id]["text"]
                    gold_sci_fact = '\"' + item["fact1"] + '\"'
                    instance["label"] = sci_facts.index(gold_sci_fact)
            instances_list.append(instance)

        return instances_list

    sci_facts = get_knowledge_base(fact_path)

    train_list = file_to_list(train_path, sci_facts)
    dev_list = file_to_list(dev_path, sci_facts)
    test_list = file_to_list(test_path, sci_facts)

    return train_list, dev_list, test_list, sci_facts

def construct_retrieval_dataset_openbook():
    train_path = "data_raw/openbook/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl"
    dev_path = "data_raw/openbook/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl"
    test_path = "data_raw/openbook/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl"
    fact_path = "data_raw/openbook/OpenBookQA-V1-Sep2018/Data/Main/openbook.txt"

    # Build model:
    # Construct dataset
    train_raw, dev_raw, test_raw, sci_kb = construct_dataset(train_path, dev_path, test_path, fact_path)

    def add_distractor(instances_list, kb_as_list):
        instances_list_new = list([])
        for instance in instances_list:
            target_fact_num = instance["label"]
            negative_indices = random_negative_from_kb([target_fact_num], kb_as_list, 10)
            instance["documents"] = [target_fact_num]+negative_indices
            instance["query"] = [instance["text"]]
            instance["facts"] = [target_fact_num]
            instances_list_new.append(instance)

        return instances_list_new

    train_list = add_distractor(train_raw, sci_kb)
    dev_list = add_distractor(dev_raw, sci_kb)
    test_list = add_distractor(test_raw, sci_kb)

    print("openbook data constructed! train size:", len(train_list),"\tdev size:", len(dev_list),"\tkb size:", len(sci_kb))

    sci_kb_clean = [fact[1:-1] for fact in sci_kb]

    return train_list, dev_list, test_list, sci_kb_clean


def generate_query_string_for_useqa():

    train_list, dev_list, test_list, sci_kb = construct_retrieval_dataset_openbook()

    query_string = " ".join([instance["text"]+" \n" for instance in train_list])

    with open("openbook_train_query.txt", "w") as handle:
        handle.write(query_string)

    return 0
