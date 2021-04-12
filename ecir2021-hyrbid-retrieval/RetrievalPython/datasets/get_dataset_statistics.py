import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
generated_data_path = parent_folder_path+"/data_generated/"
bm25_folder = str(Path('.').absolute().parent.parent)+"/IR_BM25/"

sys.path+=[parent_folder_path, datasets_folder_path, generated_data_path, bm25_folder]

import numpy as np
import pickle
import json

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

def construct_dataset():
    train_path = parent_folder_path + "/data_raw/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl"
    dev_path = parent_folder_path + "/data_raw/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl"
    test_path = parent_folder_path + "/data_raw/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl"
    fact_path = parent_folder_path + "/data_raw/OpenBookQA-V1-Sep2018/Data/Main/openbook.txt"

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


def get_squad_statistics():
    with open(generated_data_path+"squad_useqa/squad_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]
    dev_list = squad_pickle["dev_list"]

    sent_list=  squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list =squad_pickle["resp_list"]

    print("squad n train:", len(train_list))

    token_len_list = []
    for question in train_list+squad_pickle["dev_list"]:
        token_len_list.append(len(question["question"].split(" ")))

    token_len_doc_list = []
    for resp in resp_list:
        doc = sent_list[int(resp[0])]+" "+doc_list[int(resp[1])]
        token_len_doc_list.append(len(doc.split(" ")))

    print("avg query len:", sum(token_len_list)/len(token_len_list), "avg doc len:", sum(token_len_doc_list)/len(token_len_doc_list))

    return 0

def get_squad_example():
    with open(generated_data_path+"squad_useqa/squad_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]
    dev_list = squad_pickle["dev_list"]

    sent_list = squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list = squad_pickle["resp_list"]

    for question in train_list:
        print("question:", question["question"])
        print("answer sent:", sent_list[int(resp_list[question["response"]][0])])
        print("answer doc:", doc_list[int(resp_list[question["response"]][1])])
        input("="*20)

def get_nq_example():
    with open(generated_data_path+"nq_retrieval_raw/nq_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]

    sent_list=  squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list =squad_pickle["resp_list"]

    for question in train_list:
        print("question:", question["question"])
        print("answer sent:", sent_list[int(resp_list[question["response"]][0])])
        print("answer doc:", doc_list[int(resp_list[question["response"]][1])])
        input("="*20)



def get_nq_statistics():
    with open(generated_data_path+"nq_retrieval_raw/nq_retrieval_data.pickle", "rb") as handle:
        openbook_pickle = pickle.load(handle)

    train_list = openbook_pickle["train_list"]

    sent_list=  openbook_pickle["sent_list"]
    doc_list = openbook_pickle["doc_list"]
    resp_list =openbook_pickle["resp_list"]

    print("squad n train:", len(train_list))

    token_len_list = []
    for question in train_list:
        token_len_list.append(len(question["question"].split(" ")))

    token_len_doc_list = []
    for resp in resp_list:
        doc = sent_list[int(resp[0])]+" "+doc_list[int(resp[1])]
        token_len_doc_list.append(len(doc.split(" ")))

    print("avg query len:", sum(token_len_list)/len(token_len_list), "avg doc len:", sum(token_len_doc_list)/len(token_len_doc_list))

    return 0

def get_openbook_statistics():

    train_list, dev_list, test_list,kb  = construct_dataset()


    print("squad n train:", len(train_list))

    token_len_list = []
    for question in train_list+dev_list+test_list:
        token_len_list.append(len(question["text"].split(" ")))


    token_len_doc_list = []
    for fact in kb:

        token_len_doc_list.append(len(fact[1:-1].split(" ")))

    print("avg query len:", sum(token_len_list)/len(token_len_list), "avg doc len:", sum(token_len_doc_list)/len(token_len_doc_list))

    return 0

# get_squad_statistics()
# get_nq_statistics()
get_openbook_statistics()


