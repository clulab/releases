import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
models_folder_path = parent_folder_path+"/models"
data_folder_path = parent_folder_path+"/data"
experiment_folder_path = parent_folder_path+"/experiments"
sys.path+=[parent_folder_path, data_folder_path, models_folder_path, experiment_folder_path]

import torch
import torch.nn as nn
import random
import torch.nn.functional as F

import pickle
import random
from torch import optim
import json

from experiment_t5_small import Seq2SeqT5Experiment

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

use_hint_in_inference = True if sys.argv[1]=="1" else False
epoch = sys.argv[2]
get_loss = int(sys.argv[3])

def load_questions_json(question_path: str):
    questions_list = list([])
    with open(question_path, 'r', encoding='utf-8') as dataset:
        for i, line in enumerate(dataset):
            item = json.loads(line.strip())
            questions_list.append(item)

    return questions_list


def get_train_test_data(input_with_hint = False):
    train_json_list = load_questions_json(data_folder_path+"/QASC_Dataset/train.jsonl")
    dev_json_list = load_questions_json(data_folder_path+"/QASC_Dataset/dev.jsonl")

    train_data = []
    test_data = []

    if input_with_hint:
        answer_key_map = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10}

        for item in train_json_list:
            question_text = item["question"]["stem"].lower().replace("?", " ").replace("."," ")  # Used for remove question marks
            answer_text = item["question"]["choices"][answer_key_map[item["answerKey"]]]["text"].lower().replace(".", " ")
            fact1_text = item["fact1"].lower().replace(".", " ")  # used for remove period
            fact2_text = item["fact2"].lower().replace(".", " ")  # used for remove period

            hint_tokens = " ".join(list(set(question_text.split(" ")+answer_text.split(" ")).intersection(set(fact1_text.split(" ")+fact2_text.split(" ")))))

            train_data.append({"input": "substitution statement 1: " + fact1_text + " . statement 2: " +
                                        fact2_text + " . hint: " + hint_tokens + " . </s>",
                               "output": item["combinedfact"].lower().replace(".", " ") + " . </s>"})


        for item in dev_json_list:
            question_text = item["question"]["stem"].lower().replace("?", " ").replace(".",
                                                                                       " ")  # Used for remove question marks
            answer_text = item["question"]["choices"][answer_key_map[item["answerKey"]]]["text"].lower().replace(".",
                                                                                                                 " ")
            fact1_text = item["fact1"].lower().replace(".", " ")  # used for remove period
            fact2_text = item["fact2"].lower().replace(".", " ")  # used for remove period

            hint_tokens = " ".join(list(set(question_text.split(" ") + answer_text.split(" ")).intersection(
                set(fact1_text.split(" ") + fact2_text.split(" ")))))

            test_data.append({"input": "substitution statement 1: " + fact1_text + " . statement 2: " +
                                        fact2_text + " . hint: " + hint_tokens + " . </s>",
                               "output": item["combinedfact"].lower().replace(".", " ") + " . </s>"})

    else:
        for item in train_json_list:
            fact1_text = item["fact1"].lower().replace(".", " ")  # used for remove period
            fact2_text = item["fact2"].lower().replace(".", " ")  # used for remove period

            train_data.append({"input": "substitution statement 1: "+fact1_text+" . statement 2: "+fact2_text + " . </s>", "output":item["combinedfact"].lower().replace(".", " ")+" . </s>"})

        for item in dev_json_list:
            fact1_text = item["fact1"].lower().replace(".", " ")  # used for remove period
            fact2_text = item["fact2"].lower().replace(".", " ")  # used for remove period

            test_data.append({"input": "substitution statement 1: "+fact1_text+" . statement 2: "+fact2_text+" . </s>", "output":item["combinedfact"].lower().replace(".", " ")+" . </s>"})

    return train_data, test_data



def main():
    print("use hint in inference:", use_hint_in_inference)
    train_pairs, test_pairs = get_train_test_data(use_hint_in_inference)

    print(random.choice(train_pairs))
    print(random.choice(test_pairs))

    t5_qasc_experiment = Seq2SeqT5Experiment(learning_rate=0.00001, device=device)  # tutorial uses 1e-4
    t5_qasc_experiment.t5_model = torch.load("saved_model/20200814_t5_small_QASC_hint_"+sys.argv[1]+"_epoch_"+epoch)

    if get_loss==1:
        t5_qasc_experiment.evaluate_iters_and_get_loss(test_pairs)

    else:
        print("=" * 20)
        #t5_qasc_experiment.evaluate_iters(train_pairs[:20])
        #print("-"*20)
        random.seed(0)
        eval_indices = random.sample(range(len(test_pairs)), 20)
        t5_qasc_experiment.evaluate_iters([test_pairs[idx] for idx in eval_indices])

    return 0


main()

