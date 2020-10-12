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
import os
import numpy as np

from experiment_t5_small import Seq2SeqT5Experiment

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

use_hint_in_inference = True if sys.argv[1]=="1" else False


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

    # tutorial uses 1e-4
    # https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
    t5_qasc_experiment = Seq2SeqT5Experiment(learning_rate=0.0001, device=device)

    minimal_avg_loss = 100
    avg_loss_list = []
    save_folder_path = "saved_model"

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)


    for i in range(10):
        print("=" * 20)
        print("epoch:",i)
        t5_qasc_experiment.train_iters(train_pairs, 8000, print_every=500)
        print("-"*20)
        avg_loss = t5_qasc_experiment.evaluate_iters_and_get_loss(test_pairs)
        avg_loss_list.append(avg_loss)

        if avg_loss<minimal_avg_loss:

            minimal_avg_loss = avg_loss
            t5_qasc_experiment.save_tuned_model(save_folder_path+"/20200814_t5_small_QASC_hint_" + sys.argv[1])
            t5_qasc_experiment.evaluate_iters_and_save_output(test_pairs, avg_loss, save_folder_path+"/20200814_t5_small_QASC_hint_" + sys.argv[1] + ".tsv")

        else:
            break

    avg_loss_list = np.array(avg_loss_list)
    np.savetxt(save_folder_path+"/20200814_t5_small_QASC_hint_" + sys.argv[1] +"_loss.csv", avg_loss_list, delimiter=",")

    return 0


main()

