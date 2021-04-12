import sys
from pathlib import Path
import argparse

data_folder_path = str(Path('.').absolute().parent.parent)+"/Data/rule-reasoning-dataset-V2020.2.4"

parent_folder_path = str(Path('.').absolute().parent)
data_processing_folder_path = parent_folder_path+"/DataProcessing"
experiment_class_path = parent_folder_path+"/ExperimentClass"
sys.path+=[parent_folder_path, data_processing_folder_path, experiment_class_path]

import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import optim
import torch
import os
import json

import csv
import editdistance

from LoadData import loadAsSingleTasks
from T5Vanilla import T5Vanilla


data_depth = "none" #sys.argv[1]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("device: %s, inference depth: %s. " % (device, data_depth))

class T5ManualCheck():
    def __init__(self, device):
        t5Exp = T5Vanilla(0.0001, device)

        self.tokenizer = t5Exp.tokenizer
        self.t5_model = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_f")
        self.t5_model.to(device)

        # This learning rate 0.0001 is used in one of the tutorial, but might not be the best choice.
        self.device = device
        self.depth_limit = 5

    def check_by_manual_input(self, input_string, target_string):
        self.t5_model.eval()
        with torch.no_grad():
            input_tensor = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

            predicted_tensor = self.t5_model.generate(input_tensor, max_length=200)
            predicted_text = self.tokenizer.decode(predicted_tensor[0])

        print("="*20)
        print("input string:", input_string)
        print("output string:", predicted_text)
        print("target string:", target_string)
        print("edit distance:", editdistance.eval(predicted_text, target_string))

        input("-"*20)

    def check_edit_distance_by_manual_input(self, input_string, target_string):
        self.t5_model.eval()
        with torch.no_grad():
            input_tensor = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

            predicted_tensor = self.t5_model.generate(input_tensor, max_length=200)
            predicted_text = self.tokenizer.decode(predicted_tensor[0])

        print("="*20)
        print("input string:", input_string)
        print("output string:", predicted_text)
        print("target string:", target_string)
        print("output list:", list(predicted_text))
        print("target list:", list(target_string))
        print("edit distance:", editdistance.eval(predicted_text, target_string))

        input("-"*20)

def check_error():

    t5checker = T5ManualCheck(device)

    test_strings = [
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"something is big\". </s>",
            "target_string": "GENERATE_SUBGOALS"
        },
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"someone is big\". </s>",
            "target_string": "GENERATE_SUBGOALS"
        },
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"something is not big\". </s>",
            "target_string": "GENERATE_SUBGOALS"
        },
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"someone is not big\". </s>",
            "target_string": "GENERATE_SUBGOALS"
        },

        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"something is big\". operator: GENERATE_SUBGOALS </s>",
            "target_string": "i want to judge whether the facts can prove \"something is big\". OR i want to judge whether the rules can prove \"something is big\"."
        },
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"someone is big\". operator: GENERATE_SUBGOALS </s>",
            "target_string": "i want to judge whether the facts can prove \"someone is big\". OR i want to judge whether the rules can prove \"someone is big\"."
        },
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"something is not big\". operator: GENERATE_SUBGOALS </s>",
            "target_string": "i want to judge whether the facts do not contradict \"something is not big\". OR i want to judge whether the rules do not contradict \"something is not big\"."
        },
        {
            "input_string": "episodic buffer: there are 2 fact buffers and 1 rule buffers. episodic buffer: i want to prove \"someone is not big\". operator: GENERATE_SUBGOALS </s>",
            "target_string": "i want to judge whether the facts do not contradict \"someone is not big\". OR i want to judge whether the rules do not contradict \"someone is not big\"."
        },
    ]

    for instance in test_strings:
        t5checker.check_by_manual_input(instance["input_string"], instance["target_string"])

    return 0

def check_error_1():
    t5checker = T5ManualCheck(device)

    input_string = "buffer input:episodic buffer: there are 3 fact buffers and 3 rule buffers. episodic buffer: i want to judge whether fact buffer 1 can prove \"the mouse sees the rabbit\". fact 1: the cat chases the rabbit. fact 2: the cat is red. fact 3: the cat sees the rabbit. fact 4: the cat visits the mouse. fact 5: the lion is green. operator: RUN  </s>"
    target_string = "False"

    t5checker.check_by_manual_input(input_string, target_string)

    input_string = "buffer input:episodic buffer: there are 3 fact buffers and 3 rule buffers. episodic buffer: i want to judge whether fact buffer 1 can prove \"the lion sees the rabbit\". fact 1: the cat chases the rabbit. fact 2: the cat is red. fact 3: the cat sees the rabbit. fact 4: the cat visits the mouse. fact 5: the lion is green. operator: RUN  </s>"
    target_string = "False"

    t5checker.check_by_manual_input(input_string, target_string)

    input_string = "buffer input:episodic buffer: there are 3 fact buffers and 3 rule buffers. episodic buffer: i want to judge whether fact buffer 1 can prove \"the mouse sees the lion\". fact 1: the cat chases the rabbit. fact 2: the cat is red. fact 3: the cat sees the rabbit. fact 4: the cat visits the mouse. fact 5: the lion is green. operator: RUN  </s>"
    target_string = "False"

    t5checker.check_by_manual_input(input_string, target_string)

def check_error_2():
    t5checker = T5ManualCheck(device)

    input_string = "buffer input:episodic buffer: there are 2 fact buffers and 3 rule buffers. episodic buffer: i want to judge whether fact buffer 1 can prove \"the mouse sees the mouse\". fact 1: the bear sees the mouse. fact 2: the cow visits the dog. fact 3: the dog visits the cow. fact 4: the mouse chases the bear. fact 5: the mouse chases the dog. operator: RUN  </s>"
    target_string = "False"

    t5checker.check_by_manual_input(input_string, target_string)

def check_edit_distance_pattern_1():

    t5checker = T5ManualCheck(device)

    instances = loadAsSingleTasks()

    for i in range(100):
        input_string = instances["train"]['pattern1'][i]["input"]
        target_string = instances["train"]['pattern1'][i]['output']

        t5checker.check_edit_distance_by_manual_input(input_string, target_string[:-5])

def check_edit_distance_pattern_3_4():

    t5checker = T5ManualCheck(device)

    instances = loadAsSingleTasks()

    for i in range(100):
        input_string = instances["train"]['pattern3'][i]["input"]
        target_string = instances["train"]['pattern3'][i]['output']

        t5checker.check_edit_distance_by_manual_input(input_string, target_string[:-5])

        input_string = instances["train"]['pattern4'][i]["input"]
        target_string = instances["train"]['pattern4'][i]['output']

        t5checker.check_edit_distance_by_manual_input(input_string, target_string[:-5])

#check_edit_distance_pattern_3_4()
#check_error_3()
#check_error()

check_error_1()
check_error_2()