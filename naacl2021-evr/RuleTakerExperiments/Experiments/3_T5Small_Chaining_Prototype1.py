import sys
from pathlib import Path
import argparse

data_folder_path = str(Path('.').absolute().parent.parent)+"/Data/rule-reasoning-dataset-V2020.2.4"
parent_folder_path = str(Path('.').absolute().parent)
experiment_folder_path = parent_folder_path+"/ExperimentClass"
data_processing_folder_path = parent_folder_path+"/DataProcessing"
sys.path+=[data_folder_path, parent_folder_path, experiment_folder_path, data_processing_folder_path]

import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import optim
import torch
import os
import json
import math

import csv
import editdistance
import re

from T5Vanilla import T5Vanilla

data_depth = sys.argv[1]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("device: %s, inference depth: %s. " % (device, data_depth))

class Dataset():
    def __init__(self):
        self.fact_buffer_size = 5
        self.rule_buffer_size = 3

        self.instances = self._load_dataset()

    def _load_dataset(self):
        data_file_path = data_folder_path+"/depth-5/meta-test.jsonl"

        with open(data_file_path, "r") as f:
            raw_jsons = list(f)

        instances = []
        for raw_json in raw_jsons:
            item = json.loads(raw_json)
            question_tuples = list(item["questions"].items())

            n_fact = str(item["NFact"])
            n_rule = str(item["NRule"])
            n_fact_buffer = math.ceil(int(n_fact)/self.fact_buffer_size)
            n_rule_buffer = math.ceil(int(n_rule)/self.rule_buffer_size)

            all_facts = ["fact "+str(idx+1)+": "+triple[1]["text"].lower() for idx, triple in enumerate(list(item["triples"].items()))]
            all_rules = ["rule "+str(idx+1)+": "+rule[1]["text"].lower() for idx, rule in enumerate(list(item["rules"].items()))]

            fact_buffers = []
            for fact_buffer_index in range(n_fact_buffer):
                fact_buffer_key = "FACT_BUFFER_"+str(fact_buffer_index+1)
                fact_buffer_values = " ".join(all_facts[fact_buffer_index*self.fact_buffer_size:min(len(all_facts), (fact_buffer_index+1)*self.fact_buffer_size)])
                fact_buffers.append((fact_buffer_key, fact_buffer_values))

            rule_buffers = []
            for rule_buffer_index in range(n_rule_buffer):
                rule_buffer_key = "RULE_BUFFER_"+str(rule_buffer_index+1)
                rule_buffer_values = " ".join(all_rules[rule_buffer_index*self.rule_buffer_size:min(len(all_rules), (rule_buffer_index+1)*self.rule_buffer_size)])
                rule_buffers.append((rule_buffer_key, rule_buffer_values))

            for question_tuple in question_tuples: # TODO: this is for debugging purpose. Only look at the first four problems.

                #print(int(question_tuple[1]["QDep"]))
                if int(question_tuple[1]["QDep"]) == int(data_depth):

                    question_text = question_tuple[1]["question"].lower()
                    answer_text = question_tuple[1]["answer"]
                    instance_dict = {"question":question_text, "answer":answer_text,
                                     "n_fact": n_fact, "n_rule": n_rule,
                                     "facts_text": all_facts, "rules_text":all_rules}

                    for buffer_tuple in fact_buffers:
                        instance_dict[buffer_tuple[0]] = buffer_tuple[1]
                    for buffer_tuple in rule_buffers:
                        instance_dict[buffer_tuple[0]] = buffer_tuple[1]

                    instances.append(instance_dict)

        return instances

    @staticmethod
    def print_problem(instance):
        print("facts:")
        for fact in instance["facts_text"]:
            print("\t"+fact)
        print("rules:")
        for rule in instance["rules_text"]:
            print("\t"+rule)
        print("question and answer:")
        print("\t"+instance["question"]+"    "+str(instance["answer"]))

class NeuralBackwardChainer:

    def __init__(self, device):
        t5Exp = T5Vanilla(0.0001, device)

        self.tokenizer = t5Exp.tokenizer
        # 1021 is the one before formal production, and without random seed.
        self.t5_c = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_c")
        self.t5_c.to(device)

        self.t5_f = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_f")
        self.t5_f.to(device)

        self.t5_r = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_r")
        self.t5_r.to(device)

        # This learning rate 0.0001 is used in one of the tutorial, but might not be the best choice.
        self.device = device
        self.depth_limit = 20

        self.computation_count = 0
        self.computation_limit = 100

    def neural_backward_chaining(self, episodic_buffer, instance):

        depth_count = 0
        self.computation_count = 0
        final_answer = self._one_step_inference(episodic_buffer, instance, depth_count+1)

        return final_answer

    def _one_step_inference(self, episodic_buffer, instance, depth_count, label_return_option = "standard"):
        self.computation_count+=1
        if self.computation_count>self.computation_limit:
            return False

        operation = self._t5_c_forward(" ".join(episodic_buffer)+" </s>")

        print("\t"+"-"*20)
        print("\tepisodic buffer", episodic_buffer)
        print("\tgenerated operation:",operation)
        if "GENERATE_SUBGOALS" in operation:
            if depth_count<self.depth_limit:
                subgoals_text = self._t5_c_forward(" ".join(episodic_buffer)+" operator: GENERATE_SUBGOALS </s>")
                print("\tgenerated subgoal:"+subgoals_text)
                # TODO: parse the subgoals and discriminate the OR branch/AND branch.
                if label_return_option!="flip":
                    for or_branch in subgoals_text.split(" OR "):

                        and_branch_results = []
                        for and_branch in or_branch.split(" AND "):
                            episodic_buffer_ = [episodic_buffer[0], "episodic buffer: "+and_branch]
                            branch_result = self._one_step_inference(episodic_buffer_, instance, depth_count+1)
                            and_branch_results.append(branch_result)

                            if branch_result == False:
                                 break

                        if False not in and_branch_results:
                            return True
                else:
                    # The outer loop handles different matched rules.
                    # The inner loop handles the preconditions of each matched rule.
                    # Logic: in each inner loop, not all preconditions should be true.
                    # Logic: in the outer loop, none of the rule should return true.
                    and_out_branch_results = []
                    for and_out_branch in subgoals_text.split(" )AND( "):

                        and_branch_results = []
                        for and_branch in and_out_branch.split(" AND "):
                            episodic_buffer_ = [episodic_buffer[0], "episodic buffer: " + and_branch]
                            branch_result = self._one_step_inference(episodic_buffer_, instance, depth_count + 1)
                            and_branch_results.append(branch_result)

                            if branch_result == False:
                                and_out_branch_results.append(False)
                                break

                        if False not in and_branch_results:
                            return False

                    return True

            return False

        else:
            try: # This is to handle the situation where the operations are not successfully generated.
                operators = operation.split(" THEN ")
                episodic_buffer_ = episodic_buffer
                buffer_key = operators[0].split("GET")[-1][2:-2]
                if "GET" in operators[0] and "FACT_BUFFER" in operators[0]:
                    episodic_buffer_ = episodic_buffer +[instance[buffer_key]]
                if "GET" in operators[0] and "RULE_BUFFER" in operators[0]:
                    episodic_buffer_ = episodic_buffer+ [instance[buffer_key]]
                if "RUN" in operators[1]:
                    print("\tbuffer input:"+" ".join(episodic_buffer_+["operator: RUN "])+" </s>")
                    if "FACT" in buffer_key:
                        answer = self._t5_f_forward(" ".join(episodic_buffer_+["operator: RUN "])+" </s>")
                    else:
                        answer = self._t5_r_forward(" ".join(episodic_buffer_ + ["operator: RUN "]) + " </s>")

                    print("\tgenerated answer:"+answer)

                    if "true" in answer:
                        return True
                    elif "false" in answer:
                        return False
                    else:
                        if depth_count<self.depth_limit:
                            if "does not contradict" in episodic_buffer_[1]:
                                final_answer = self._one_step_inference([episodic_buffer[0], "episodic buffer: "+answer], instance, depth_count+1, label_return_option="flip")
                            else:
                                final_answer = self._one_step_inference([episodic_buffer[0], "episodic buffer: "+answer], instance, depth_count+1)
                            return final_answer
                        else:
                            if "does not contradict" in episodic_buffer_[1]:
                                return True
                            else:
                                return False
                else:
                    return False
            except:
                return False

    def _t5_c_forward(self, input_string):
        self.t5_c.eval()
        with torch.no_grad():
            input_tensor = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

            predicted_tensor = self.t5_c.generate(input_tensor, max_length=200)
            predicted_text = self.tokenizer.decode(predicted_tensor[0])

        return predicted_text

    def _t5_f_forward(self, input_string):
        self.t5_f.eval()
        with torch.no_grad():
            input_tensor = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

            predicted_tensor = self.t5_f.generate(input_tensor, max_length=200)
            predicted_text = self.tokenizer.decode(predicted_tensor[0])

        return predicted_text

    def _t5_r_forward(self, input_string):
        self.t5_r.eval()
        with torch.no_grad():
            input_tensor = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

            predicted_tensor = self.t5_r.generate(input_tensor, max_length=200)
            predicted_text = self.tokenizer.decode(predicted_tensor[0])

        return predicted_text

    def _parse_output(self):

        return 0

    def _parse_operators(self):

        return 0


def main():
    chaining_dataset = Dataset()
    print("depth:", data_depth, " n samples:",len(chaining_dataset.instances))

    neural_chaining_solver = NeuralBackwardChainer(device)

    for instance in chaining_dataset.instances:

        print("="*20)
        Dataset.print_problem(instance)

        episodic_buffer  = ["episodic buffer: there are "+str(math.ceil(int(instance['n_fact'])/chaining_dataset.fact_buffer_size))+
                            " fact buffers and "+ str(math.ceil(int(instance['n_rule'])/chaining_dataset.rule_buffer_size))+ " rule buffers.",
                            "episodic buffer: i want to prove \""+ instance["question"][:-1]+"\"."]
        pred = neural_chaining_solver.neural_backward_chaining(episodic_buffer, instance)
        print("-"*20)
        print("final prediction:"+str(pred)+"  true answer:"+str(instance["answer"]))

        input("---------")


main()
