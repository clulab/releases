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
import time
import numpy as np

from T5Vanilla import T5Vanilla

data_option = sys.argv[1]
data_depth = sys.argv[2]

if data_option == "0":
    dataset_name = "depth-5"
elif data_option =="1":
    dataset_name = "birds-electricity"
else:
    dataset_name = "NatLang"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("device: %s, dataset name: %s, inference depth: %s. " % (device, dataset_name, data_depth))

class Dataset():
    def __init__(self):
        self.fact_buffer_size = 5
        self.rule_buffer_size = 3

        self.instances = self._load_dataset()

    def _load_dataset(self):
        data_file_path = data_folder_path+"/"+dataset_name+"/meta-test.jsonl"

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

            for question_tuple in question_tuples:

                #print(int(question_tuple[1]["QDep"]))
                if int(question_tuple[1]["QDep"]) == int(data_depth):
                    proof_strategy = question_tuple[1]["strategy"]
                    proofs = self._get_proofs(question_tuple[1]["proofs"])

                    question_text = question_tuple[1]["question"].lower()
                    answer_text = question_tuple[1]["answer"]
                    instance_dict = {"question":question_text, "answer":answer_text,
                                     "n_fact": n_fact, "n_rule": n_rule,
                                     "facts_text": all_facts, "rules_text":all_rules,
                                     "strategy": proof_strategy, "proofs": proofs}

                    for buffer_tuple in fact_buffers:
                        instance_dict[buffer_tuple[0]] = buffer_tuple[1]
                    for buffer_tuple in rule_buffers:
                        instance_dict[buffer_tuple[0]] = buffer_tuple[1]

                    instances.append(instance_dict)

        return instances


    def _get_proofs(self, proofs_string):
        cleaned_proof = []

        for raw_proof in proofs_string[2:-2].split("OR"):
            while raw_proof[0]==" ":
                raw_proof = raw_proof[1:]
            while raw_proof[-1]==" ":
                raw_proof = raw_proof[:-1]

            cleaned_proof.append(raw_proof)

        return cleaned_proof


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
        # 1021 is the one before formal result production, and without random seed.
        #self.t5_c = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_c")
        # 0407 is for the camera ready paper, trained on du5
        self.t5_c = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_c")
        self.t5_c.to(device)

        #self.t5_f = torch.load("saved_models/20201021_t5_small_ruletaker_multitask_type_f")
        self.t5_f.to(device)

        #self.t5_r = torch.load("saved_models/20201111_t5_small_ruletaker_multitask_type_r")
        self.t5_r.to(device)

        # This learning rate 0.0001 is used in one of the tutorial, but might not be the best choice.
        self.device = device
        self.depth_limit = 25

        self.computation_count = 0
        self.computation_limit = 10000

        self.print_inference_steps = True

    def neural_backward_chaining(self, episodic_buffer, instance):

        depth_count = 0
        self.computation_count = 0
        start_time = time.time()
        final_answer, full_proof = self._one_step_inference(episodic_buffer, instance, depth_count+1)
        end_time = time.time()

        return final_answer, full_proof, end_time-start_time

    def _one_step_inference(self, episodic_buffer, instance, depth_count, label_return_option = "standard"):
        # We don't need computation limit anymore.

        #print("depth count:", depth_count)

        operation = self._t5_c_forward(" ".join(episodic_buffer)+" </s>")

        if self.print_inference_steps:
            print("\t"+"-"*20)
            print("\tepisodic buffer", episodic_buffer)
            print("\tgenerated operation:",operation)

        if "GENERATE_SUBGOALS" in operation:
            proof_to_return = ""

            if depth_count<self.depth_limit:
                subgoals_text = self._t5_c_forward(" ".join(episodic_buffer)+" operator: GENERATE_SUBGOALS </s>")

                if self.print_inference_steps:
                    print("\tgenerated subgoal:"+subgoals_text)

                if label_return_option!="flip":
                    for or_branch_idx, or_branch in enumerate(subgoals_text.split(" OR ")):
                        and_branch_results = []
                        and_branch_proofs = []
                        for and_branch in or_branch.split(" AND "):
                            episodic_buffer_ = [episodic_buffer[0], "episodic buffer: "+and_branch]
                            branch_result, proof_string = self._one_step_inference(episodic_buffer_, instance, depth_count+1)
                            and_branch_results.append(branch_result)
                            if "not" not in and_branch:
                                if branch_result==True:
                                    and_branch_proofs.append(proof_string)
                            else:
                                if branch_result==False:
                                    proof_to_return = proof_string
                                else:
                                    and_branch_proofs.append("NAF")

                            if branch_result == False:
                                 break

                        if False not in and_branch_results:
                            if "according to" in episodic_buffer[-1]:
                                episodic_buffer_of_interest = re.findall(r"according to.+", episodic_buffer[-1])[0]
                                rules_buffer_parsed = [re.findall(r"\d+", matched_rule_)[0]
                                                           for matched_rule_ in
                                                           episodic_buffer_of_interest.split(" or ")]
                                proof_to_return = "(("+ " ".join(and_branch_proofs) +") -> rule"+rules_buffer_parsed[or_branch_idx]+")"

                                return True, proof_to_return
                            else:
                                if "facts do not contradict" in subgoals_text and \
                                        "rules do not contradict" in subgoals_text:
                                    return True, "NAF"
                                else:
                                    return True, " ".join(and_branch_proofs)

                else:
                    # The outer loop handles different matched rules.
                    # The inner loop handles the preconditions of each matched rule.
                    # Logic: in each inner loop, not all preconditions should be true.
                    # Logic: in the outer loop, none of the rule should return true.
                    and_out_branch_results = []
                    for and_out_branch_idx, and_out_branch in enumerate(subgoals_text.split(" )AND( ")):

                        and_branch_proofs = []
                        and_branch_results = []
                        for and_branch in and_out_branch.split(" AND "):
                            episodic_buffer_ = [episodic_buffer[0], "episodic buffer: " + and_branch]
                            branch_result, proof_string = self._one_step_inference(episodic_buffer_, instance, depth_count + 1)
                            and_branch_results.append(branch_result)
                            if proof_string != "":
                                # If the result of the branch is true, then the returned proof should be either
                                # an actual proof of NAF, and should not be "". "" proof should only appear in false answer.
                                and_branch_proofs.append(proof_string)

                            if branch_result == False:
                                and_out_branch_results.append(False)
                                break

                        if False not in and_branch_results:
                            if "according to" in episodic_buffer[-1]:
                                episodic_buffer_of_interest = re.findall(r"according to.+", episodic_buffer[-1])[0]
                                rules_buffer_parsed = [re.findall(r"\d+", matched_rule_)[0]
                                                       for matched_rule_ in
                                                       episodic_buffer_of_interest.split(" )and( ")]
                                proof_to_return = "(("+ " ".join(and_branch_proofs) +") -> rule"+rules_buffer_parsed[and_out_branch_idx]+")"

                                return False, proof_to_return
                            else:
                                # This condition should never be reached.
                                return False, " ".join(and_branch_proofs) if len(and_branch_proofs)>0 else ""

                    return True, "NAF"

            # This is to handle the situation of hitting the depth limit.
            if label_return_option=="standard":
                if "not" not in episodic_buffer[-1]:
                    return False, ""
                else:
                    return False, proof_to_return
            else:
                return True, ""

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

                    if "FACT" in buffer_key:
                        answer = self._t5_f_forward(" ".join(episodic_buffer_+["operator: RUN "])+" </s>")
                    else:
                        answer = self._t5_r_forward(" ".join(episodic_buffer_ + ["operator: RUN "]) + " </s>")

                    if self.print_inference_steps:
                        print("\tbuffer input:"+" ".join(episodic_buffer_+["operator: RUN "])+" </s>")
                        print("\tgenerated answer:"+answer)

                    if "true" in answer:
                        if bool(re.findall(r"confirmed", answer)):
                            text_to_return = "triple"+re.findall(r"\d+", answer)[0]
                            return True, text_to_return
                        else:
                            return True, ""
                    elif "false" in answer:
                        if bool(re.findall(r"contradicted", answer)):
                            text_to_return = "triple" + re.findall(r"\d+", answer)[0]
                            return False, text_to_return
                        else:
                            return False, ""
                    else:
                        if depth_count<self.depth_limit:
                            if "does not contradict" in episodic_buffer_[1]:
                                final_answer, proof_string = self._one_step_inference([episodic_buffer[0], "episodic buffer: "+answer], instance, depth_count+1, label_return_option="flip")
                            else:
                                final_answer, proof_string = self._one_step_inference([episodic_buffer[0], "episodic buffer: "+answer], instance, depth_count+1)
                            return final_answer, proof_string
                        else:
                            if "does not contradict" in episodic_buffer_[1]:
                                return True, ""
                            else:
                                return False, ""
                else:
                    return False, ""
            except:
                return False, ""

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

            predicted_tensor = self.t5_r.generate(input_tensor, max_length=300)
            predicted_text = self.tokenizer.decode(predicted_tensor[0])

        return predicted_text

    def _parse_output(self):

        return 0

    def _parse_operators(self):

        return 0

def manual_debugging():
    chaining_dataset = Dataset()
    print("depth:", data_depth, " n samples:",len(chaining_dataset.instances))

    neural_chaining_solver = NeuralBackwardChainer(device)

    all_statistics = []
    correct_count_list = []
    time_list = []
    while 1:
        print("="*20)
        instance_idx_to_check = int(input("input the index of the instance to check:"))
        instance = chaining_dataset.instances[instance_idx_to_check]
        Dataset.print_problem(instance)

        episodic_buffer  = ["episodic buffer: there are "+str(math.ceil(int(instance['n_fact'])/chaining_dataset.fact_buffer_size))+
                            " fact buffers and "+ str(math.ceil(int(instance['n_rule'])/chaining_dataset.rule_buffer_size))+ " rule buffers.",
                            "episodic buffer: i want to prove \""+ instance["question"][:-1]+"\"."]
        pred, full_proof, used_time = neural_chaining_solver.neural_backward_chaining(episodic_buffer, instance)

        print("pred:", pred, " label:", instance["answer"])



def main():
    chaining_dataset = Dataset()
    print("depth:", data_depth, " n samples:",len(chaining_dataset.instances))

    neural_chaining_solver = NeuralBackwardChainer(device)

    all_statistics = []
    correct_count_list = []
    time_list = []

    instance_count = 0
    correct_count = 0
    for i, instance in enumerate(chaining_dataset.instances):

        #if instance["strategy"] == "proof" or instance["strategy"] == "inv-proof":
        if i==0:
            print("=" * 20)
            print("procssing instance ", i)
            Dataset.print_problem(instance)

            episodic_buffer  = ["episodic buffer: there are "+str(math.ceil(int(instance['n_fact'])/chaining_dataset.fact_buffer_size))+
                                " fact buffers and "+ str(math.ceil(int(instance['n_rule'])/chaining_dataset.rule_buffer_size))+ " rule buffers.",
                                "episodic buffer: i want to prove \""+ instance["question"][:-1]+"\"."]
            pred, full_proof, used_time = neural_chaining_solver.neural_backward_chaining(episodic_buffer, instance)

            print("-"*20)
            print("instance ",i, " label:", instance["answer"], " pred:", pred)
            print("full proof:", full_proof)
            print("all candidate proofs:", instance["proofs"])
            print("proof good?", full_proof in instance["proofs"])

            instance_count+=1
            if full_proof in instance["proofs"]:
                correct_count+=1
            print("correct proof/all:", correct_count,"/", instance_count)

            print("pred:", pred, " bool pred:", bool(pred), " answer:",
                  instance["answer"], " bool answer:", bool(instance["answer"]),
                  " equivalence:", bool(pred) == bool(instance['answer']))

            input("-"*20)


        # print("-"*20)
        # print("final prediction:"+str(pred)+"  true answer:"+str(instance["answer"]))
        # input("---------")

        # all_statistics.append([i, 1 if pred else 0, 1 if bool(instance["answer"]) else 0, used_time]) # question idx, pred answer, true answer, time
        # correct_count_list.append(1 if pred==bool(instance["answer"]) else 0)
        # time_list.append(used_time)

        # print("pred:", pred, " answer:", instance["answer"])
        # print("full proof:", full_proof)
        # print("proof strategy:", instance["strategy"])
        # if instance["strategy"]=="proof" or instance["strategy"]=="inv-proof":
        #     print("all candidate proofs:", instance["proofs"])
        #     print("proof good?", full_proof in instance["proofs"])
        # input("="*30)
        #print(pred, instance["answer"], all_statistics[-1])

        # if (i+1)%10==0:
        #     print("evaluating instance ",i,
        #           " average acc:", sum(correct_count_list)/len(correct_count_list),
        #           " average time:", sum(time_list)/len(time_list))

        # if i>2:
        #     break

    #np.savetxt("saved_models/20201109_t5_small_ruletaker_chaining_results_depth_"+data_depth+".csv", np.array(all_statistics), delimiter=",")

#manual_debugging()
main()
