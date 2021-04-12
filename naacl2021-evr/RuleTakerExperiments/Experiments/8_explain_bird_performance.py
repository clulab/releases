import sys
from pathlib import Path
import argparse

data_folder_path = str(Path('.').absolute().parent.parent)+"/Data/rule-reasoning-dataset-V2020.2.4"
parent_folder_path = str(Path('.').absolute().parent)
experiment_folder_path = parent_folder_path+"/ExperimentClass"
data_processing_folder_path = parent_folder_path+"/DataProcessing"
sys.path+=[data_folder_path, parent_folder_path, experiment_folder_path, data_processing_folder_path]

import math
import json
import pickle
import numpy as np

DATA_DEPTH = 2

class Dataset():
    def __init__(self, fact_buffer_size, rule_buffer_size):
        self.fact_buffer_size = int(fact_buffer_size)
        self.rule_buffer_size = int(rule_buffer_size)

        self.instances = self._load_dataset()

    def _load_dataset(self):

        dataset_name = "birds-electricity"

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
                if int(question_tuple[1]["QDep"]) == int(DATA_DEPTH):
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

def load_results():
    file_name = "saved_models/exp_eval_results/exp_6_depth_" + str(DATA_DEPTH) + ".pickle"
    with open(file_name, "rb") as handle:
        result_dict = pickle.load(handle)

    return result_dict

def check_output():
    be_dataset = Dataset(5,3)
    result_dict = load_results()

    for i, instance_result_dict in enumerate(result_dict["instance_output"]):
        if bool(instance_result_dict["proof_flag"])==False:

            Dataset.print_problem(be_dataset.instances[i])
            print(be_dataset.instances[i]["proofs"])
            print("-"*20)
            print(instance_result_dict)

            input("+"*20)



check_output()
