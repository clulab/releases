'''
This is to load the data in the standard manner:
Load the data for train/dev/test on depth 0,1,2,3,5
This is used for the training of the standard T5 system and the formal evaluation of the trained system.
'''
import sys
from pathlib import Path
import argparse

data_folder_path = str(Path('.').absolute().parent.parent)+"/Data/rule-reasoning-dataset-V2020.2.4"
parent_folder_path = str(Path('.').absolute().parent)

import json
import math


class RuleTakerParsedInstances():
    def __init__(self):
        self.fact_buffer_size = 5
        self.rule_buffer_size = 3

        self.instances = self._load_dataset()

    def _load_dataset(self):

        instances_all = {}
        for data_depth in [0,1,2,3,5]:
            instances_all["depth-"+str(data_depth)] = {}
            for split in ["train", "dev", "test"]:
                instances_all["depth-"+str(data_depth)][split] = []

                data_file_path = data_folder_path+"/depth-"+str(data_depth)+"/meta-"+split+".jsonl"

                with open(data_file_path, "r") as f:
                    raw_jsons = list(f)

                for raw_json in raw_jsons:
                    item = json.loads(raw_json)
                    question_tuples = list(item["questions"].items())

                    n_fact = str(item["NFact"])
                    n_rule = str(item["NRule"])
                    n_fact_buffer = math.ceil(int(n_fact)/self.fact_buffer_size)
                    n_rule_buffer = math.ceil(int(n_rule)/self.rule_buffer_size)

                    all_facts = ["fact "+str(idx+1)+": "+triple[1]["text"].lower() for idx, triple in enumerate(list(item["triples"].items()))]
                    all_rules = ["rule "+str(idx+1)+": "+rule[1]["text"].lower() for idx, rule in enumerate(list(item["rules"].items()))]

                    context = " ".join([triple[1]["text"].lower() for idx, triple in enumerate(list(item["triples"].items()))] +
                                [rule[1]["text"].lower() for idx, rule in enumerate(list(item["rules"].items()))])

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
                        question_text = question_tuple[1]["question"].lower()
                        answer_text = str(question_tuple[1]["answer"]).lower()
                        instance_dict = {"question":question_text, "answer":answer_text,
                                         "n_fact": n_fact, "n_rule": n_rule,
                                         "facts_text": all_facts, "rules_text":all_rules,
                                         "context": context}

                        # answer text in json item is already in

                        for buffer_tuple in fact_buffers:
                            instance_dict[buffer_tuple[0]] = buffer_tuple[1]
                        for buffer_tuple in rule_buffers:
                            instance_dict[buffer_tuple[0]] = buffer_tuple[1]

                        instances_all["depth-" + str(data_depth)][split].append(instance_dict)

        return instances_all

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


