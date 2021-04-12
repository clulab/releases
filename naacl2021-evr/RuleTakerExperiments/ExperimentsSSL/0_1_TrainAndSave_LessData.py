import sys
from pathlib import Path
import argparse


'''
This script is to test the ability of the model to learn from superisingly small amount of data. 
'''

parent_folder_path = str(Path('.').absolute().parent)
experiment_folder_path = parent_folder_path+"/ExperimentClass"
data_processing_folder_path = parent_folder_path+"/DataProcessing"
sys.path+=[parent_folder_path, experiment_folder_path, data_processing_folder_path]

from T5Vanilla import T5Vanilla
from LoadData import loadAsSingleTasks
from torch.utils.data import Dataset, DataLoader
from PrepareBatch import RuleTakerDataset, PadCollate
from itertools import permutations


import torch
import os
import numpy as np
import random
import time
import pickle
import copy
import re

task_setting = sys.argv[1]  # 3nn / 1nn
train_module = sys.argv[2]  # c/f/r / s
train_amount = sys.argv[3]  # 10k / 30k / 70k
FACT_BUFFER_SIZE = sys.argv[4]  # 5 / 20
RULE_BUFFER_SIZE = sys.argv[5]  # 3 / 10
RANDOM_SEED = 0

N_EPOCH = 20 # 40
N_TRAIN_EACH_EPOCH = 1000 if train_module=="c" or train_module == "s" else 500  #24000
N_EVAL_EACH_EPOCH = 2000  #2000

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print('='*40)
print('='*40)

print("device:",device)
print("task setting:", task_setting," train module:", train_module, " train amount:", train_amount)
print("fact bf size:", FACT_BUFFER_SIZE, " rule bf size:", RULE_BUFFER_SIZE, " seed:", RANDOM_SEED)
print("n epoch:", N_EPOCH, " N Sample each train:", N_TRAIN_EACH_EPOCH, " N sample each dev:", N_EVAL_EACH_EPOCH)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def augment_pattern_5_9(instances):
    new_instances = copy.deepcopy(instances)
    return_instances = []
    for instance in new_instances:
        if "fact buffer 1" in instance["input"]:
            if random.random()>0.5:
                instance["input"] = instance["input"].replace("fact buffer 1", "fact buffer 2")
                instance["output"] = instance["output"].replace("FACT_BUFFER_1", "FACT_BUFFER_2")
            else:
                instance["input"] = instance["input"].replace("fact buffer 1", "fact buffer 3")
                instance["output"] = instance["output"].replace("FACT_BUFFER_1", "FACT_BUFFER_3")
            return_instances.append(instance)

        if "rule buffer 1" in instance["input"]:
            if random.random()>0.5:
                instance["input"] = instance["input"].replace("rule buffer 1", "rule buffer 2")
                instance["output"] = instance["output"].replace("RULE_BUFFER_1", "RULE_BUFFER_2")
            else:
                instance["input"] = instance["input"].replace("rule buffer 1", "rule buffer 3")
                instance["output"] = instance["output"].replace("RULE_BUFFER_1", "RULE_BUFFER_3")
            return_instances.append(instance)

    return return_instances

def augment_pattern_6(instances, debug_flag = False):
    fact_buffer_extract_pattern = r"(fact \d+:.*\.)+"
    instances_augmented = []

    for instance in instances:
        if debug_flag:
            print("=" * 20)
            print("original input:", instance["input"])
            print("original output:", instance["output"])

        instance_new = copy.deepcopy(instance)

        fact_buffer_span = re.search(fact_buffer_extract_pattern, instance["input"]).span()
        fact_buffer_text_old = instance["input"][fact_buffer_span[0]: fact_buffer_span[1]]

        # augment strategy 1: just shuffle the facts, but the answer dose not have change.
        fact_buffer_text_list = fact_buffer_text_old[:-1].split(". ")
        random.shuffle(fact_buffer_text_list)
        fact_buffer_text_new = ". ".join(fact_buffer_text_list) + "."
        instance_new["input"] = instance["input"][:fact_buffer_span[0]] + fact_buffer_text_new + instance["input"][fact_buffer_span[1]:]
        instances_augmented.append(instance_new)

        if debug_flag:
            print("-" * 20)
            print("aug 1 input:", instance_new["input"])
            print("aug 1 output:", instance_new["output"])

        # augment strategy 2: shuffle the facts, and also change the number in the answer.
        if "confirmed by" in instance["output"] or "contradicted by" in instance["output"]:
            fact_buffer_text_list = fact_buffer_text_old[:-1].split(". ")
            fact_num_spans = [re.search(r"fact \d+: ", fact_text).span() for fact_text in fact_buffer_text_list]
            fact_nums = [re.search(r"\d+", fact_text).group() for fact_text in fact_buffer_text_list]
            fact_without_nums = [fact_text[fact_num_spans[i][1]:] for i, fact_text in enumerate(fact_buffer_text_list)]

            answer_fact_num_str = re.search(r"\d+", instance["output"]).group()
            answer_idx_in_buffer = fact_nums.index(answer_fact_num_str)

            facts_in_buffer_indices = list(range(len(fact_without_nums)))

            perm = permutations(facts_in_buffer_indices)
            for facts_in_buffer_indices_ in perm:
                fact_initial_num = random.randint(1,10)

                fact_without_nums_shuffled = ["fact "+str(fact_initial_num+i)+": "+fact_without_nums[idx] for i, idx in enumerate(facts_in_buffer_indices_)]

                gold_answer_idx_new = fact_initial_num + facts_in_buffer_indices_.index(answer_idx_in_buffer)
                fact_buffer_text_new = ". ".join(fact_without_nums_shuffled)+"."
                if "this is confirmed by" in instance["output"]:
                    target_new = "true, this is confirmed by fact "+str(gold_answer_idx_new)+". </s>"
                else:
                    target_new = "false, this is contradicted by fact " + str(gold_answer_idx_new) + ". </s>"

                instance_new["input"] = instance["input"][:fact_buffer_span[0]] + \
                                        fact_buffer_text_new + instance["input"][fact_buffer_span[1]:]
                instance_new["output"] = target_new

                if random.random()<0.2:
                    instances_augmented.append(instance_new)

                if debug_flag:
                    print("-" * 20)
                    print("aug 2 input:", instance_new["input"])
                    print("aug 2 output:", instance_new["output"])

    if debug_flag:
        input("press enter to continue ... ")

    return instances_augmented

def augment_pattern_10(instances, debug_flag = False):
    rule_buffer_extract_pattern = r"(rule \d+:.*\.)+"
    instances_augmented = []

    for instance in instances:
        if debug_flag:
            print("=" * 20)
            print("original input:", instance["input"])
            print("original output:", instance["output"])

        instance_new = copy.deepcopy(instance)

        rule_buffer_span = re.search(rule_buffer_extract_pattern, instance["input"]).span()
        rule_buffer_text_old = instance["input"][rule_buffer_span[0]: rule_buffer_span[1]]

        # augment strategy 1: just shuffle the rules, but the answer dose not have change.
        if "true" in instance["output"] or "false" in instance["output"]:
            rule_buffer_text_list = rule_buffer_text_old[:-1].split(". ")
            if len(rule_buffer_text_list)>1:
                random.shuffle(rule_buffer_text_list)
                rule_buffer_text_new = ". ".join(rule_buffer_text_list) + "."
                instance_new["input"] = instance["input"][:rule_buffer_span[0]] + rule_buffer_text_new + instance["input"][
                                                                                                         rule_buffer_span[1]:]
                instances_augmented.append(instance_new)

                if debug_flag:
                    print("-" * 20)
                    print("aug 1 input:", instance_new["input"])
                    print("aug 1 output:", instance_new["output"])

        # augment strategy 2: shuffle the rules, and also change the number in the answer.
        if "according to" in instance["output"]:
            rule_buffer_text_list = rule_buffer_text_old[:-1].split(". ")
            rule_num_spans = [re.search(r"rule \d+: ", rule_text).span() for rule_text in rule_buffer_text_list]
            rule_nums = [re.search(r"\d+", rule_text).group() for rule_text in rule_buffer_text_list]
            rule_without_nums = [rule_text[rule_num_spans[i][1]:] for i, rule_text in enumerate(rule_buffer_text_list)]

            # handle the gold answer
            answer_rules_num_str = re.findall(r"\d+", instance["output"])
            answer_rules_text = [re.search(r"according to rule \d+, i need to prove (.*)", rule_text_raw).group(1)
                                 for rule_text_raw in instance["output"][:-6].split(" or ")] \
                                if " or " in instance["output"] \
                                else \
                                    [re.search(r"according to rule \d+, i need to prove (.*)", rule_text_raw).group(1)
                                    for rule_text_raw in instance["output"][:-6].split(" )and( ")]
            answer_indices_in_buffer = [rule_nums.index(answer_rule_num_str) for answer_rule_num_str in answer_rules_num_str]

            rules_in_buffer_indices = list(range(len(rule_without_nums)))
            perm = permutations(rules_in_buffer_indices)
            for rules_in_buffer_indices_ in perm:
                rule_initial_num = random.randint(1, 10)

                rule_without_nums_shuffled = ["rule " + str(rule_initial_num + i) + ": " + rule_without_nums[idx] for i, idx
                                              in enumerate(rules_in_buffer_indices_)]
                rule_buffer_text_new = ". ".join(rule_without_nums_shuffled) + "."

                answer_literals_new = []
                for i in range(len(answer_rules_text)):
                    gold_answer_idx_new = rule_initial_num + rules_in_buffer_indices_.index(answer_indices_in_buffer[i])
                    answer_literal_new = "according to rule "+str(gold_answer_idx_new)+", i need to prove "+answer_rules_text[i]
                    answer_literals_new.append(answer_literal_new)
                answer_literals_new = sorted(answer_literals_new)

                instance_new["input"] = instance["input"][:rule_buffer_span[0]] + \
                                        rule_buffer_text_new + instance["input"][rule_buffer_span[1]:]
                instance_new["output"] = " or ".join(answer_literals_new) if " or " in instance["output"] else \
                                        " )and( ".join(answer_literals_new) + ". </s>"
                instances_augmented.append(instance_new)

                if debug_flag:
                    print("-" * 20)
                    print("aug 2 input:", instance_new["input"])
                    print("aug 2 output:", instance_new["output"])

            # There should also be and augmentation option to shuffle the and/or order.
            # Hey, there might should not be. Because that means one input pattern may correspond to multiple output patterns.
            # if " or " in instance_new["output"] or " )and( " in instance_new["output"]:
            #     answer_literals_list = instance_new["output"][:-6].split(" or ") if " or " in instance_new["output"] \
            #                             else instance_new["output"][:-6].split(" )and( ")
            #     random.shuffle(answer_literals_list)
            #     instance_new["output"] = " or ".join(answer_literals_list) if " or " in instance_new["output"] \
            #                                     else " )and( ".join(answer_literals_list) + ". </s>"
            #
            #     instances_augmented.append(instance_new)
            #
            #     if debug_flag:
            #         print("-" * 20)
            #         print("aug 3 input:", instance_new["input"])
            #         print("aug 3 output:", instance_new["output"])

    if debug_flag:
        input("press enter to continue ... ")

    print("data aug 10 finished")

    return instances_augmented


def train_with_batch():
    t5Exp = T5Vanilla(0.0001, device)

    instances = loadAsSingleTasks(fact_buffer_size = int(FACT_BUFFER_SIZE), rule_buffer_size = int(RULE_BUFFER_SIZE),
                                  train_amount_option = train_amount)

    train_module_patterns = {
        "c": ["pattern"+str(pattern) for pattern in range(1,6,1)]+
             ["pattern7", "pattern8", "pattern9", "pattern11", "pattern12"],
        "f": ["pattern6"],
        "r": ["pattern10"],
        "s": ["pattern"+str(pattern) for pattern in range(1,13,1)]  # This is the single neural module method,
    }

    print("train module patterns")
    print(train_module_patterns)

    train_pairs = []
    for pattern in train_module_patterns[train_module]:
        instances_sampled = random.sample(instances['train'][pattern], 500)
        train_pairs.extend(instances_sampled)
        if pattern in ["pattern5", "pattern9"]:
            instances_augmented = augment_pattern_5_9(instances_sampled)
            train_pairs.extend(instances_augmented)

        if pattern == "pattern6":
            instances_augmented = augment_pattern_6(instances_sampled, False)
            train_pairs.extend(instances_augmented)

            tf_examples = [instance for instance in train_pairs if "CWA" in instance["output"]
                           or "NAF" in instance["output"]]
            n_tf_examples = [instance for instance in train_pairs if "CWA" not in instance["output"]
                             and "NAF" not in instance["output"]]

            print("tf example:", tf_examples[0])
            print("n tf example:", n_tf_examples[0])

            #train_pairs = tf_examples + [random.choice(n_tf_examples) for i in range(len(tf_examples) * 3)]
            print("pattern 6 statistics, t/f examples:", len(tf_examples),
                  " literal examples:", len(n_tf_examples))

        if pattern == "pattern10":
            instances_augmented = augment_pattern_10(instances_sampled, False)
            train_pairs.extend(instances_augmented)
            tf_examples = [instance for instance in train_pairs if "true" in instance["output"]
                                                               or "false" in instance["output"]]
            n_tf_examples = [instance for instance in train_pairs if "true" not in instance["output"]
                                                               and "false" not in instance["output"]]

            train_pairs = tf_examples + [random.choice(n_tf_examples) for i in range(len(tf_examples)*3)]
            print("pattern 10 statistics, t/f examples:", len(tf_examples),
                  " literal examples:", len(n_tf_examples))

    print("n train pairs:", len(train_pairs))

    ruletaker_train_dataset = RuleTakerDataset(train_pairs)
    ruletaker_train_dataloader = DataLoader(ruletaker_train_dataset, batch_size=2,
                                            shuffle=True, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

    if not os.path.exists("saved_models/"):
        os.mkdir("saved_models/")

    save_folder_path = "saved_models/20210124_t5_small_less_training_500/"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    min_epoch_dist = 1000
    epoch_dists = []
    pattern_dists_flat_prev = [0]

    try:
        for epoch in range(N_EPOCH):
            epoch_train_start_time = time.time()

            print("=" * 20)
            print("epoch "+str(epoch))

            t5Exp.train_iters_batch(ruletaker_train_dataloader, n_iters = N_TRAIN_EACH_EPOCH, print_every=2000)

            epoch_train_end_time = time.time()
            epoch_train_time = epoch_train_end_time-epoch_train_start_time

            print('-'*20)
            epoch_eval_start_time = time.time()


            # temporarily comment this out for fast training

            pattern_dists = []
            pattern_dists_flat = []
            for pattern in train_module_patterns[train_module]:
                ruletaker_dev_dataset = RuleTakerDataset(instances["dev"][pattern][:min(N_EVAL_EACH_EPOCH, len(instances["dev"][pattern]))])
                ruletaker_dev_dataloader = DataLoader(ruletaker_dev_dataset, batch_size=4,
                                                        shuffle=True, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

                count_dict = t5Exp.evaluate_iters_batch_keyword(ruletaker_dev_dataloader, pattern, constraint=False)
                #count_dict = t5Exp.evaluate_iters_batch(ruletaker_dev_dataloader, pattern)
                pattern_dists.append(count_dict)
                pattern_dists_flat.extend(count_dict.values())

            epoch_eval_end_time = time.time()
            epoch_eval_time = epoch_eval_end_time - epoch_eval_start_time

            #epoch_dist =sum(pattern_dists)/len(pattern_dists)
            #epoch_dists.append(pattern_dists+[epoch_dist]+[epoch_train_time, epoch_eval_time])
            print("epoch train time:", epoch_train_time, " epoch eval time:", epoch_eval_time)

            #if epoch_dist<min_epoch_dist:
            #    min_epoch_dist = epoch_dist
            print(pattern_dists_flat)
            if min(pattern_dists_flat)>0:
                t5Exp.save_tuned_model(save_folder_path+"/task_"+task_setting+"_module_"+train_module+
                                       "_amount_"+train_amount+"_fbs_"+FACT_BUFFER_SIZE+"_rbs_"+RULE_BUFFER_SIZE+
                                       "_seed_"+str(RANDOM_SEED))
                np.save(save_folder_path + "/task_" + task_setting + "_module_" + train_module +
                               "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE +
                               "_seed_" + str(RANDOM_SEED)+".csv", np.array(pattern_dists_flat))

                pattern_dists_flat_prev = pattern_dists_flat
                break


        t5Exp.save_tuned_model(save_folder_path + "/task_" + task_setting + "_module_" + train_module +
                               "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE +
                               "_seed_" + str(RANDOM_SEED))

        np.save(save_folder_path + "/task_" + task_setting + "_module_" + train_module +
                "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE +
                "_seed_" + str(RANDOM_SEED) + ".csv", np.array(pattern_dists_flat))

    except:
        t5Exp.save_tuned_model(save_folder_path + "/task_" + task_setting + "_module_" + train_module +
                               "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE +
                               "_seed_" + str(RANDOM_SEED))

        np.save(save_folder_path + "/task_" + task_setting + "_module_" + train_module +
                "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE +
                "_seed_" + str(RANDOM_SEED) + ".csv", np.array(pattern_dists_flat))

    with open(save_folder_path+"/task_"+task_setting+"_module_"+train_module+
                                   "_amount_"+train_amount+"_fbs_"+FACT_BUFFER_SIZE+"_rbs_"+RULE_BUFFER_SIZE+
                                   "_seed_"+str(RANDOM_SEED)+".pickle" , "wb") as handle:
        pickle.dump(pattern_dists, handle)

train_with_batch()