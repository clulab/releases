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


import torch
import os
import numpy as np
import random
import time

task_setting = sys.argv[1]  # 3nn / 1nn
train_module = sys.argv[2]  # c/f/r / s
train_amount = sys.argv[3]  # 10k / 30k / 70k
FACT_BUFFER_SIZE = sys.argv[4]  # 5 / 20
RULE_BUFFER_SIZE = sys.argv[5]  # 3 / 10
RANDOM_SEED = 0

pattern_num_in = sys.argv[6]

N_EPOCH = 40 # 40
N_TRAIN_EACH_EPOCH = 1000 if train_module=="c" else 1200  #24000
N_EVAL_EACH_EPOCH = 50  #2000

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

def check_trained_model():
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

    #save_model_path = "saved_models/20201021_t5_small_ruletaker_multitask_type_c"
    #save_model_path = "saved_models/20201021_t5_small_ruletaker_multitask_type_f"
    #save_model_path = "saved_models/20201111_t5_small_ruletaker_multitask_type_r"

    #save_model_path = "saved_models/20201217_t5_small_less_training/task_"+task_setting+"_module_"+train_module+ \
    #                              "_amount_"+train_amount+"_fbs_"+FACT_BUFFER_SIZE+"_rbs_"+RULE_BUFFER_SIZE+ \
    #                              "_seed_"+str(RANDOM_SEED)

    # save_model_path = "saved_models/20200105_t5_small_less_training/task_"+task_setting+"_module_"+train_module+ \
    #                              "_amount_"+train_amount+"_fbs_"+FACT_BUFFER_SIZE+"_rbs_"+RULE_BUFFER_SIZE+ \
    #                              "_seed_"+str(RANDOM_SEED)   # This is the single model

    # This is the directly for the R module
    if train_module == "c":
        save_model_path = "saved_models/20210111_t5_small_less_training/task_" + task_setting + "_module_" + train_module + \
                          "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE + \
                          "_seed_" + str(RANDOM_SEED)  # This is the c, f, r model trained on 100 examples each pattern.

    if train_module == "f":
        save_model_path = "saved_models/20210124_t5_small_less_training_100/task_" + task_setting + "_module_" + train_module + \
                          "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE + \
                          "_seed_" + str(RANDOM_SEED)  # This is the c, f, r model trained on 100 examples each pattern.

    if train_module == "r":
        save_model_path = "saved_models/20210124_t5_small_less_training_100/task_" + task_setting + "_module_" + train_module + \
                      "_amount_" + train_amount + "_fbs_" + FACT_BUFFER_SIZE + "_rbs_" + RULE_BUFFER_SIZE + \
                      "_seed_" + str(RANDOM_SEED)  # This is the c, f, r model trained on 100 examples each pattern.

    t5Exp.t5_model = torch.load(save_model_path)

    pattern_dists = []
    for pattern in ["pattern"+pattern_num_in]: #train_module_patterns[train_module]:
    #for pattern in train_module_patterns[train_module]:
        print('=' * 20)
        print("pattern ", pattern)
        ruletaker_dev_dataset = RuleTakerDataset(instances["dev"][pattern][:min(N_EVAL_EACH_EPOCH, len(instances["dev"][pattern]))])
        ruletaker_dev_dataloader = DataLoader(ruletaker_dev_dataset, batch_size=1,
                                                shuffle=False, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

        # pattern_dist = t5Exp.evaluate_iters_batch(ruletaker_dev_dataloader)

        #t5Exp.evaluate_iters(instances["dev"][pattern][:20], print_flag = True)
        count_dict = t5Exp.evaluate_iters_batch_keyword(ruletaker_dev_dataloader, pattern = pattern, debug_flag = True, constraint = True)

    # This is for full evaluations:
    # for pattern in train_module_patterns[train_module]:
    #     print('=' * 20)
    #     print("pattern ", pattern)
    #     ruletaker_dev_dataset = RuleTakerDataset(instances["dev"][pattern][:min(2000, len(instances["dev"][pattern]))])
    #     ruletaker_dev_dataloader = DataLoader(ruletaker_dev_dataset, batch_size=4,
    #                                             shuffle=True, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

        # pattern_dist = t5Exp.evaluate_iters_batch(ruletaker_dev_dataloader)

        #t5Exp.evaluate_iters(instances["dev"][pattern][:20], print_flag = True)
    #    count_dict = t5Exp.evaluate_iters_batch_keyword(ruletaker_dev_dataloader, pattern = pattern)


check_trained_model()