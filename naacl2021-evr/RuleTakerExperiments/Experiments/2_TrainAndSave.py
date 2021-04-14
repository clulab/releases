import sys
from pathlib import Path
import argparse

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
TRAIN_DEPTH = sys.argv[6]
RANDOM_SEED = 0

N_EPOCH = 40 # 40
N_TRAIN_EACH_EPOCH = 24000  #24000
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

def train_with_batch():
    t5Exp = T5Vanilla(0.0001, device)

    instances = loadAsSingleTasks(fact_buffer_size = int(FACT_BUFFER_SIZE), rule_buffer_size = int(RULE_BUFFER_SIZE),
                                  train_amount_option = train_amount, train_depth = TRAIN_DEPTH)

    train_module_patterns = {
        "c": ["pattern"+str(pattern) for pattern in range(1,6,1)]+
             ["pattern7", "pattern8", "pattern9", "pattern11", "pattern12"],
        "f": ["pattern6"],
        "r": ["pattern10"],
        "s": ["pattern"+str(pattern) for pattern in range(1,13,1)]  # This is the single neural module method,
    }

    print("train module patterns")
    print(train_module_patterns)


    if not os.path.exists("saved_models/"):
        os.mkdir("saved_models/")

    save_folder_path = "saved_models/20201118_t5_small/"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    min_epoch_dist = 1000
    epoch_dists = []
    for epoch in range(N_EPOCH):
        epoch_train_start_time = time.time()

        print("=" * 20)
        print("epoch "+str(epoch))
        train_pairs = []
        for pattern in train_module_patterns[train_module]:
            train_pairs.extend(instances['train'][pattern])

        ruletaker_train_dataset = RuleTakerDataset(train_pairs)
        ruletaker_train_dataloader = DataLoader(ruletaker_train_dataset, batch_size=4,
                                                shuffle=True, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

        t5Exp.train_iters_batch(ruletaker_train_dataloader, n_iters = N_TRAIN_EACH_EPOCH, print_every=2000)

        epoch_train_end_time = time.time()
        epoch_train_time = epoch_train_end_time-epoch_train_start_time


        print('-'*20)
        epoch_eval_start_time = time.time()

        pattern_dists = []
        for pattern in train_module_patterns[train_module]:
            ruletaker_dev_dataset = RuleTakerDataset(instances["dev"][pattern][:min(N_EVAL_EACH_EPOCH, len(instances["dev"][pattern]))])
            ruletaker_dev_dataloader = DataLoader(ruletaker_dev_dataset, batch_size=16,
                                                    shuffle=True, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

            pattern_dist = t5Exp.evaluate_iters_batch(ruletaker_dev_dataloader)
            pattern_dists.append(pattern_dist)

        epoch_eval_end_time = time.time()
        epoch_eval_time = epoch_eval_end_time - epoch_eval_start_time

        epoch_dist =sum(pattern_dists)/len(pattern_dists)
        epoch_dists.append(pattern_dists+[epoch_dist]+[epoch_train_time, epoch_eval_time])
        print("epoch train time:", epoch_train_time, " epoch eval time:", epoch_eval_time)

        if epoch_dist<min_epoch_dist:
            min_epoch_dist = epoch_dist
            t5Exp.save_tuned_model(save_folder_path+"/task_"+task_setting+"_module_"+train_module+
                                   "_amount_"+train_amount+"_fbs_"+FACT_BUFFER_SIZE+"_rbs_"+RULE_BUFFER_SIZE+
                                   "_seed_"+str(RANDOM_SEED))

        else:
            break

    np.savetxt(save_folder_path+"/task_"+task_setting+"_module_"+train_module+
                                   "_amount_"+train_amount+"_fbs_"+FACT_BUFFER_SIZE+"_rbs_"+RULE_BUFFER_SIZE+
                                   "_seed_"+str(RANDOM_SEED)+".csv",
               np.array(epoch_dists), delimiter=",")

train_with_batch()