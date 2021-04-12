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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train_with_1_sample():
    print("="*20)
    t5Exp = T5Vanilla(0.0001, device)

    instances = loadAsSingleTasks()
    train_pairs = []
    for pattern in ["pattern" + str(i) for i in range(1, 13, 1)]:
        train_pairs.extend(instances['train'][pattern])

    print("="*20)
    for epoch in range(3):
        t5Exp.train_iters(train_pairs, n_iters=2000, print_every=100)
        print('-'*20)
        for pattern in range(1,13,1):
            t5Exp.evaluate_iters_and_get_loss(instances["dev"]["pattern"+str(pattern)][:50])


def train_with_batch():
    t5Exp = T5Vanilla(0.0001, device)

    instances = loadAsSingleTasks()
    train_pairs = []
    for pattern in ["pattern" + str(i) for i in range(1, 13, 1)]:
        train_pairs.extend(instances['train'][pattern])

    ruletaker_train_dataset = RuleTakerDataset(train_pairs)
    ruletaker_train_dataloader = DataLoader(ruletaker_train_dataset, batch_size=4,
                                            shuffle=True, num_workers=1, collate_fn=PadCollate(t5Exp.tokenizer))

    print("="*20)
    for epoch in range(3):
        t5Exp.train_iters_batch(ruletaker_train_dataloader, n_iters=2000, print_every=100)
        print('-'*20)
        for pattern in range(1,13,1):
            t5Exp.evaluate_iters_and_get_loss(instances["dev"]["pattern"+str(pattern)][:50])

train_with_1_sample()
#train_with_batch()