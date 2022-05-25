import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
data_folder_path = parent_folder_path + "/DataRaw"
data_processing_path = parent_folder_path + "/DataProcessing"
model_path = parent_folder_path + "/ModelClasses"

sys.path += [parent_folder_path, data_folder_path, data_processing_path, model_path]

from BERTClass import BERTExperiment
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import precision_score, recall_score
import numpy as np
import torch
import os
import pickle

import torch.multiprocessing
from utils import RawDataProcessing, CausalDetectionDataset, PadCollateBERT, Metrics
from transformers import BertTokenizer
import copy
import json

torch.multiprocessing.set_sharing_strategy('file_system')

RANDOM_SEED = int(sys.argv[1])
TRAIN_DEV_SPLIT_NUM = int(sys.argv[2])
MODEL_OPT = sys.argv[3]
LR = float(sys.argv[4])

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

assert(MODEL_OPT in ["BERT", "BioBERT", "TinyBERT", "MiniBERT", "BioTinyBERT", "BioMiniBERT", "TinyBERTPubmed", "MiniBERTPubmed"])


class TrainBERT:

    @classmethod
    def train_and_eval(cls):

        print(" seed:", RANDOM_SEED)

        # Load raw data
        train_list_, test_list_ = RawDataProcessing.load_event_pairs_json()
        all_list = train_list_ + test_list_
        all_labels = [instance["relation_label"] for instance in all_list]

        # Cross validation split.
        # Flags to read and write of existing files might be disabled at debugging time.
        all_splits = RawDataProcessing.get_train_dev_test_splits(all_list, all_labels, save_flag=False,
                                                                 load_flag=False, shuffle_train_dev=False,
                                                                 n_train_dev_splits=TRAIN_DEV_SPLIT_NUM)

        save_root_dir = "saved_models_20220127/"
        if not os.path.exists(save_root_dir):
            os.mkdir(save_root_dir)

        save_folder_path = save_root_dir + "20220127_model_" + MODEL_OPT + "_seed_" + str(RANDOM_SEED) + \
                           "_trainDevSplit_" + str(TRAIN_DEV_SPLIT_NUM) + "_lr_" + str(LR) + "/"
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        all_test_indices = []
        all_folds_labels = []
        all_folds_preds = []

        all_folds_labels_dev = []
        all_folds_preds_dev = []

        all_folds_time = []
        # Looping over the splits to do cross validation.
        for split_num, (train_index, dev_index, test_index) in enumerate(all_splits):
            train_list = [all_list[idx] for idx in train_index]
            dev_list = [all_list[idx] for idx in dev_index]
            test_list = [all_list[idx] for idx in test_index]

            print("=" * 20)
            print("split num:", split_num, "  n train:", len(train_index), "  n test:", len(test_index))

            print("n e1 precedes e2 train:", len([1 for idx in train_index if all_labels[idx] == 1]))
            print("n e1 precedes e2 test:", len([1 for idx in test_index if all_labels[idx] == 1]))

            print("n e2 precedes e1 train:", len([1 for idx in train_index if all_labels[idx] == 2]))
            print("n e2 precedes e1 test:", len([1 for idx in test_index if all_labels[idx] == 2]))

            experiment = BERTExperiment(device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
                                        model_opt=MODEL_OPT,
                                        lr=LR)

            # Build the dataloader
            causal_detection_dataset_train = CausalDetectionDataset(train_list)
            causal_detection_dataloader_train = DataLoader(causal_detection_dataset_train, batch_size=2,
                                                           shuffle=True, num_workers=1,
                                                           collate_fn=PadCollateBERT(tokenizer=experiment.tokenizer),
                                                           drop_last=False)

            causal_detection_dataset_dev = CausalDetectionDataset(dev_list)
            causal_detection_dataloader_dev = DataLoader(causal_detection_dataset_dev, batch_size=1,
                                                         shuffle=False, num_workers=1,
                                                         collate_fn=PadCollateBERT(tokenizer=experiment.tokenizer),
                                                         drop_last=False)

            causal_detection_dataset_test = CausalDetectionDataset(test_list)
            causal_detection_dataloader_test = DataLoader(causal_detection_dataset_test, batch_size=1,
                                                          shuffle=False, num_workers=1,
                                                          collate_fn=PadCollateBERT(tokenizer=experiment.tokenizer),
                                                          drop_last=False)

            # question: what does this num_workers mean
            best_test_f1 = -1
            best_dev_f1 = -1
            best_pred_list = []
            patience_counter = 0

            for epoch in range(40):
                print("\t" + "-" * 20)
                print("\t training epoch " + str(epoch), " the model does not improve for ", patience_counter,
                      " epochs.")
                experiment.train_epoch(causal_detection_dataloader_train, experiment.tokenizer,)
                f1_dev, dev_label_list, dev_pred_list, _ = experiment.eval_epoch(causal_detection_dataloader_dev,
                                                                              experiment.tokenizer,
                                                                              debug_flag=False)

                if not (f1_dev >= 0 and f1_dev <= 1):
                    f1_dev = -1

                if f1_dev >= best_dev_f1:  # The train continues as long as the current f1 is not too worse than the best f1.
                    print("\n\tepoch dev f1:", f1_dev)
                    if f1_dev > best_dev_f1:
                        patience_counter = 0
                        best_dev_pred_list = dev_pred_list
                        best_dev_f1 = f1_dev
                        f1_test, test_label_list, best_pred_list, test_time_list = experiment.eval_epoch(
                            causal_detection_dataloader_test,
                            experiment.tokenizer,
                            debug_flag=False)
                        best_test_f1 = f1_test
                        torch.save(experiment.bert_classifier,
                                   save_folder_path + "BERT1Seg_split_" + str(split_num) + "_seed_" + str(
                                       RANDOM_SEED) +
                                   "_trainDevSplit_" + str(TRAIN_DEV_SPLIT_NUM))

                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        break

            if len(best_pred_list) == 0:
                f1_test, test_label_list, best_pred_list, test_time_list = experiment.eval_epoch(causal_detection_dataloader_test,
                                                                                 experiment.tokenizer,
                                                                                 debug_flag=False)
                best_test_f1 = f1_test

            print("\n\tbest fold test f1:", best_test_f1)
            all_test_indices.extend(test_index)

            all_folds_labels_dev.extend(dev_label_list)
            all_folds_preds_dev.extend(best_dev_pred_list)

            all_folds_labels.extend(test_label_list)
            all_folds_preds.extend(best_pred_list)

            all_folds_time.extend(test_time_list)

        # precision = precision_score(all_folds_labels, all_folds_preds, average="micro")
        # recall = recall_score(all_folds_labels, all_folds_preds, average="micro")
        # f1 = 2 * precision * recall / (precision + recall)

        # The calculation of p, r, f1 might be slightly different, so please take a look at the function to understand the logic.
        p_dev, r_dev, f1_dev = Metrics.calculate_p_r_f1(all_folds_labels_dev, all_folds_preds_dev)
        precision, recall, f1 = Metrics.calculate_p_r_f1(all_folds_labels, all_folds_preds)
        print("dev p:", p_dev, "r:", r_dev, "f1:", f1_dev)
        print("test p:", precision, "r:", recall, "f1:", f1)

        final_metrics_save_path = save_folder_path + "BERT1Seg_seed_" + str(RANDOM_SEED) + "_trainDevSplit_" + str(TRAIN_DEV_SPLIT_NUM) + "_result.json"
        with open(final_metrics_save_path, "w") as handle:
            json.dump(
                {"precision_dev": p_dev,
                 "recall_dev": r_dev,
                 "f1_dev": f1_dev,
                 "precision_test": precision,
                 "recall_test": recall,
                 "f1_test": f1,
                 "all_test_indices": all_test_indices,
                 "all_folds_labels": all_folds_labels,
                 "all_folds_preds": all_folds_preds,
                 "all_folds_labels_dev": all_folds_labels_dev,
                 "all_folds_preds_dev": all_folds_preds_dev,
                 "all_folds_time": all_folds_time}, handle)

        return 0


TrainBERT.train_and_eval()
