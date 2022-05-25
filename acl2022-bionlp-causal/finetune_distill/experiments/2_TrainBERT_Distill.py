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
DATA_OPT = sys.argv[5]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

assert(MODEL_OPT in ["BERT", "BioBERT", "TinyBERT", "MiniBERT", "BioTinyBERT", "BioMiniBERT", "TinyBERTPubmed", "MiniBERTPubmed"])


class PadCollateDistillBERT:

    def __init__(self, tokenizer, seed, split, labeled_data_teacher_scores):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.tokenizer = tokenizer
        self.seed = seed
        self.split = split
        self.labeled_data_teacher_scores = labeled_data_teacher_scores
        self.debug_flag = False

    def pad_collate(self, batch):
        teacher_scores = []
        for sample in batch:
            if sample["gold_label_flag"] == 1:
                sample_id = str(sample["id"])
                teacher_scores.append(self.labeled_data_teacher_scores[sample_id][
                                          "bert_teacher_score_seed_" + str(self.seed) + "_split_" + str(self.split)])
            else:
                teacher_scores.append(sample["bert_teacher_score_seed_" + str(self.seed) + "_split_" + str(self.split)])
        teacher_scores_tensor = torch.tensor(teacher_scores, dtype=torch.float32)

        batch_returned = {"target": torch.tensor([sample["target"] for sample in batch], dtype=torch.int64),
                          "gold_label_flag": torch.tensor([sample["gold_label_flag"] for sample in batch],
                                                          dtype=torch.int64),
                          "teacher_pred_scores": teacher_scores_tensor}

        # start to generate bert input data for distillation.
        all_pair_text = [" ".join(sample["tokens"]) for sample in batch]
        bert_input_dict = self.tokenizer(all_pair_text, padding=True, truncation=True, max_length=512)

        batch_returned["input_ids"] = bert_input_dict['input_ids']
        batch_returned["attention_mask"] = bert_input_dict["attention_mask"]
        batch_returned["token_type_ids"] = bert_input_dict["token_type_ids"]
        batch_returned["tokens"] = all_pair_text

        if self.debug_flag:
            bert_tokens_list_debug = [self.tokenizer.tokenize(pair_text) for pair_text in all_pair_text]
            bert_ids_debug = [self.tokenizer.convert_tokens_to_ids(bert_tokens_debug) for bert_tokens_debug in
                              bert_tokens_list_debug]
            print("=" * 20)
            print("tokens:", [sample["tokens"] for sample in batch])
            print("token indices (lstm):", batch_returned["token_indices"])
            print("input ids (bert):", batch_returned["input_ids"])
            print("att mask (bert):", batch_returned["attention_mask"])
            print("token type ids (bert):", batch_returned["token_type_ids"])
            print("bert tokens:", bert_tokens_list_debug)
            print("bert ids:", bert_ids_debug)
            input("-" * 20)

        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)


class DistillMiniBERT:

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Open the files on alix or clara
    teacher_score_folder_path = "/home/zhengzhongliang/CLU_Projects/2020_ASKE/20220127_BioBERT_TeacherScore/" \
        if os.path.exists("/home/zhengzhongliang/CLU_Projects/2020_ASKE/20220127_BioBERT_TeacherScore/") else \
        "/work/zhengzhongliang/2020_ASKE/20220127_BioBERT_TeacherScore/"

    unlabeled_paper_scores_all_dir = teacher_score_folder_path + "all_event_pairs_biobert_teacher_score.json"

    unlabeled_paper_scores_2000_dir = {
        0: teacher_score_folder_path + "2000_events_seed_0.json",
        1: teacher_score_folder_path + "2000_events_seed_1.json",
        2: teacher_score_folder_path + "2000_events_seed_2.json",
        3: teacher_score_folder_path + "2000_events_seed_3.json",
        4: teacher_score_folder_path + "2000_events_seed_4.json",
    }

    labeled_data_score_dir = teacher_score_folder_path + "labeled_data_biobert_teacher_scores.json"

    @classmethod
    def load_unlabeled_data(cls,
                            seed,
                            data_opt,
                            debug_flag=False):
        '''
        This function should load unlabeled data as needed.

        According to this paper: https://arxiv.org/pdf/1903.12136.pdf
        The distillation (1) directly uses the logits and (2) mix the original data and the augmented data.

        :return: the train dataset and dataloader.
        '''

        assert (data_opt in ["labeled", "unlabeled_2k", "unlabeled_20k"])

        # According to here: https://arxiv.org/pdf/1804.07612.pdf, a batch size of 2 is not "too small"
        batch_size = {"labeled": 2,
                      "unlabeled_2k": 16,
                      "unlabeled_20k": 32}[data_opt]

        # Now load the unlabeled list.
        if data_opt == "labeled":
            unlabeled_list = []
        elif data_opt == "unlabeled_2k":
            unlabeled_list = RawDataProcessing.load_json(cls.unlabeled_paper_scores_2000_dir[seed])
            for inst in unlabeled_list:
                inst["gold_label_flag"] = 0  # This is needed to construct the dataset
                inst["relation_label"] = 0  # This is needed to construct the dataset
        else:  # Load all
            unlabeled_list = RawDataProcessing.load_json(cls.unlabeled_paper_scores_all_dir)
            for inst in unlabeled_list:
                inst["gold_label_flag"] = 0  # This is needed to construct the dataset
                inst["relation_label"] = 0  # This is needed to construct the dataset

        # Don't forget to load the scores for the labeled data:
        labeled_data_teacher_scores_dict = RawDataProcessing.load_json(cls.labeled_data_score_dir)

        return unlabeled_list, labeled_data_teacher_scores_dict, batch_size

    @classmethod
    def train_and_eval(cls):
        assert(MODEL_OPT in ["MiniBERT", "BioMiniBERT", "MiniBERTPubmed"])
        # Load raw data
        train_list_, test_list_ = RawDataProcessing.load_event_pairs_json()
        all_list = train_list_ + test_list_
        all_labels = [instance["relation_label"] for instance in all_list]

        # Cross validation split.
        # Flags to read and write of existing files might be disabled at debugging time.
        all_splits = RawDataProcessing.get_train_dev_test_splits(all_list, all_labels, save_flag=False, load_flag=False,
                                                                 shuffle_train_dev=False,
                                                                 n_train_dev_splits=TRAIN_DEV_SPLIT_NUM)

        root_folder = "saved_models_20220127_distill/"
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        save_folder_path = root_folder + "20220127_distill_model_" + MODEL_OPT + "_seed_" + str(RANDOM_SEED) + \
                           "_trainDevSplit_" + str(TRAIN_DEV_SPLIT_NUM) + "_lr_" + str(LR) + \
                           "_distill_data_" + DATA_OPT + "/"
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        all_test_indices = []
        all_folds_labels = []
        all_folds_preds = []

        all_folds_labels_dev = []
        all_folds_preds_dev = []
        # Looping over the splits to do cross validation.
        for split_num, (train_index, dev_index, test_index) in enumerate(all_splits):
            train_list = [all_list[idx] for idx in train_index]
            dev_list = [all_list[idx] for idx in dev_index]
            test_list = [all_list[idx] for idx in test_index]

            unlabeled_list, labeled_data_teacher_scores_dict, batch_size = cls.load_unlabeled_data(RANDOM_SEED, DATA_OPT)

            print("=" * 20)
            print("split num:", split_num, "  n train:", len(train_list), "  n test:", len(test_list))

            print("n e1 precedes e2 train:", len([1 for idx in range(len(train_list)) if all_labels[idx] == 1]))
            print("n e1 precedes e2 test:", len([1 for idx in range(len(test_list)) if all_labels[idx] == 1]))

            print("n e2 precedes e1 train:", len([1 for idx in range(len(train_list)) if all_labels[idx] == 2]))
            print("n e2 precedes e1 test:", len([1 for idx in range(len(test_list)) if all_labels[idx] == 2]))

            print("n unlabeled:", len(unlabeled_list))

            experiment = BERTExperiment(
                device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
                lr=float(LR),
                model_opt=MODEL_OPT)

            # Note that although many experiments seem to use 1e-5, it actually uses 2e-5.

            # Build the dataloader
            causal_detection_dataset_train = CausalDetectionDataset(train_list + unlabeled_list)
            causal_detection_dataloader_train = DataLoader(causal_detection_dataset_train, batch_size=batch_size,
                                                           shuffle=True, num_workers=0,
                                                           collate_fn=PadCollateDistillBERT(
                                                               tokenizer=experiment.tokenizer,
                                                               seed=RANDOM_SEED,
                                                               split=split_num,
                                                               labeled_data_teacher_scores=labeled_data_teacher_scores_dict,
                                                           ),
                                                           drop_last=False)

            causal_detection_dataset_dev = CausalDetectionDataset(dev_list)
            causal_detection_dataloader_dev = DataLoader(causal_detection_dataset_dev, batch_size=32,
                                                         shuffle=False, num_workers=0,
                                                         collate_fn=PadCollateBERT(tokenizer=experiment.tokenizer),
                                                         drop_last=False)

            causal_detection_dataset_test = CausalDetectionDataset(test_list)
            causal_detection_dataloader_test = DataLoader(causal_detection_dataset_test, batch_size=32,
                                                          shuffle=False, num_workers=0,
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
                experiment.train_distill_epoch(causal_detection_dataloader_train,
                                              experiment.tokenizer,
                                              print_every=20,
                                              debug_flag=False)
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
                        f1_test, test_label_list, best_pred_list, _ = experiment.eval_epoch(
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
                f1_test, test_label_list, best_pred_list, _ = experiment.eval_epoch(causal_detection_dataloader_test,
                                                                                 experiment.tokenizer,
                                                                                 debug_flag=False)
                best_test_f1 = f1_test

            print("\n\tbest fold test f1:", best_test_f1)
            all_test_indices.extend(test_index)

            all_folds_labels_dev.extend(dev_label_list)
            all_folds_preds_dev.extend(best_dev_pred_list)

            all_folds_labels.extend(test_label_list)
            all_folds_preds.extend(best_pred_list)

        # The calculation of p, r, f1 might be slightly different, so please take a look at the function to understand the logic.
        p_dev, r_dev, f1_dev = Metrics.calculate_p_r_f1(all_folds_labels_dev, all_folds_preds_dev)
        precision, recall, f1 = Metrics.calculate_p_r_f1(all_folds_labels, all_folds_preds)
        print("dev p:", p_dev, "r:", r_dev, "f1:", f1_dev)
        print("test p:", precision, "r:", recall, "f1:", f1)

        final_metrics_save_path = save_folder_path + "MiniBERT1Seg_Distill_seed_" + str(RANDOM_SEED) + "_trainDevSplit_" + str(
            TRAIN_DEV_SPLIT_NUM) + "_result.json"
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
                 "all_folds_preds_dev": all_folds_preds_dev}, handle)

DistillMiniBERT.train_and_eval()