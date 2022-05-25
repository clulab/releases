import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
data_folder_path = parent_folder_path+"/DataRaw"
data_processing_path = parent_folder_path+"/DataProcessing"
model_path = parent_folder_path+"/ModelClasses"

sys.path+=[parent_folder_path, data_folder_path, data_processing_path, model_path]

import os
import random
import pickle
import json
import torch

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader


class RawDataProcessing:

    @classmethod
    def load_json(cls, file_path):
        with open(file_path, "r") as handle:
            result = json.load(handle)
        return result

    @classmethod
    def load_event_pairs_json(cls, print_flag=True):
        train_path = data_folder_path+"/event-pairs-python-train.json"
        test_path = data_folder_path+"/event-pairs-python-test.json"

        instance_id_dict = {}  # This is for tracking repeating examples.

        def load_event_pairs(path):
            with open(path) as f:
                data = list(json.load(f))

            for event_pair in data:
                # This gold label flag is to discriminate the labeled and unlabeled data.
                event_pair["gold_label_flag"] = 1

                # This is just a placeholder for later distillation.
                # If there is a trained teacher, this pred score is the prediction of the teacher.
                event_pair["teacher_pred_scores"] = [0.0, 0.0, 0.0]

                if event_pair["relation"] == "E1 precedes E2":
                    event_pair["relation_label"] = 1
                elif event_pair["relation"] == "E2 precedes E1":
                    event_pair["relation_label"] = 2
                else:
                    event_pair["relation_label"] = 0

            instance_list = []
            event_bound_invalid_count = 0
            event_sent_index_invalid_count = 0
            for event_pair in data:
                # Ideally this should not be used. All of the event pairs should have adjacent sentence indexes.
                # But the matching algorithm in scala does not fix somehow, so we have to do this here.
                if event_pair["id"] not in instance_id_dict:
                    e1_sent_index = int(event_pair["e1-sentence-index"])
                    e2_sent_index = int(event_pair["e2-sentence-index"])

                    e1_start = int(event_pair["e1-start"])
                    e2_start = int(event_pair["e2-start"])

                    if e1_sent_index == e2_sent_index:
                        if e1_start <= e2_start:
                            instance_list.append(event_pair)
                            instance_id_dict[event_pair["id"]] = 1
                        else:
                            event_bound_invalid_count += 1
                    elif e2_sent_index - e1_sent_index == 1:
                        instance_list.append(event_pair)
                        instance_id_dict[event_pair["id"]] = 1
                    else:
                        event_sent_index_invalid_count += 1

            for instance in instance_list:
                # Get the seg 1, seg 2, seg 3 tokens
                e1_sent_index = int(instance["e1-sentence-index"])
                e2_sent_index = int(instance["e2-sentence-index"])

                e1_start = int(instance["e1-start"])
                e1_end = int(instance["e1-end"])
                e2_start = int(instance["e2-start"])
                e2_end = int(instance["e2-end"])
                if e1_sent_index == e2_sent_index:
                    instance["seg1_tokens"] = instance["e1-sentence-tokens"][e1_start: e1_end]
                    instance["seg2_tokens"] = [] if e1_end >= e2_start else instance["e1-sentence-tokens"][e1_end: e2_start]
                    instance["seg3_tokens"] = instance["e2-sentence-tokens"][e2_start: e2_end]

                else:  # e2_index = e1_index + 1
                    instance["seg1_tokens"] = instance["e1-sentence-tokens"][e1_start: e1_end]
                    instance["seg2_tokens"] = instance["e1-sentence-tokens"][e1_end:] + \
                                              instance["e2-sentence-tokens"][:e2_start]
                    instance["seg3_tokens"] = instance["e2-sentence-tokens"][e2_start: e2_end]

            if print_flag:
                print("n sample raw:", len(data))
                print("n invalid event bound:", event_bound_invalid_count, "n invalid sent index:", event_sent_index_invalid_count)
                print("n sample remaining:", len(instance_list))

            return instance_list

        return load_event_pairs(train_path), load_event_pairs(test_path)

    @classmethod
    def get_train_dev_test_splits(cls, all_list, all_labels, save_flag=True,
                                  load_flag=True, shuffle_train_dev=True, n_train_dev_splits=5, print_flag=True):
        all_splits = []

        if shuffle_train_dev:
            split_file_name = "event_pairs_splits_shuffle.pickle"
        else:
            split_file_name = "event_pairs_splits.pickle"

        skf = StratifiedKFold(n_splits=5, shuffle=False)
        if print_flag:
            print("number of splits:", skf.get_n_splits(all_list, all_labels))
        if (not load_flag) or ((not os.path.exists(split_file_name)) and load_flag):
            if print_flag:
                print("generating splits from script.")
            all_splits_ = [(train_dev_index, test_index) for (train_dev_index, test_index) in skf.split(all_list, all_labels)]

            for (train_dev_index, test_index) in all_splits_:
                if shuffle_train_dev:
                    skf_train_dev = StratifiedKFold(n_splits=n_train_dev_splits, shuffle=True)
                else:
                    skf_train_dev = StratifiedKFold(n_splits=n_train_dev_splits, shuffle=False)
                split_train_dev_labels = [all_labels[idx] for idx in train_dev_index]

                # train_index_ returns the indices of train split in train_dev_index
                # dev_index_ returns the indices of dev split in train_dev_index
                train_index_, dev_index_ = list(skf_train_dev.split(train_dev_index, split_train_dev_labels))[0]
                train_index = [train_dev_index[idx] for idx in train_index_]
                dev_index = [train_dev_index[idx] for idx in dev_index_]
                test_index = test_index.tolist()

                all_splits.append((train_index, dev_index, test_index))

            if save_flag:
                print("saving splits to disk ...")
                with open(split_file_name, "wb") as handle:
                    pickle.dump(all_splits, handle)
        else:
            if load_flag:
                print("loading splits from disk ...")
                with open(split_file_name, "rb") as handle:
                    all_splits = pickle.load(handle)

        return all_splits


class CausalDetectionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, instances_list):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_instances = []
        self.debug_flag = False
        self.debug_inter_sentence = False

        for instance in instances_list:
            # According to Gus, both the 1 seg and 3 seg implementation should use this e1 [sep] inter [sep] e2 input.
            seg1_tokens, seg2_tokens, seg3_tokens = self.get_event_tokens(instance)
            all_tokens = seg1_tokens + ["[SEP]"] + seg2_tokens + ["[SEP]"] + seg3_tokens if len(seg2_tokens) > 0 \
                        else seg1_tokens + ["[SEP]"] + seg3_tokens

            target = instance["relation_label"]

            instance["tokens"] = all_tokens
            instance["target"] = target
            instance["seg1_tokens"] = seg1_tokens
            instance["seg2_tokens"] = seg2_tokens
            instance["seg3_tokens"] = seg3_tokens

            self.all_instances.append(instance)

            if self.debug_flag:
                print("="*20)
                self._debug_show_original_data(instance)
                print("seg 1 tokens:", seg1_tokens)
                print("seg 2 tokens:", seg2_tokens)
                print("seg 3 tokens:", seg3_tokens)
                print("constructed tokens:", all_tokens)
                input("-----")

    @classmethod
    def get_event_tokens_by_bound(cls, instance):
        e1_sent_index = instance["e1-sentence-index"]
        e2_sent_index = instance["e2-sentence-index"]

        e1_start = int(instance["e1-start"])
        e1_end = int(instance["e1-end"])
        e2_start = int(instance["e2-start"])
        e2_end = int(instance["e2-end"])

        if e1_sent_index == e2_sent_index:
            pair_start = min([e1_start, e1_end, e2_start, e2_end])
            pair_end = max([e1_start, e1_end, e2_start, e2_end])

            tokens = instance["e1-sentence-tokens"][pair_start:pair_end]
            entities = instance["e1-sentence-entities"][pair_start:pair_end]

            masked_tokens = cls.mask_entities(tokens, entities)

        else:
            # Theoretically, abs(e1_sent_index-e2_sent_index) should <=1. If not, the data should be discarded.
            tokens = []
            if e1_sent_index < e2_sent_index:
                tokens = instance["e1-sentence-tokens"][e1_start:] + instance["e2-sentence-tokens"][:e2_end]
                entities = instance["e1-sentence-entities"][e1_start:] + instance["e2-sentence-entities"][:e2_end]
                masked_tokens = cls.mask_entities(tokens, entities)
            else:
                tokens = instance["e2-sentence-tokens"][e2_start:] + instance["e1-sentence-tokens"][:e1_end]
                entities = instance["e2-sentence-entities"][e2_start:] + instance["e1-sentence-entities"][:e1_end]
                masked_tokens = cls.mask_entities(tokens, entities)

        return tokens, masked_tokens

    @classmethod
    def mask_entities(cls, tokens, entities, mask_token="_ENTITY_"):
        tokens_masked = tokens.copy()
        for idx, entity in enumerate(entities):
            if entity != "O":
                tokens_masked[idx] = mask_token

        return tokens_masked

    @classmethod
    def get_event_tokens(cls, instance):

        seg1_tokens = [token.lower() for token in instance["seg1_tokens"]]
        seg2_tokens = [token.lower() for token in instance["seg2_tokens"]]
        seg3_tokens = [token.lower() for token in instance["seg3_tokens"]]

        return seg1_tokens, seg2_tokens, seg3_tokens

    def _debug_show_original_data(self, instance):
        if self.debug_inter_sentence:
            if instance["e1-sentence-index"]!=instance["e2-sentence-index"]:
                print("sent 1 index:", instance["e1-sentence-index"], " sent 2 index:", instance["e2-sentence-index"])
                print("sent 1 tokens:", instance["e1-sentence-tokens"])
                print("sent 2 tokens:", instance["e2-sentence-tokens"])
                print("e1 tokens:", instance["e1-sentence-tokens"][int(instance["e1-start"]):int(instance["e1-end"])])
                print("e2 tokens:", instance["e2-sentence-tokens"][int(instance["e2-start"]):int(instance["e2-end"])])
        else:
            print("sent 1 index:", instance["e1-sentence-index"], " sent 2 index:", instance["e2-sentence-index"])
            print("sent 1 tokens:", instance["e1-sentence-tokens"])
            print("sent 2 tokens:", instance["e2-sentence-tokens"])
            print("e1 tokens:", instance["e1-sentence-tokens"][int(instance["e1-start"]):int(instance["e1-end"])])
            print("e2 tokens:", instance["e2-sentence-tokens"][int(instance["e2-start"]):int(instance["e2-end"])])

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        return self.all_instances[idx]


class PadCollateBERT:

    def __init__(self, tokenizer):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.tokenizer = tokenizer
        self.debug_flag = False

    def pad_collate(self, batch):
        batch_returned = {"target": torch.tensor([sample["target"] for sample in batch], dtype=torch.int64),
                          "gold_label_flag": torch.tensor([sample["gold_label_flag"] for sample in batch],
                                                          dtype=torch.int64),
                          "teacher_pred_scores": torch.tensor([sample["teacher_pred_scores"] for sample in batch], dtype=torch.float32)}

        # start to generate bert input data for distillation.
        all_pair_text = [" ".join(sample["tokens"]) for sample in batch]
        bert_input_dict = self.tokenizer(all_pair_text, padding=True, truncation=True, max_length=512)

        batch_returned["input_ids"] = bert_input_dict['input_ids']
        batch_returned["attention_mask"] = bert_input_dict["attention_mask"]
        batch_returned["token_type_ids"] = bert_input_dict["token_type_ids"]

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


class PadCollateLSTM:

    def __init__(self, tokenizer, embd_opt):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.tokenizer = tokenizer
        self.token_to_index = tokenizer.vocab
        self.debug_flag = False
        self.embd_opt = embd_opt

    def pad_collate(self, batch):
        batch_returned = {"target": torch.tensor([sample["target"] for sample in batch], dtype=torch.int64),
                          "gold_label_flag": torch.tensor([sample["gold_label_flag"] for sample in batch],
                                                          dtype=torch.int64),
                          "teacher_pred_scores": torch.tensor([sample["teacher_pred_scores"] for sample in batch], dtype=torch.float32)}

        if self.embd_opt == "w2v_general" or self.embd_opt == "w2v_in_domain":
            batch_returned["token_indices"] = []
            for sample in batch:

                seg1_indices = [
                    self.token_to_index[token.lower()] if token.lower() in self.token_to_index else self.token_to_index["unk"] for
                    token in
                    sample["seg1_tokens"]]
                seg2_indices = [
                    self.token_to_index[token.lower()] if token.lower() in self.token_to_index else self.token_to_index["unk"] for
                    token in
                    sample["seg2_tokens"]]
                seg3_indices = [
                    self.token_to_index[token.lower()] if token.lower() in self.token_to_index else self.token_to_index["unk"] for
                    token in
                    sample["seg3_tokens"]]

                all_indices = seg1_indices + [self.token_to_index["[SEP]"]] + seg2_indices + [
                    self.token_to_index["[SEP]"]] + seg3_indices \
                    if len(sample["seg2_tokens"]) > 0 \
                    else seg1_indices + [self.token_to_index["[SEP]"]] + seg3_indices

                batch_returned["token_indices"].append(torch.tensor(all_indices, dtype=torch.int64))
        else:
            assert(self.embd_opt == "bert_pubmed_10000")
            all_tokens = []
            for sample in batch:
                if len(sample["seg2_tokens"]) == 0:
                    all_tokens.append(" ".join(sample["seg1_tokens"] + ["[SEP]"] + sample["seg3_tokens"]))
                else:
                    all_tokens.append(" ".join(sample["seg1_tokens"] + ["[SEP]"] + sample["seg2_tokens"] + ["[SEP]"] + sample["seg3_tokens"]))

            # This [1:] is to eliminate the [CLS] embedding
            batch_returned["token_indices"] = [input_tensor[1: ]
                                               for input_tensor in self.tokenizer(all_tokens, return_tensors="pt",
                                                             padding=True, truncation=True, max_length=512)["input_ids"]]

        batch_returned["seg1_tokens"] = [sample["seg1_tokens"] for sample in batch]
        batch_returned["seg2_tokens"] = [sample["seg2_tokens"] for sample in batch]
        batch_returned["seg3_tokens"] = [sample["seg3_tokens"] for sample in batch]
        if self.debug_flag:
            print("=" * 20)
            print("tokens:", [sample["tokens"] for sample in batch])
            input("-" * 20)

        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)


class Metrics:

    @classmethod
    def calculate_p_r_f1(cls, label_list, pred_list):
        tp = 0
        fp = 0
        fn = 0

        smooth = 1e-7

        for i in range(len(label_list)):
            if pred_list[i] != 0 and label_list[i] == pred_list[i]:
                tp += 1

            if pred_list[i] != 0 and label_list[i] != pred_list[i]:
                fp += 1

            if pred_list[i] == 0 and label_list[i] != pred_list[i]:
                fn += 1

        precision = tp/(tp+fp+smooth)
        recall = tp/(tp+fn+smooth)
        f1 = 2*precision*recall/(precision + recall+smooth)

        return precision, recall, f1
