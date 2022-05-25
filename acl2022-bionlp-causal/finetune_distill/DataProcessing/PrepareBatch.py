import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
data_folder_path = parent_folder_path+"/DataRaw"

sys.path+=[parent_folder_path, data_folder_path]


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

from transformers import BertTokenizer


def load_event_pairs_json():
    train_path = data_folder_path+"/event-pairs-python-train.json"
    test_path = data_folder_path+"/event-pairs-python-test.json"

    def load_event_pairs(path):
        with open(path) as f:
            data = list(json.load(f))

        for event_pair in data:
            # This gold label flag is to discriminate the labeld and unlabeled data.
            event_pair["gold_label_flag"] = 1

            if event_pair["relation"] == "E1 precedes E2":
                event_pair["relation_label"] = 1
            elif event_pair["relation"]=="E2 precedes E1":
                event_pair["relation_label"] = 2
            else:
                event_pair["relation_label"] = 0

        instance_list = []
        event_bound_invalid_count = 0
        event_sent_index_invalid_count = 0
        for event_pair in data:
            # Ideally this should not be used. All of the evnet pairs shoud have adjacent sentence indexes.
            # But the matching algorithm in scala does not fix somehow, so we have to do this here.
            e1_sent_index = int(event_pair["e1-sentence-index"])
            e2_sent_index = int(event_pair["e2-sentence-index"])

            e1_start = int(event_pair["e1-start"])
            e2_start = int(event_pair["e2-start"])

            if e1_sent_index == e2_sent_index:
                if e1_start <= e2_start:
                    instance_list.append(event_pair)
                else:
                    event_bound_invalid_count += 1
            elif e2_sent_index - e1_sent_index == 1:
                instance_list.append(event_pair)
            else:
                event_sent_index_invalid_count += 1

        print("n sample raw:", len(data))
        print("n invalid event bound:", event_bound_invalid_count, "n invalid sent index:", event_sent_index_invalid_count)
        print("n sample remaining:", len(instance_list))

        return instance_list

    return load_event_pairs(train_path), load_event_pairs(test_path)

def load_event_pairs_unlabeled_json():
    instance_path = data_folder_path + "/20210117_unlabeled_extractions/event_pairs_0_1000.json"

    def load_event_pairs(path):
        with open(path) as f:
            data = list(json.load(f))

        for event_pair in data:
            event_pair["relation_label"] = 3

        instance_list = []
        for event_pair in data:
            # TODO: fix this later, This should not be used. All of the evnet pairs shoud have adjacent sentence indexes.
            if abs(event_pair["e1-sentence-index"] - event_pair["e2-sentence-index"]) <= 1:
                # TODO: this is for temporal masking. Change this back later.
                event_pair["e1-sentence-entities"] = ["O"] * len(event_pair["e1-sentence-tokens"])
                event_pair["e2-sentence-entities"] = ["O"] * len(event_pair["e2-sentence-tokens"])
                instance_list.append(event_pair)

        return instance_list

    return load_event_pairs(instance_path)

def pad_tensor(token_indices, pad):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    return token_indices + [0]*(pad - len(token_indices))


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, model_name, tokenizer=BertTokenizer.from_pretrained("bert-base-cased")):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.model_name = model_name

        self.tokenizer = tokenizer

        self.debug_flag = False

    def pad_collate(self, batch):

        batch_returned = {}
        batch_returned["token_indices"] = [torch.tensor(sample["token_indices"], dtype=torch.int64) for sample in
                                           batch]

        batch_returned["target"] = torch.tensor([sample["target"] for sample in batch], dtype=torch.int64)
        batch_returned["gold_label_flag"] = torch.tensor([sample["gold_label_flag"] for sample in batch],
                                                         dtype=torch.int64)

        # start to generate bert input data for distillation.
        all_pair_text = [" ".join(sample["tokens"]) for sample in batch]
        bert_input_dict = self.tokenizer(all_pair_text, padding=True, truncation=True, max_length=512)

        batch_returned["input_ids"] = bert_input_dict['input_ids']
        batch_returned["attention_mask"] = bert_input_dict["attention_mask"]
        batch_returned["token_type_ids"] = bert_input_dict["token_type_ids"]

        if self.debug_flag:
            bert_tokens_list_debug = [self.tokenizer.tokenize(pair_text) for pair_text in all_pair_text]
            bert_ids_debug = [self.tokenizer.convert_tokens_to_ids(bert_tokens_debug) for bert_tokens_debug in bert_tokens_list_debug]
            print("="*20)
            print("tokens:", [sample["tokens"] for sample in batch])
            print("token indices (lstm):", batch_returned["token_indices"])
            print("input ids (bert):", batch_returned["input_ids"])
            print("att mask (bert):", batch_returned["attention_mask"])
            print("token type ids (bert):", batch_returned["token_type_ids"])
            print("bert tokens:", bert_tokens_list_debug)
            print("bert ids:", bert_ids_debug)
            input("-"*20)

        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)

# This is obselete. The function can be finished by "PadCollate"
class PadCollateTransformer():
    def __init__(self, model_name, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer

    def pad_collate(self, batch):

        # To use BERT with batch and padding, we need
        # (1) pad 0 to the origin tokens
        # (2) pad the attention mask, 1 for valid, 0 for padding.
        # (3) pad the segment mask. Should use all 0s for our case.
        if self.model_name.startswith("BERT"):
            all_pair_labels = [sample["target"] for sample in batch]

            if self.model_name == "BERT":
                all_pair_text = [" ".join(sample["tokens"]) for sample in batch]
            else:
                all_pair_text = [" ".join(sample["masked_tokens"]) for sample in batch]

            dict_to_return = self.tokenizer(all_pair_text, padding = True, truncation = True, max_length = 512)
            dict_to_return["target"] = all_pair_labels

            return dict_to_return
        else:
            return {}

    def __call__(self, batch):
        return self.pad_collate(batch)


class CausalDetectionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, instances_list, token_to_index, model_name, case_flag):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_instances = []
        self.model_name = model_name
        self.debug_flag = False
        self.debug_inter_sentence = False

        self.cased_lstm_vocab = case_flag

        for instance in instances_list:
            gold_label_flag = instance["gold_label_flag"]

            # According to Gus, both the 1 seg and 3 seg implementation should use this e1 [sep] inter [sep] e2 input.
            seg1_tokens, seg2_tokens, seg3_tokens = self.get_event_tokens(instance)
            all_tokens = seg1_tokens + ["[SEP]"] + seg2_tokens + ["[SEP]"] + seg3_tokens if len(seg2_tokens)>0 \
                        else seg1_tokens + ["[SEP]"] + seg3_tokens


            # The LSTM can be either cased or uncased.
            # No matter it is cased or uncased, the tokens are the same. Only the indices are different.

            if self.cased_lstm_vocab:
                seg1_indices = [token_to_index[token] if token in token_to_index else token_to_index["unk"] for token in seg1_tokens]
                seg2_indices = [token_to_index[token] if token in token_to_index else token_to_index["unk"] for token in seg2_tokens]
                seg3_indices = [token_to_index[token] if token in token_to_index else token_to_index["unk"] for token in seg3_tokens]

            else:
                seg1_indices = [token_to_index[token.lower()] if token.lower() in token_to_index else token_to_index["unk"] for token in
                                seg1_tokens]
                seg2_indices = [token_to_index[token.lower()] if token.lower() in token_to_index else token_to_index["unk"] for token in
                                seg2_tokens]
                seg3_indices = [token_to_index[token.lower()] if token.lower() in token_to_index else token_to_index["unk"] for token in
                                seg3_tokens]

            all_indices = seg1_indices + [token_to_index["[SEP]"]] + seg2_indices + [token_to_index["[SEP]"]] + seg3_indices \
                            if len(seg2_tokens)>0 \
                            else seg1_indices + [token_to_index["[SEP]"]] + seg3_indices

            target = instance["relation_label"]

            self.all_instances.append({"tokens": all_tokens, "token_indices": all_indices, "target": target,
                                       "seg1_tokens": seg1_tokens, "seg1_indices": seg1_indices,
                                        "seg2_tokens":seg2_tokens, "seg2_indices": seg2_indices,
                                        "seg3_tokens": seg3_tokens, "seg3_indices": seg3_indices,
                                       "gold_label_flag":gold_label_flag})

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
            if e1_sent_index<e2_sent_index:
                tokens = instance["e1-sentence-tokens"][e1_start:] + instance["e2-sentence-tokens"][:e2_end]
                entities = instance["e1-sentence-entities"][e1_start:] + instance["e2-sentence-entities"][:e2_end]
                masked_tokens = cls.mask_entities(tokens, entities)
            else:
                tokens = instance["e2-sentence-tokens"][e2_start:] + instance["e1-sentence-tokens"][:e1_end]
                entities = instance["e2-sentence-entities"][e2_start:] + instance["e1-sentence-entities"][:e1_end]
                masked_tokens = cls.mask_entities(tokens, entities)

        return tokens, masked_tokens

    @classmethod
    def mask_entities(cls, tokens, entities, mask_token = "_ENTITY_"):
        tokens_masked = tokens.copy()
        for idx, entity in enumerate(entities):
            if entity != "O":
                tokens_masked[idx] = mask_token

        return tokens_masked

    @classmethod
    def get_event_tokens(cls, instance):

        e1_sent_index = instance["e1-sentence-index"]
        e2_sent_index = instance["e2-sentence-index"]

        e1_start = int(instance["e1-start"])
        e1_end = int(instance["e1-end"])
        e2_start = int(instance["e2-start"])
        e2_end = int(instance["e2-end"])


        if e1_sent_index==e2_sent_index:
            assert (e1_start <= e2_start)

            # non-overlapping situation
            if e1_end<=e2_start:
                seg1_tokens = instance["e1-sentence-tokens"][int(e1_start): int(e1_end)]
                seg2_tokens = instance["e1-sentence-tokens"][int(e1_end): int(e2_start)]
                seg3_tokens = instance["e1-sentence-tokens"][int(e2_start): int(e2_end)]
            elif e2_end<=e1_start:
                seg1_tokens = instance["e1-sentence-tokens"][int(e2_start): int(e2_end)]
                seg2_tokens = instance["e1-sentence-tokens"][int(e2_end): int(e1_start)]
                seg3_tokens = instance["e1-sentence-tokens"][int(e1_start): int(e1_end)]

            # overlapping situation
            else:
                seg2_tokens = []
                if e1_start<=e2_start:
                    seg1_tokens = instance["e1-sentence-tokens"][int(e1_start): int(e1_end)]
                    seg3_tokens = instance["e1-sentence-tokens"][int(e2_start): int(e2_end)]
                else:
                    seg1_tokens = instance["e1-sentence-tokens"][int(e2_start): int(e2_end)]
                    seg3_tokens = instance["e1-sentence-tokens"][int(e1_start): int(e1_end)]
        else:
            assert (e1_sent_index == e2_sent_index-1)

            seg1_tokens = instance["e1-sentence-tokens"][int(e1_start): int(e1_end)]
            seg2_tokens = instance["e1-sentence-tokens"][int(e1_end) :] + \
                          instance["e2-sentence-tokens"][:int(e2_start)]
            seg3_tokens = instance["e2-sentence-tokens"][int(e2_start): int(e2_end)]


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
