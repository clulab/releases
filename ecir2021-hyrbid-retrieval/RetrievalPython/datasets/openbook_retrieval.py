from random import sample
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import os
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

import sys
from pathlib import Path

parent_folder_path = str(Path('.').absolute().parent)
sys.path+=[parent_folder_path]


def random_negative_from_kb(target_fact_num_list, kb_as_list, num_of_negative_facts):
    candidate_indexes = list(range(len(kb_as_list)))
    candidate_indexes_new = [x for x in candidate_indexes if x not in target_fact_num_list]
    selected_indexes = sample(candidate_indexes_new,num_of_negative_facts)

    return selected_indexes


def get_knowledge_base(kb_path: str):
    kb_data = list([])
    with open(kb_path, 'r') as the_file:
        kb_data = [line.strip() for line in the_file.readlines()]

    return kb_data

# Load questions as list of json files
def load_questions_json(question_path: str):
    questions_list = list([])
    with open(question_path, 'r', encoding='utf-8') as dataset:
        for i, line in enumerate(dataset):
            item = json.loads(line.strip())
            questions_list.append(item)

    return questions_list

def construct_dataset(train_path: str, dev_path: str, test_path: str, fact_path: str) -> (list, list, list):
    # This function is used to generate instances list for train, dev and test.
    def file_to_list(file_path: str, sci_facts: list) -> list:
        choice_to_id = {"A": 0, "B": 1, "C": 2, "D": 3}
        json_list = load_questions_json(file_path)

        instances_list = list([])
        for item in json_list:
            instance = {}
            instance["id"] = item["id"]
            for choice_id in range(4):
                if choice_id == choice_to_id[item['answerKey']]:
                    instance["text"] = item["question"]["stem"] + " " + item["question"]["choices"][choice_id]["text"]
                    gold_sci_fact = '\"' + item["fact1"] + '\"'
                    instance["label"] = sci_facts.index(gold_sci_fact)
            instances_list.append(instance)

        return instances_list

    sci_facts = get_knowledge_base(fact_path)

    train_list = file_to_list(train_path, sci_facts)
    dev_list = file_to_list(dev_path, sci_facts)
    test_list = file_to_list(test_path, sci_facts)

    return train_list, dev_list, test_list, sci_facts

def construct_retrieval_dataset_openbook(num_neg_sample, random_seed):
    train_path = parent_folder_path+"/data_raw/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl"
    dev_path = parent_folder_path+"/data_raw/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl"
    test_path = parent_folder_path+"/data_raw/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl"
    fact_path = parent_folder_path+"/data_raw/OpenBookQA-V1-Sep2018/Data/Main/openbook.txt"

    # Build model:
    # Construct dataset
    train_raw, dev_raw, test_raw, sci_kb = construct_dataset(train_path, dev_path, test_path, fact_path)

    random.seed(random_seed)
    def add_distractor(instances_list, kb_as_list):
        instances_list_new = list([])
        for instance in instances_list:
            target_fact_num = instance["label"]
            negative_indices = random_negative_from_kb([target_fact_num], kb_as_list, num_neg_sample)
            instance["documents"] = [target_fact_num]+negative_indices
            instance["query"] = [instance["text"]]
            instance["facts"] = [target_fact_num]
            instances_list_new.append(instance)

        return instances_list_new

    train_list = add_distractor(train_raw, sci_kb)
    dev_list = add_distractor(dev_raw, sci_kb)
    test_list = add_distractor(test_raw, sci_kb)

    print("openbook data constructed! train size:", len(train_list),"\tdev size:", len(dev_list),"\tkb size:", len(sci_kb))

    return train_list, dev_list, test_list, sci_kb

# define padcollate and dataset function for train, eval query and eval fact.

class PadCollateOpenbookTrain:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):
        """
        Nothing to add here
        """

    def pad_tensor(self, vec, pad, pad_value = 0):

        return vec + [pad_value] * (pad - len(vec))

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """

        # The input here is actually a list of dictionary.
        # find longest sequence
        max_len_query = max([len(sample["query_token_ids"]) for sample in batch]) # this should be equivalent to "for x in batch"
        # pad according to max_len
        for sample in batch:
            sample["query_token_ids"]  = self.pad_tensor(sample["query_token_ids"], pad=max_len_query)
            sample["query_att_mask_ids"] = self.pad_tensor(sample["query_att_mask_ids"], pad=max_len_query)
        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned = {}
        batch_returned["query_token_ids"] = torch.tensor([sample["query_token_ids"] for sample in batch])
        batch_returned["query_att_mask_ids"] = torch.tensor([sample["query_att_mask_ids"] for sample in batch])
        batch_returned["query_seg_ids"] = torch.tensor([[0]*max_len_query for sample in batch])

        all_facts_ids = []
        all_facts_att_mask_ids = []
        for sample in batch:
            all_facts_ids.extend(sample["fact_token_ids"])
            all_facts_att_mask_ids.extend(sample["fact_att_mask_ids"])

        max_len_fact = max([len(fact_token_ids) for fact_token_ids in all_facts_ids])

        for i, fact_ids in enumerate(all_facts_ids):
            all_facts_ids[i] = self.pad_tensor(fact_ids, pad=max_len_fact)

        for i, fact_att_mask_ids in enumerate(all_facts_att_mask_ids):
            all_facts_att_mask_ids[i] = self.pad_tensor(fact_att_mask_ids, pad=max_len_fact)
        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned["fact_token_ids"] = torch.tensor([fact_ids for fact_ids in all_facts_ids])
        batch_returned["fact_att_mask_ids"] = torch.tensor([fact_att_mask_ids for fact_att_mask_ids in all_facts_att_mask_ids])
        batch_returned["fact_seg_ids"] = torch.tensor([[0]*max_len_fact for fact_ids in all_facts_ids])

        batch_returned["label_in_distractor"] = torch.tensor([sample["label_in_distractor"] for sample in batch])

        return batch_returned

    #TODO: check if the keys in the batch are the same in openbook and in squad.

    def __call__(self, batch):
        return self.pad_collate(batch)

class OpenbookRetrievalDatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, instance_list, kb,  tokenizer, num_neg_sample):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.kb = kb
        self.tokenizer = tokenizer
        self.num_neg_sample = num_neg_sample

        self.instance_list=  []
        for instance in instance_list:
            # cls_id = 101; sep_id = 102; pad_id = 0;
            query_tokens = tokenizer.tokenize(instance["text"])   # this is for strip quotes
            query_token_ids = [101] + tokenizer.convert_tokens_to_ids(query_tokens) + [102]   # this does not include pad, cls or sep
            query_att_mask_ids = [1]*len(query_token_ids)   # use [1] on non-pad token

            instance["query_token_ids"] = query_token_ids
            instance["query_seg_ids"] = [0] * len(query_token_ids)
            instance["query_att_mask_ids"] = query_att_mask_ids

            instance["label_in_distractor"] = 0

            instance["documents"] = [instance["label"]] + random_negative_from_kb(
                [instance["label"]], self.kb, self.num_neg_sample)

            # TODO: should not use in-batch negative sampling because it causes worse performance
            fact_token_ids = []
            fact_seg_ids = []
            fact_att_mask_ids = []
            for fact_index in instance["documents"]:
                single_fact_tokens = self.tokenizer.tokenize(
                    self.kb[fact_index][1:-1])  # this if for removing the quotes
                single_fact_token_ids = [101] + self.tokenizer.convert_tokens_to_ids(single_fact_tokens) + [102]
                fact_token_ids.append(single_fact_token_ids)
                fact_seg_ids.append([0] * len(single_fact_token_ids))
                fact_att_mask_ids.append([1] * len(single_fact_token_ids))  # use [1] on non-pad token

            instance["fact_token_ids"] = fact_token_ids
            instance["fact_seg_ids"] = fact_seg_ids
            instance["fact_att_mask_ids"] = fact_att_mask_ids

        self.instance_list = instance_list


    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):

        return self.instance_list[idx]

class PadCollateOpenbookEvalQuery:
    def __init__(self):
        """
        Nothing to add here
        """
    def _pad_tensor(self, vec, pad):
        return vec + [0] * (pad - len(vec))

    def pad_collate(self, batch):
        # The input here is actually a list of dictionary.
        # find longest sequence
        max_len_query = max([len(sample["query_token_ids"]) for sample in batch]) # this should be equivalent to "for x in batch"
        # pad according to max_len
        for sample in batch:
            sample["query_token_ids"]  = self._pad_tensor(sample["query_token_ids"], pad=max_len_query)
            sample["query_att_mask_ids"] = self._pad_tensor(sample["query_att_mask_ids"], pad = max_len_query)
        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned = {}
        batch_returned["query_token_ids"] = torch.tensor([sample["query_token_ids"] for sample in batch])
        batch_returned["query_att_mask_ids"] = torch.tensor([sample["query_att_mask_ids"] for sample in batch])
        batch_returned["query_seg_ids"] = torch.tensor([[0]*max_len_query for sample in batch])
        batch_returned["response"] = torch.tensor([sample["label"] for sample in batch])

        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)

class OpenbookRetrievalDatasetEvalQuery(Dataset):
    def __init__(self, instance_list, tokenizer):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.instance_list=  []
        for instance in instance_list:
            # cls_id = 101; sep_id = 102; pad_id = 0;
            query_tokens = tokenizer.tokenize(instance["text"])  # this is for strip quotes
            query_token_ids = [101]+ tokenizer.convert_tokens_to_ids(query_tokens)+[102]    # this does not include pad, cls or sep

            instance["query_token_ids"] = query_token_ids
            instance["query_seg_ids"] = [0]*len(query_token_ids)  # use seg id 0 for query.
            instance["query_att_mask_ids"] = [1]*len(query_token_ids)   # use [1] on non-pad token

        self.instance_list = instance_list

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        return self.instance_list[idx]

class PadCollateOpenbookEvalFact:
    def __init__(self):
        """
        Nothing to add here
        """

    def _pad_tensor(self, vec, pad):
        return vec + [0] * (pad - len(vec))

    def pad_collate(self, batch):
        # The input here is actually a list of dictionary.
        # find longest sequence
        batch_returned = {}
        all_facts_ids = []
        all_facts_att_mask_ids = []

        max_len_fact = max([len(sample["fact_token_ids"]) for sample in batch])

        for sample in batch:
            all_facts_ids.append(self._pad_tensor(sample["fact_token_ids"], pad=max_len_fact)[:min(256, max_len_fact)])
            all_facts_att_mask_ids.append(self._pad_tensor(sample["fact_att_mask_ids"], pad=max_len_fact)[:min(256, max_len_fact)])

        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned["fact_token_ids"] = torch.tensor([fact_ids for fact_ids in all_facts_ids])
        batch_returned["fact_seg_ids"] = torch.tensor([[0]*min(max_len_fact, 256) for fact_ids in all_facts_ids])
        batch_returned["fact_att_mask_ids"] = torch.tensor([fact_att_mask_ids for fact_att_mask_ids in all_facts_att_mask_ids])

        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)

class OpenbookRetrievalDatasetEvalFact(Dataset):
    def __init__(self, kb, tokenizer):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.instance_list=  []
        for sent in kb:
            # cls_id = 101; sep_id = 102; pad_id = 0;
            fact_token_ids = [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
            fact_seg_ids = [0]*len(fact_token_ids)

            instance_new = {}
            instance_new["fact_token_ids"] = fact_token_ids
            instance_new["fact_seg_ids"] = fact_seg_ids
            instance_new["fact_att_mask_ids"] = [1]*len(fact_token_ids)   # use [1] on non-pad tokens

            self.instance_list.append(instance_new)

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        return self.instance_list[idx]


def check_openbook_dataloader(n_neg_fact = 3, seed = 0, batch_size = 2, num_workers = 3):
    n_batch_check =3

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook(
        num_neg_sample=n_neg_fact, random_seed=seed)


    '''
    ===============================================================
    Check openbook retrieval train
    '''
    openbook_retrieval_train_dataset = OpenbookRetrievalDatasetTrain(
        instance_list=train_list,
        kb=kb,
        tokenizer=tokenizer,
        num_neg_sample = n_neg_fact)

    retrieval_train_dataloader = DataLoader(openbook_retrieval_train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            collate_fn=PadCollateOpenbookTrain())

    print("="*20+"\n check training")
    for i, batch in enumerate(retrieval_train_dataloader):
        if i>=n_batch_check:
            break

        print("-"*20)
        print(batch["query_token_ids"].size())
        print(batch["fact_token_ids"].size())
        print("\n")

        for j in range(batch["query_token_ids"].size()[0]):
            print("query ids:", batch["query_token_ids"][j])
            print("query seg ids:", batch["query_seg_ids"][j])
            print("query att mask ids:", batch["query_att_mask_ids"][j])
            print("query tokens:", tokenizer.convert_ids_to_tokens(batch["query_token_ids"][j].tolist()))
            print("\n")

            for k in range(n_neg_fact+1):
                print("\tfact ids:", batch["fact_token_ids"][j*(n_neg_fact+1)+k])
                print("\tfact seg ids:", batch["fact_seg_ids"][j*(n_neg_fact+1)+k])
                print("\tfact att mask ids:", batch["fact_att_mask_ids"][j*(n_neg_fact+1)+k])
                print("\tfact tokens:", tokenizer.convert_ids_to_tokens(batch["fact_token_ids"][j*(n_neg_fact+1)+k].tolist()))
                print("\n")

        input("AAA")



    '''
    ===============================================================
    Check openbook retrieval dev query
    '''
    openbook_retrieval_dev_dataset = OpenbookRetrievalDatasetEvalQuery(
        instance_list=dev_list,
        tokenizer=tokenizer)

    retrieval_dev_dataloader = DataLoader(openbook_retrieval_dev_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers,
                                          collate_fn=PadCollateOpenbookEvalQuery())

    print("=" * 20 + "\n check dev query")
    for i, batch in enumerate(retrieval_dev_dataloader):
        if i >= n_batch_check:
            break

        print("-" * 20)
        print(batch["query_token_ids"].size())
        print("\n")

        for j in range(batch["query_token_ids"].size()[0]):
            print("query ids:", batch["query_token_ids"][j])
            print("query seg ids:", batch["query_seg_ids"][j])
            print("query att mask ids:", batch["query_att_mask_ids"][j])
            print("query tokens:", tokenizer.convert_ids_to_tokens(batch["query_token_ids"][j].tolist()))
            print("\n")

        input("AAA")

    '''
    ===============================================================
    Check openbook retrieval test query
    '''
    openbook_retrieval_test_dataset = OpenbookRetrievalDatasetEvalQuery(
        instance_list=test_list,
        tokenizer=tokenizer)

    retrieval_test_dataloader = DataLoader(openbook_retrieval_test_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers,
                                           collate_fn=PadCollateOpenbookEvalQuery())

    print("=" * 20 + "\n check test query")
    for i, batch in enumerate(retrieval_test_dataloader):
        if i >= n_batch_check:
            break

        print("-" * 20)
        print(batch["query_token_ids"].size())
        print("\n")

        for j in range(batch["query_token_ids"].size()[0]):
            print("query ids:", batch["query_token_ids"][j])
            print("query seg ids:", batch["query_seg_ids"][j])
            print("query att mask ids:", batch["query_att_mask_ids"][j])
            print("query tokens:", tokenizer.convert_ids_to_tokens(batch["query_token_ids"][j].tolist()))
            print("\n")

        input("AAA")

    '''
    ===============================================================
    Check openbook retrieval eval fact
    '''
    openbook_retrieval_eval_fact_dataset = OpenbookRetrievalDatasetEvalFact(
        kb=kb,
        tokenizer=tokenizer)

    retrieval_eval_fact_dataloader = DataLoader(openbook_retrieval_eval_fact_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers,
                                                collate_fn=PadCollateOpenbookEvalFact())

    print("=" * 20 + "\n check eval fact")
    for i, batch in enumerate(retrieval_train_dataloader):
        if i >= n_batch_check:
            break

        print("-" * 20)
        print(batch["fact_token_ids"].size())
        print("\n")

        for j in range(batch["fact_token_ids"].size()[0]):
            print("fact ids:", batch["fact_token_ids"][j])
            print("fact seg ids:", batch["fact_seg_ids"][j])
            print("fact att mask ids:", batch["fact_att_mask_ids"][j])
            print("fact tokens:", tokenizer.convert_ids_to_tokens(batch["fact_token_ids"][j].tolist()))
            print("\n")

        input("AAA")

def save_openbook_pickle(n_neg_fact = 1, seed = 0):
    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook(
        num_neg_sample=n_neg_fact, random_seed=seed)

    for instance in dev_list:
        instance["question"] = instance["text"]
    for instance in test_list:
        instance["question"] = instance["text"]

    for i in range(len(kb)):
        kb[i] = kb[i][1:-1]

    print("="*20)
    print(dev_list[0])
    print("=" * 20)
    print(test_list[0])
    print("="*20)
    print(kb[0])

    with open("openbook_useqa_retrieval_data.pickle", "wb") as handle:
        pickle.dump({"dev_list":dev_list, "test_list":test_list, "kb":kb}, handle)

    return 0

def generate_data_for_lucene():
    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook(1,0)

    with open('openbook_dev_query.txt', 'a') as the_file:
        for instance in dev_list:
            the_file.write(instance["text"]+"\n")

    with open('openbook_dev_label.txt', 'a') as the_file:
        for instance in dev_list:
            the_file.write(str(instance["label"])+"\n")

    with open('openbook_test_query.txt', 'a') as the_file:
        for instance in test_list:
            the_file.write(instance["text"] + "\n")

    with open('openbook_test_label.txt', 'a') as the_file:
        for instance in test_list:
            the_file.write(str(instance["label"]) + "\n")

    return 0
