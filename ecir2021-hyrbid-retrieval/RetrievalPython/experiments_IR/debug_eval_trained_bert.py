import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
sys.path+=[parent_folder_path, datasets_folder_path]

import os
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader


from pytorch_pretrained_bert import BertTokenizer, BertModel

import time

import squad_retrieval, openbook_retrieval
import random
import datetime
import os

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class BertSQuADRetriever(nn.Module):
    def __init__(self, n_neg_sample, device, batch_size_train, batch_size_eval):
        super(BertSQuADRetriever, self).__init__()

        self.bert_q = BertModel.from_pretrained('bert-base-uncased')
        self.bert_d = BertModel.from_pretrained('bert-base-uncased')

        self.criterion = torch.nn.CrossEntropyLoss()

        self.n_neg_sample = n_neg_sample
        self.device = device
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval

class BertEvalLoader(nn.Module):
    def __init__(self, n_neg_sample, device, batch_size_train, batch_size_eval, bert_directory = ""):
        super(BertEvalLoader, self).__init__()

        bert_model = torch.load(bert_directory)

        self.bert_q = bert_model.bert_q
        self.bert_d = bert_model.bert_d

        self.n_neg_sample = n_neg_sample
        self.device = device
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval

    def forward_train(self, query_token_ids, query_seg_ids, query_att_mask_ids, fact_token_ids, fact_seg_ids,
                      fact_att_mask_ids):
        query_output_tensor_, _ = self.bert_q(input_ids=query_token_ids, token_type_ids=query_seg_ids,
                                              attention_mask=query_att_mask_ids)
        fact_output_tensor_, _ = self.bert_d(input_ids=fact_token_ids, token_type_ids=fact_seg_ids,
                                             attention_mask=fact_att_mask_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768, 1)
        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, self.n_neg_sample + 1,
                                                                768)  # the middle number should be n_neg_sample+1

        return query_output_tensor, fact_output_tensor

    def forward_eval_query(self, query_token_ids, query_seg_ids, query_att_mask_ids):
        query_output_tensor_, _ = self.bert_q(input_ids=query_token_ids, token_type_ids=query_seg_ids,
                                              attention_mask=query_att_mask_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768)

        return query_output_tensor

    def forward_eval_fact(self, fact_token_ids, fact_seg_ids, fact_att_mask_ids):
        fact_output_tensor_, _ = self.bert_d(input_ids=fact_token_ids, token_type_ids=fact_seg_ids,
                                             attention_mask=fact_att_mask_ids)

        batch_size = fact_token_ids.size()[0]

        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, 768)

        return fact_output_tensor

    def eval_epoch(self, retrieval_dev_dataloader, retrieval_test_dataloader, retrieval_eval_fact_dataloader):

        # ref size: 1,000*768 numpy array is about 3 MB.
        # dev query size: 2,000*768 = 6 MB.
        # test query size: 10,000*768 = 30 MB
        # facts size: 100,000*768 = 300 MB

        # score size without cut:
        # dev score size = 2,000 * 100,000 = 830 MB
        # test score size = 10,000 * 100,000 = 4.2 GB

        # What data we need to save for each question (for the best epoch):
        # gold fact index, gold fact ranking, gold fact score
        # top 64 fact scores.

        # What other things we need to save:
        # the best performed model (model with the best test mrr).
        # training loss, dev mrr, test mrr.

        self.eval()

        with torch.no_grad():
            # First step: compute all fact embeddings
            fact_embds = []
            for i, batch in enumerate(retrieval_eval_fact_dataloader):
                fact_embds_batch = self.forward_eval_fact(batch["fact_token_ids"].to(self.device), batch["fact_seg_ids"].to(self.device), batch["fact_att_mask_ids"].to(self.device))
                fact_embds.append(fact_embds_batch.detach().cpu().numpy())
                if (i+1)%100==0:
                    print("\tget fact "+str(i+1))
            fact_embds = np.transpose(np.concatenate(fact_embds, axis = 0))  # transpose the embedding for better multiplication.
            #fact_embds = np.random.rand(768, 102003)

            # Second step: compute the query embedding for each batch. At the same time return the needed results.
            dev_results_dict = {"mrr": [], "gold_fact_index": [], "gold_fact_ranking": [], "gold_fact_score": [], "top_64_facts":[], "top_64_scores":[]}
            for i, batch in enumerate(retrieval_dev_dataloader):
                query_embds_batch = self.forward_eval_query(batch["query_token_ids"].to(self.device),batch["query_seg_ids"].to(self.device),batch["query_att_mask_ids"].to(self.device))
                query_embds_batch = query_embds_batch.detach().cpu().numpy()

                self._fill_results_dict(batch, query_embds_batch, fact_embds, dev_results_dict)

                if (i+1)%100==0:
                    print("\tget dev query "+str(i+1))

            # Third step: compute the query embedding for each batch, then store the result to a dict.
            test_results_dict = {"mrr": [], "gold_fact_index": [], "gold_fact_ranking": [], "gold_fact_score": [], "top_64_facts":[], "top_64_scores":[]}
            for i, batch in enumerate(retrieval_test_dataloader):
                query_embds_batch = self.forward_eval_query(batch["query_token_ids"].to(self.device), batch["query_seg_ids"].to(self.device),batch["query_att_mask_ids"].to(self.device))
                query_embds_batch = query_embds_batch.detach().cpu().numpy()

                self._fill_results_dict(batch, query_embds_batch, fact_embds, test_results_dict)

                if (i+1)%100==0:
                    print("\tget test query "+str(i+1))

        return dev_results_dict, test_results_dict

    def _fill_results_dict(self, batch, query_embds_batch, fact_embds, result_dict):
        # Things to return:
        batch_size = len(query_embds_batch)

        gold_facts_indices = batch["response"].numpy().reshape((batch_size, 1))  # size: n_query * 1

        batch_scores = softmax(np.matmul(query_embds_batch, fact_embds))   # size: n_query * n_facts
        sorted_scores = np.flip(np.sort(batch_scores, axis=1), axis = 1)
        sorted_facts = np.flip(np.argsort(batch_scores, axis=1), axis= 1)

        gold_fact_rankings_indices_row,  gold_fact_rankings= np.where(sorted_facts==gold_facts_indices)

        result_dict["gold_fact_index"].extend(gold_facts_indices.flatten().tolist())
        result_dict["gold_fact_ranking"].extend(gold_fact_rankings.tolist())   # get the gold fact ranking of each query
        result_dict["gold_fact_score"].extend(sorted_scores[ gold_fact_rankings_indices_row,  gold_fact_rankings].tolist())
        result_dict["mrr"].extend((1/(1+gold_fact_rankings)).tolist())

        result_dict["top_64_facts"].extend(sorted_facts[:,:64].tolist())
        result_dict["top_64_scores"].extend(sorted_scores[:,:64].tolist())

        return 0

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    BATCH_SIZE_EVAL = 4
    NUM_WORKERS = 3

    train_list, dev_list, test_list, kb = openbook_retrieval.construct_retrieval_dataset_openbook(
        num_neg_sample=4, random_seed=0)

    openbook_retrieval_dev_dataset = openbook_retrieval.OpenbookRetrievalDatasetEvalQuery(
        instance_list=dev_list,
        tokenizer=tokenizer)

    retrieval_dev_dataloader = DataLoader(openbook_retrieval_dev_dataset, batch_size=BATCH_SIZE_EVAL,
                                          shuffle=False, num_workers=NUM_WORKERS,
                                          collate_fn=openbook_retrieval.PadCollateOpenbookEvalQuery())

    openbook_retrieval_test_dataset = openbook_retrieval.OpenbookRetrievalDatasetEvalQuery(
        instance_list=test_list,
        tokenizer=tokenizer)

    retrieval_test_dataloader = DataLoader(openbook_retrieval_test_dataset, batch_size=BATCH_SIZE_EVAL,
                                           shuffle=False, num_workers=NUM_WORKERS,
                                           collate_fn=openbook_retrieval.PadCollateOpenbookEvalQuery())

    openbook_retrieval_eval_fact_dataset = openbook_retrieval.OpenbookRetrievalDatasetEvalFact(
        kb=kb,
        tokenizer=tokenizer)

    retrieval_eval_fact_dataloader = DataLoader(openbook_retrieval_eval_fact_dataset, batch_size=BATCH_SIZE_EVAL,
                                                shuffle=False, num_workers=NUM_WORKERS,
                                                collate_fn=openbook_retrieval.PadCollateOpenbookEvalFact())

    device = torch.device("cuda:0")

    bertEvalLoader = BertEvalLoader(n_neg_sample=10,
                                    device = device,
                                    batch_size_train=1,
                                    batch_size_eval=BATCH_SIZE_EVAL,
                                    bert_directory="/home/zhengzhongliang/CLU_Projects/2019_QA/Hybrid_Retrieval/data_generated/openbook_retrieval_seed_1_2020-05-21_2225/saved_bert_retriever")
    bertEvalLoader.to(device)

    dev_result_dict, test_result_dict = bertEvalLoader.eval_epoch(retrieval_dev_dataloader, retrieval_test_dataloader,
                                                                  retrieval_eval_fact_dataloader)

    dev_mrr = sum(dev_result_dict["mrr"]) / len(dev_result_dict["mrr"])
    test_mrr = sum(test_result_dict["mrr"]) / len(test_result_dict["mrr"])

    print("dev_mrr:", dev_mrr)
    print("test_mrr:", test_mrr)

main()