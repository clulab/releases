import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
sys.path+=[parent_folder_path, datasets_folder_path]

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.optim as optim

import pickle
import numpy as np
import time

import squad_retrieval, openbook_retrieval
import random
import datetime
import os

# TODO: maybe later move this function to another module
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# TODO: print essential information.


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

    def forward_train(self, query_token_ids, query_seg_ids, query_att_mask_ids, fact_token_ids, fact_seg_ids, fact_att_mask_ids):
        query_output_tensor_, _ = self.bert_q(input_ids = query_token_ids, token_type_ids = query_seg_ids, attention_mask = query_att_mask_ids)
        fact_output_tensor_, _ = self.bert_d(input_ids = fact_token_ids, token_type_ids = fact_seg_ids, attention_mask = fact_att_mask_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768, 1)
        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, self.n_neg_sample+1, 768)  # the middle number should be n_neg_sample+1

        return query_output_tensor, fact_output_tensor

    def forward_eval_query(self, query_token_ids, query_seg_ids, query_att_mask_ids):
        query_output_tensor_, _ = self.bert_q(input_ids = query_token_ids, token_type_ids = query_seg_ids, attention_mask = query_att_mask_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768)

        return query_output_tensor

    def forward_eval_fact(self, fact_token_ids, fact_seg_ids, fact_att_mask_ids):
        fact_output_tensor_, _ = self.bert_d(input_ids = fact_token_ids, token_type_ids = fact_seg_ids, attention_mask = fact_att_mask_ids)

        batch_size = fact_token_ids.size()[0]

        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, 768)

        return fact_output_tensor

    def train_epoch(self, optimizer, squad_retrieval_train_dataloader):
        self.train()

        total_loss = 0
        start_time = time.time()
        for i, batch in enumerate(squad_retrieval_train_dataloader):
            optimizer.zero_grad()

            query_output_tensor, fact_output_tensor = self.forward_train(batch["query_token_ids"].to(self.device),
                                                                 batch["query_seg_ids"].to(self.device),
                                                                 batch["query_att_mask_ids"].to(self.device),
                                                                 batch["fact_token_ids"].to(self.device),
                                                                 batch["fact_seg_ids"].to(self.device),
                                                                 batch["fact_att_mask_ids"].to(self.device))

            scores = torch.matmul(fact_output_tensor, query_output_tensor).squeeze(dim=2)

            label = batch["label_in_distractor"].to(self.device)

            loss = self.criterion(scores, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()

            if (i+1)%200==0:
                print("\t\tprocessing batch "+str(i+1)+" ... , batch time:"+str(time.time()-start_time)+ " avg loss:"+str(total_loss/(i+1)))
                start_time = time.time()

        return total_loss/len(squad_retrieval_train_dataloader)


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

def train_and_eval_model(args, saved_pickle_path = parent_folder_path + "/data_generated/squad_retrieval_data_seed_0_dev_2000.pickle"):
    # TODO: later think about a way to pass this folder directory in a clever way.

    N_EPOCH = args.n_epoch
    BATCH_SIZE_TRAIN = args.batch_size_train
    BATCH_SIZE_EVAL = args.batch_size_eval
    NUM_WORKERS = args.n_worker
    N_NEG_FACT = args.n_neg_sample
    DEVICE = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate BERT retriever, optimizer and tokenizer.
    bert_retriever = BertSQuADRetriever(N_NEG_FACT, DEVICE, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL)
    bert_retriever.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    optimizer = optim.Adam(bert_retriever.parameters(), lr=0.00001)

    now = datetime.datetime.now()
    date_time = str(now)[:10] + '_' + str(now)[11:13] + str(now)[14:16]

    if args.dataset=="squad":
        # Load SQuAD dataset and dataloader.
        squad_retrieval_data = squad_retrieval.convert_squad_to_retrieval(tokenizer, random_seed = args.seed, num_dev = args.num_dev)

        squad_retrieval_train_dataset = squad_retrieval.SQuADRetrievalDatasetTrain(instance_list=squad_retrieval_data["train_list"],
                                                                   sent_list=squad_retrieval_data["sent_list"],
                                                                   doc_list=squad_retrieval_data["doc_list"],
                                                                   resp_list=squad_retrieval_data["resp_list"],
                                                                   tokenizer=tokenizer,
                                                                   random_seed=args.seed,
                                                                    n_neg_sample = N_NEG_FACT)

        retrieval_train_dataloader = DataLoader(squad_retrieval_train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                                      shuffle=True, num_workers=NUM_WORKERS, collate_fn=squad_retrieval.PadCollateSQuADTrain())

        squad_retrieval_dev_dataset = squad_retrieval.SQuADRetrievalDatasetEvalQuery(instance_list=squad_retrieval_data["dev_list"],
                                                                     sent_list=squad_retrieval_data["sent_list"],
                                                                     doc_list=squad_retrieval_data["doc_list"],
                                                                     resp_list=squad_retrieval_data["resp_list"],
                                                                     tokenizer=tokenizer)

        retrieval_dev_dataloader = DataLoader(squad_retrieval_dev_dataset, batch_size=BATCH_SIZE_EVAL,
                                                    shuffle=False, num_workers=NUM_WORKERS, collate_fn=squad_retrieval.PadCollateSQuADEvalQuery())

        squad_retrieval_test_dataset = squad_retrieval.SQuADRetrievalDatasetEvalQuery(instance_list=squad_retrieval_data["test_list"],
                                                                      sent_list=squad_retrieval_data["sent_list"],
                                                                      doc_list=squad_retrieval_data["doc_list"],
                                                                      resp_list=squad_retrieval_data["resp_list"],
                                                                      tokenizer=tokenizer)

        retrieval_test_dataloader = DataLoader(squad_retrieval_test_dataset, batch_size=BATCH_SIZE_EVAL,
                                                     shuffle=False, num_workers=NUM_WORKERS, collate_fn=squad_retrieval.PadCollateSQuADEvalQuery())

        squad_retrieval_eval_fact_dataset = squad_retrieval.SQuADRetrievalDatasetEvalFact(instance_list=squad_retrieval_data["resp_list"],
                                                                          sent_list=squad_retrieval_data["sent_list"],
                                                                          doc_list=squad_retrieval_data["doc_list"],
                                                                          resp_list=squad_retrieval_data["resp_list"],
                                                                          tokenizer=tokenizer)

        retrieval_eval_fact_dataloader = DataLoader(squad_retrieval_eval_fact_dataset, batch_size=BATCH_SIZE_EVAL,
                                                          shuffle=False, num_workers=NUM_WORKERS,
                                                          collate_fn=squad_retrieval.PadCollateSQuADEvalFact())

        save_folder_path = parent_folder_path+'/data_generated/squad_retrieval_seed_' + str(args.seed) + "_" + date_time+"/"


    elif args.dataset=="openbook":
        train_list, dev_list, test_list, kb = openbook_retrieval.construct_retrieval_dataset_openbook(num_neg_sample = N_NEG_FACT, random_seed = args.seed)

        openbook_retrieval_train_dataset = openbook_retrieval.OpenbookRetrievalDatasetTrain(
            instance_list=train_list,
            kb=kb,
            tokenizer=tokenizer,
            num_neg_sample = N_NEG_FACT)

        retrieval_train_dataloader = DataLoader(openbook_retrieval_train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=True, num_workers=NUM_WORKERS,
                                                collate_fn=openbook_retrieval.PadCollateOpenbookTrain())

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

        save_folder_path = parent_folder_path+'/data_generated/openbook_retrieval_seed_' + str(args.seed) + "_" + date_time+"/"

    else:
        return 0

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Start evaluation.
    best_mrr = 0
    main_result_array = np.zeros((N_EPOCH, 3))
    for epoch in range(N_EPOCH):
        print("="*20)
        print("Epoch ", epoch+1)
        train_loss = bert_retriever.train_epoch(optimizer, retrieval_train_dataloader)
        dev_result_dict, test_result_dict = bert_retriever.eval_epoch(retrieval_dev_dataloader, retrieval_test_dataloader, retrieval_eval_fact_dataloader)

        dev_mrr = sum(dev_result_dict["mrr"])/len(dev_result_dict["mrr"])
        test_mrr = sum(test_result_dict["mrr"])/len(test_result_dict["mrr"])

        print("\t\tepoch "+str(epoch+1)+" training loss:"+str(train_loss)+" dev mrr:"+str(dev_mrr)+" test mrr:"+str(test_mrr))

        main_result_array[epoch,:] = [train_loss, dev_mrr, test_mrr]

        if dev_mrr > best_mrr:
            best_mrr = dev_mrr

            torch.save(bert_retriever, save_folder_path+"saved_bert_retriever")

            with open(save_folder_path+"dev_dict.pickle", "wb") as handle:
                pickle.dump(dev_result_dict, handle)

            with open(save_folder_path+"test_dict.pickle", "wb") as handle:
                pickle.dump(test_result_dict, handle)

        np.save(save_folder_path+"main_result.npy", main_result_array)

    return 0

def main():

    # Things need to be parsed:
    # device, batch_size, n_epoch, num_workers, n_neg_sample,
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size_train", type=int, default=1)  # batch size will indeed affact this. Larger batch size will cause out of memory issue
    parser.add_argument("--batch_size_eval", type=int, default=4)
    parser.add_argument("--n_epoch", type=int, default=4)
    parser.add_argument("--n_worker", type=int, default=3)
    parser.add_argument("--n_neg_sample", type=int, default=4)
    parser.add_argument("--num_dev", type=int, default=2000)
    parser.add_argument("--max_seq_len", type = int, default = 256)  # TODO: think about a way to pass this value to the collate function.
    parser.add_argument("--dataset", type=str, default="openbook")

    # parse the input arguments
    args = parser.parse_args()

    # set the random seeds
    torch.manual_seed(args.seed)  # set pytorch seed
    random.seed(args.seed)     # set python seed.
    # #This python random library is used in two places: one is constructing the raw dataset, the other is when constructing train data.
    np.random.seed(args.seed)   # set numpy seed

    torch.set_num_threads(1) # this has nothing to do with dataloader num worker

    print("="*20)
    print("args:", args)
    print("num thread:", torch.get_num_threads())
    print("="*20)

    train_and_eval_model(args)

    return 0

main()






