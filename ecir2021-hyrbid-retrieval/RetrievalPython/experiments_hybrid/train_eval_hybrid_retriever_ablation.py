import sys
from pathlib import Path
import argparse


parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
generated_data_path = parent_folder_path+"/data_generated/"
bm25_folder = str(Path('.').absolute().parent.parent)+"/IR_BM25/"

sys.path+=[parent_folder_path, datasets_folder_path, generated_data_path, bm25_folder]

import numpy as np
import pickle
import csv
import time
import os

from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression
from debug_dataloader import LoadRawData
import random

dataset_statistics = {
    "openbook":{"n_dev":500 , "n_test":500, "n_kb":1326},
    "squad":{"n_dev":10000 , "n_test":11426, "n_kb":101957},
    "nq":{"n_test": 74097, "n_kb": 239013}
}

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    # This is to prevent overflow of large bm25 scores when using softmax.
    # MAX_FLOAT is 3.402823466 E + 38
    if np.max(x) > 85:  # this value is calculated by: log(MAX_FLOAT/64), where MAX_FLOAT is the maximum number of float64.
        x = x / np.max(x) * 85
    return np.exp(x) / np.sum(np.exp(x))

def get_data_statistics():

    print("squad statistics")
    with open(bm25_folder+"data/squad/raw_data/squad_dev_query.txt","r") as handle:
        query_dev_list = handle.read().split(" QUERY_SEP\n ")[:-1]

    with open(bm25_folder+"data/squad/raw_data/squad_test_query.txt","r") as handle:
        query_test_list = handle.read().split(" QUERY_SEP\n ")[:-1]

    with open(bm25_folder+"data/squad/raw_data/squad_kb.txt","r") as handle:
        resp_list = handle.read().split(" DOC_SEP\n ")[:-1]

    print("\tdev len:", len(query_dev_list))
    print("\ttest len:", len(query_test_list))
    print("\tkb len:", len(resp_list))

    print("nq statistics")

    with open(bm25_folder + "data/nq/raw_data/nq_test_query.txt", "r") as handle:
        query_test_list = handle.read().split(" QUERY_SEP\n ")[:-1]

    with open(bm25_folder + "data/nq/raw_data/nq_kb.txt", "r") as handle:
        resp_list = handle.read().split(" DOC_SEP\n ")[:-1]

    print("\ttest len:", len(query_test_list))
    print("\tkb len:", len(resp_list))


class LogisticRegressionRouter():
    def __init__(self, result_dict_bm25_dev, result_dict_useqa_dev, dataset, feature_pos):
        self.max_dev_score_bm25 = self._get_max_score_bm25(result_dict_bm25_dev)  # max scores must be obtained on dev.
        self.dataset = dataset
        self.n_feature = 14
        self.feature_position = np.array(feature_pos)
        # indicate which feature to use. feature 0~6: top bm25 scores. feature 7~13: top useqa scores

        self.train_array, self.labels = self._build_data_for_LR( result_dict_bm25_dev, result_dict_useqa_dev)

        self.reg = LogisticRegression(solver= 'lbfgs')
        self.reg.fit(self.train_array, self.labels)

    def reranking_by_lr_router(self, bm25_result_dict, useqa_result_dict):
        n_top_facts = 2**6
        mrr_list = []
        neural_used = []
        time_list = []
        for query_idx, top_scores in enumerate(bm25_result_dict["top_scores"]):
            start_time = time.time()
            input_array = np.zeros((1,14))

            # Be careful not to modify the input data.
            top_bm25_scores_ = bm25_result_dict["top_scores"][query_idx]
            if len(top_bm25_scores_)<(n_top_facts):
                top_bm25_scores_ = np.concatenate([top_bm25_scores_, np.zeros(n_top_facts-len(top_bm25_scores_))])
            else:
                top_bm25_scores_ = top_bm25_scores_[:n_top_facts]
            top_bm25_scores_ = softmax(top_bm25_scores_)

            top_useqa_scores_ = useqa_result_dict["top_scores"][query_idx]
            if len(top_useqa_scores_) < (n_top_facts):
                top_useqa_scores_ = np.concatenate([top_useqa_scores_, np.zeros(n_top_facts - len(top_useqa_scores_))])
            else:
                top_useqa_scores_ = top_useqa_scores_[:n_top_facts]
            #top_useqa_scores_ = softmax(top_useqa_scores_)


            for i in range(7):
                input_array[0, i] = np.mean(top_bm25_scores_[:2 ** i])
            for i in range(7):
                input_array[0, 7+i] = np.mean(top_useqa_scores_[:2 ** i])

            prediction = self.reg.predict(input_array[:,self.feature_position])
            end_time = time.time()
            time_list.append(end_time-start_time)


            if prediction>0.5:
                mrr_list.append(useqa_result_dict["mrr"][query_idx])
                neural_used.append(1)
            else:
                mrr_list.append(bm25_result_dict["mrr"][query_idx])
                neural_used.append(0)

        return np.array(mrr_list), np.array(neural_used), np.array(time_list)

    def _get_max_score_bm25(self, result_dict):
        max_score = 0
        for top_scores in result_dict["top_scores"]:
            if np.max(top_scores)>max_score:
                max_score = np.max(top_scores)

        return max_score

    def _build_data_for_LR(self, bm25_result_dict, useqa_result_dict):

        # first normalize the top 2000 BM25 scores with softmax.
        top_bm25_scores_list_copy = []
        for query_idx, top_scores in enumerate(bm25_result_dict["top_scores"]):
            top_bm25_scores_list_copy.append(top_scores)

        top_useqa_scores_list_copy = []
        for query_idx, top_scores in enumerate(useqa_result_dict["top_scores"]):
            top_useqa_scores_list_copy.append(top_scores)

        # be careful not to modify the original data.
        cleaned_bm25_array = []
        for query_idx in range(len(bm25_result_dict["mrr"])):
            if len(top_bm25_scores_list_copy[query_idx])< 64:
                score_len = len(top_bm25_scores_list_copy[query_idx])
                cleaned_bm25_array.append(np.concatenate([top_bm25_scores_list_copy[query_idx], np.zeros(64-score_len)]).reshape(64))
            else:
                cleaned_bm25_array.append(top_bm25_scores_list_copy[query_idx][:64])

            cleaned_bm25_array[query_idx] = softmax(cleaned_bm25_array[query_idx])

        cleaned_bm25_array = np.array(cleaned_bm25_array)

        cleaned_useqa_array = []
        for query_idx in range(len(bm25_result_dict["mrr"])):
            if len(top_useqa_scores_list_copy[query_idx]) < 64:
                score_len = len(top_useqa_scores_list_copy[query_idx])
                cleaned_useqa_array.append(
                    np.concatenate([top_useqa_scores_list_copy[query_idx], np.zeros(64 - score_len)]).reshape(64))
            else:
                cleaned_useqa_array.append(top_useqa_scores_list_copy[query_idx][:64])

            #cleaned_useqa_array[query_idx] = softmax(cleaned_useqa_array[query_idx])

        cleaned_useqa_array = np.array(cleaned_useqa_array)


        train_array = np.zeros((len(bm25_result_dict["mrr"]), 14))
        for i in range(7):
            train_array[:, i] = np.mean(cleaned_bm25_array[:, :2 ** i], axis=1)
        for i in range(7):
            train_array[:, 7+i] = np.mean(cleaned_useqa_array[:, :2 ** i], axis=1)

        labels = np.zeros(len(bm25_result_dict["mrr"]))
        labels[useqa_result_dict["mrr"]>bm25_result_dict["mrr"]]=1

        print("n useqa better/total:",np.sum(labels), "/", len(labels))

        # for i in range(10):
        #     print(train_array[i,:])
        #     input("A")

        return train_array[:, self.feature_position], labels

class LinearSum():

    def __init__(self, bm25_dev_dict, useqa_dev_dict, dataset):
        self.dataset = dataset
        self.best_coef = self._get_best_linear_coefficient(bm25_dev_dict, useqa_dev_dict)

    def _get_best_linear_coefficient(self, bm25_dev_dict, useqa_dev_dict):
        print("looking for the best coefficient for linear combination ...")
        best_mrr = 0
        best_coef = 0
        for coef in np.arange(0, 1.1, 0.1):
            mrr_array, _ = self._reranking_by_linear_combination(bm25_dev_dict, useqa_dev_dict, coef)
            if np.mean(mrr_array)>best_mrr:
                best_mrr = np.mean(mrr_array)
                best_coef = coef

        print("best coef:", best_coef)

        return best_coef

    def _reranking_by_linear_combination(self, bm25_result_dict, useqa_result_dict, coef):
        mrr_list = []
        time_list = []

        # this uses softmax over all retrieved top 2000 facts. Maybe we should also try that on hyrbid lr model.
        for sample_idx in range(len(bm25_result_dict["mrr"])):
            start_time = time.time()
            gold_label = useqa_result_dict["gold_fact_index"][sample_idx]

            bm25_score_recon = np.zeros(dataset_statistics[self.dataset]["n_kb"])  # get how many facts in total
            bm25_score_recon[bm25_result_dict["top_facts"][sample_idx]] = softmax(
                bm25_result_dict["top_scores"][sample_idx])

            useqa_score_recon = np.zeros(dataset_statistics[self.dataset]["n_kb"])
            useqa_score_recon[useqa_result_dict["top_facts"][sample_idx]] = softmax(
                useqa_result_dict["top_scores"][sample_idx])

            rel_doc_bool_idx_bm25 = np.zeros(dataset_statistics[self.dataset]["n_kb"]).astype(bool)
            rel_doc_bool_idx_bm25[bm25_result_dict["top_facts"][sample_idx]] = True

            rel_doc_bool_idx_useqa = np.zeros(dataset_statistics[self.dataset]["n_kb"]).astype(bool)
            rel_doc_bool_idx_useqa[useqa_result_dict["top_facts"][sample_idx]] = True

            rel_doc_bool_idx_combined = rel_doc_bool_idx_bm25 + rel_doc_bool_idx_useqa

            combined_scores = coef*bm25_score_recon + (1-coef)*useqa_score_recon  # This is still the full size.

            combined_scores = combined_scores[rel_doc_bool_idx_combined]
            rel_doc_idx_combined = np.arange(dataset_statistics[self.dataset]["n_kb"])[rel_doc_bool_idx_combined]

            sorted_facts = rel_doc_idx_combined[
                np.flip(np.argsort(combined_scores))].tolist()  # sort in the order of descending score
            end_time = time.time()
            time_list.append(end_time - start_time)

            mrr = 1 / (1 + sorted_facts.index(gold_label)) if gold_label in sorted_facts else 0
            mrr_list.append(mrr)

        return np.array(mrr_list), np.array(time_list)

    def reranking_by_linear_combination(self, bm25_test_result_dict, useqa_test_result_dict):
        mrr, time = self._reranking_by_linear_combination(bm25_test_result_dict, useqa_test_result_dict, self.best_coef)

        print("=" * 20)
        print("mrr bm25:", np.mean(bm25_test_result_dict["mrr"]))
        print("mrr useqa:", np.mean(useqa_test_result_dict["mrr"]))
        print("best coef:", self.best_coef)
        print("mrr linear sum socre:", np.mean(mrr))
        print("average sum and reranking time:", np.mean(time))

        return mrr, time

def debug_experiment(dataset = "openbook"):
    dataset_results = LoadRawData(dataset)

    lrClassifier = LogisticRegressionRouter(dataset_results.result_dev_bm25, dataset_results.result_dev_useqa, dataset)
    lrClassifier.reranking_by_lr_router(dataset_results.result_test_bm25, dataset_results.result_test_useqa)


#debug_experiment("openbook")
#get_data_statistics()

class ExperimentOpenbook():
    def __init__(self, feature_pos):
        self.dataset = "openbook"
        self.dataset_results = LoadRawData(self.dataset)
        self.feature_pos = feature_pos

    def run_all_exp(self):

        # linear_sum = LinearSum(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa,
        #                        self.dataset)
        # _ = linear_sum.reranking_by_linear_combination(self.dataset_results.result_dev_bm25,
        #                                                self.dataset_results.result_dev_useqa)
        # linear_sum_mrr, linear_sum_time = linear_sum.reranking_by_linear_combination(
        #     self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        # do hyrbid logistic regression with warm ups
        lrClassifier = LogisticRegressionRouter(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa,
                                                self.dataset, self.feature_pos)
        _ =lrClassifier.reranking_by_lr_router(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa)
        hybrid_lr_mrr, hybrid_lr_neural_used, hybrid_lr_time =lrClassifier.reranking_by_lr_router(self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        final_result_dict = {
            "hybrid_lr":{"mrr": hybrid_lr_mrr, "router_output":hybrid_lr_neural_used, "time":hybrid_lr_time}
        }
        return np.mean(hybrid_lr_mrr)
        # with open(generated_data_path+"hybrid_classifier_result/openbook_hybrid_result.pickle", "wb") as handle:
        #     pickle.dump(final_result_dict, handle)

class ExperimentSquad():
    def __init__(self, feature_pos):
        self.dataset = "squad"
        self.dataset_results = LoadRawData(self.dataset)
        self.feature_pos = feature_pos

    def _split_dev_data(self):
        dev_splits = []
        for split in range(5):
            split_indices = np.arange(split*2000, (split+1)*2000)

            bm25_dev_split_dict = {}
            bm25_dev_split_dict["mrr"] = self.dataset_results.result_dev_bm25["mrr"][split_indices]
            bm25_dev_split_dict["top_scores"] = self.dataset_results.result_dev_bm25["top_scores"][split_indices]
            bm25_dev_split_dict["top_facts"] = self.dataset_results.result_dev_bm25["top_facts"][split_indices]

            useqa_dev_split_dict = {}
            useqa_dev_split_dict["gold_fact_index"] = self.dataset_results.result_dev_useqa["gold_fact_index"][
                split_indices]
            useqa_dev_split_dict["mrr"] = self.dataset_results.result_dev_useqa["mrr"][split_indices]
            useqa_dev_split_dict["top_scores"] = self.dataset_results.result_dev_useqa["top_scores"][split_indices]
            useqa_dev_split_dict["top_facts"] = self.dataset_results.result_dev_useqa["top_facts"][split_indices]
            useqa_dev_split_dict["dev_index_in_train_list"] = self.dataset_results.result_dev_useqa["dev_index_in_train_list"][split_indices]

            dev_splits.append([bm25_dev_split_dict, useqa_dev_split_dict])

        return dev_splits

    def run_one_split(self, bm25_dev_split, useqa_dev_split):
        final_result_dict = {}

        # linear_sum = LinearSum(bm25_dev_split, useqa_dev_split,
        #                        self.dataset)
        # _ = linear_sum.reranking_by_linear_combination(bm25_dev_split, useqa_dev_split)
        # linear_sum_mrr, linear_sum_time = linear_sum.reranking_by_linear_combination(
        #     self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)
        #

        # do hyrbid logistic regression with warm ups
        lrClassifier = LogisticRegressionRouter(bm25_dev_split, useqa_dev_split, self.dataset,self.feature_pos)
        _ = lrClassifier.reranking_by_lr_router(bm25_dev_split, useqa_dev_split)
        hybrid_lr_mrr, hybrid_lr_neural_used, hybrid_lr_time = lrClassifier.reranking_by_lr_router(
            self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        final_result_dict = {
            "hybrid_lr": {"mrr": hybrid_lr_mrr, "router_output": hybrid_lr_neural_used, "time": hybrid_lr_time},
            "dev_index_in_train_list":useqa_dev_split["dev_index_in_train_list"]
        }

        return final_result_dict

    def run_all_splits(self):

        dev_splits = self._split_dev_data()

        all_results = []
        unsupervised_sum = []
        hybrid_threshold = []
        linear_sum = []
        hybrid_lr = []
        for i, dev_split in enumerate(dev_splits):
            final_result_dict = self.run_one_split(dev_split[0],dev_split[1])
            all_results.append(final_result_dict)

            hybrid_lr.append(final_result_dict["hybrid_lr"]["mrr"])

        # with open(generated_data_path+"hybrid_classifier_result/squad_hybrid_result.pickle", "wb") as handle:
        #     pickle.dump(all_results, handle)

        return np.mean(np.concatenate(hybrid_lr))

class ExperimentNQ():
    def __init__(self, feature_pos):
        self.dataset = "nq"
        self.dataset_results = LoadRawData(self.dataset)
        self.feature_pos = feature_pos

    def _split_dev_data(self):
        dev_splits = []
        random.seed(0)
        all_indices = np.arange(len(self.dataset_results.result_test_bm25["mrr"]))   # all indices to nq test
        all_dev_indicies_pool = np.array(random.sample(range(len(self.dataset_results.result_test_bm25["mrr"])), 50000))
        for split in range(5):
            split_dev_indices = all_dev_indicies_pool[split*10000: (split+1)*10000]
            split_test_indices = np.delete(all_indices, split_dev_indices)

            # construct dev set
            bm25_dev_split_dict = {}
            bm25_dev_split_dict["mrr"] = self.dataset_results.result_test_bm25["mrr"][split_dev_indices]
            bm25_dev_split_dict["top_scores"] = self.dataset_results.result_test_bm25["top_scores"][split_dev_indices]
            bm25_dev_split_dict["top_facts"] = self.dataset_results.result_test_bm25["top_facts"][split_dev_indices]

            useqa_dev_split_dict = {}
            useqa_dev_split_dict["gold_fact_index"] = self.dataset_results.result_test_useqa["gold_fact_index"][
                split_dev_indices]
            useqa_dev_split_dict["mrr"] = self.dataset_results.result_test_useqa["mrr"][split_dev_indices]
            useqa_dev_split_dict["top_scores"] = self.dataset_results.result_test_useqa["top_scores"][split_dev_indices]
            useqa_dev_split_dict["top_facts"] = self.dataset_results.result_test_useqa["top_facts"][split_dev_indices]
            useqa_dev_split_dict["dev_index_in_all_list"] = split_dev_indices

            # construct test set
            bm25_test_split_dict = {}
            bm25_test_split_dict["mrr"] = self.dataset_results.result_test_bm25["mrr"][split_test_indices]
            bm25_test_split_dict["top_scores"] = self.dataset_results.result_test_bm25["top_scores"][split_test_indices]
            bm25_test_split_dict["top_facts"] = self.dataset_results.result_test_bm25["top_facts"][split_test_indices]

            useqa_test_split_dict = {}
            useqa_test_split_dict["gold_fact_index"] = self.dataset_results.result_test_useqa["gold_fact_index"][
                split_test_indices]
            useqa_test_split_dict["mrr"] = self.dataset_results.result_test_useqa["mrr"][split_test_indices]
            useqa_test_split_dict["top_scores"] = self.dataset_results.result_test_useqa["top_scores"][split_test_indices]
            useqa_test_split_dict["top_facts"] = self.dataset_results.result_test_useqa["top_facts"][split_test_indices]
            useqa_test_split_dict["test_index_in_all_list"] = split_test_indices

            dev_splits.append([bm25_dev_split_dict, useqa_dev_split_dict, bm25_test_split_dict, useqa_test_split_dict])

        return dev_splits

    def run_one_split(self, bm25_dev_split, useqa_dev_split, bm25_test_split, useqa_test_split):
        final_result_dict = {}

        # linear_sum = LinearSum(bm25_dev_split, useqa_dev_split,self.dataset)
        # _ = linear_sum.reranking_by_linear_combination(bm25_dev_split, useqa_dev_split,)
        # linear_sum_mrr, linear_sum_time = linear_sum.reranking_by_linear_combination(bm25_test_split, useqa_test_split)


        # do hyrbid logistic regression with warm ups
        lrClassifier = LogisticRegressionRouter(bm25_dev_split, useqa_dev_split, self.dataset,self.feature_pos)
        _ = lrClassifier.reranking_by_lr_router(bm25_dev_split, useqa_dev_split)
        hybrid_lr_mrr, hybrid_lr_neural_used, hybrid_lr_time = lrClassifier.reranking_by_lr_router(
            bm25_test_split, useqa_test_split)

        final_result_dict = {
            "hybrid_lr": {"mrr": hybrid_lr_mrr, "router_output": hybrid_lr_neural_used, "time": hybrid_lr_time},
            "dev_index_in_all_list":useqa_dev_split["dev_index_in_all_list"],
            "test_index_in_all_list": useqa_test_split["test_index_in_all_list"],
        }

        return final_result_dict

    def run_all_splits(self):

        dev_splits = self._split_dev_data()
        # remove warm up by running on the dev set.

        all_results = []
        unsupervised_sum = []
        hybrid_threshold = []
        hybrid_lr = []
        for i, dev_split in enumerate(dev_splits):
            final_result_dict = self.run_one_split(dev_split[0],dev_split[1], dev_split[2],dev_split[3])
            all_results.append(final_result_dict)


            hybrid_lr.append(final_result_dict["hybrid_lr"]["mrr"])

        # TODO: change this back after experiment.
        # with open(generated_data_path+"hybrid_classifier_result/nq_hybrid_result.pickle", "wb") as handle:
        #     pickle.dump(all_results, handle)

        return np.mean(np.concatenate(hybrid_lr))

def top_bm25_1_2_4_8_16_32_64():
    feature_pos = np.arange(7)
    #feature_pos = np.arange(7,14,1)

    experimentOpenbook = ExperimentOpenbook(feature_pos)
    experimentOpenbook.run_all_exp()

    experimentSquad = ExperimentSquad(feature_pos)
    experimentSquad.run_all_splits()

    experimentNQ = ExperimentNQ(feature_pos)
    experimentNQ.run_all_splits()


def generate_table_result(feature_pos, experiment_name):

    experimentOpenbook = ExperimentOpenbook(feature_pos)
    ob_mrr = experimentOpenbook.run_all_exp()

    experimentSquad = ExperimentSquad(feature_pos)
    squad_mrr = experimentSquad.run_all_splits()

    experimentNQ = ExperimentNQ(feature_pos)
    nq_mrr = experimentNQ.run_all_splits()

    print("="*20)
    print("\t"+experiment_name)
    print("\topenbook mrr", ob_mrr)
    print("\tsquad mrr:", squad_mrr)
    print("\tnq mrr:", nq_mrr)

# generate_table_result(np.array([0,1,2,3,4,5,6]), "bm25 1 2 4 8 16 32 64")
# generate_table_result(np.array([2]), "bm25 4")
# generate_table_result(np.array([4]), "bm25 16")
# generate_table_result(np.array([6]), "bm25 64")

# generate_table_result(np.array([0,1,2,3,4,5,6])+7, "useqa 1 2 4 8 16 32 64")
# generate_table_result(np.array([2])+7, "useqa 4")
# generate_table_result(np.array([4])+7, "useqa 16")
# generate_table_result(np.array([6])+7, "useqa 64")

# generate_table_result(np.arange(14), "bm25 useqa 1 2 4 8 16 32 64")
# generate_table_result(np.array([2, 9]), "bm25 useqa 4")
# generate_table_result(np.array([4, 11]), "bm25 useqa 16")
# generate_table_result(np.array([6, 13]), "bm25 useqa 64")

# generate_table_result(np.array([0]), "bm25 1")
# generate_table_result(np.array([7]), "useqa 1")

#generate_table_result(np.array([0, 7]), "useqa 1")



