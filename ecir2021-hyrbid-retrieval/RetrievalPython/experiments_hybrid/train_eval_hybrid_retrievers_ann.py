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

class UnsupervisedSum():
    def __init__(self, result_dict_bm25_dev, result_dict_useqa_dev, dataset):
        self.dataset = dataset

    def reranking_with_sum_score(self, result_dict_bm25_test,result_dict_useqa_test):
        '''
        First loop over the numpy zero element, then ranking only the non zero elements.
        :param result_dict_bm25_test:
        :param result_dict_useqa_test:
        :return:
        '''
        mrr_list = []
        time_list = []

        # this uses softmax over all retrieved top 2000 facts. Maybe we should also try that on hyrbid lr model.
        for sample_idx in range(len(result_dict_bm25_test["mrr"])):
            start_time = time.time()
            gold_label = result_dict_useqa_test["gold_fact_index"][sample_idx]

            bm25_score_recon = np.zeros(dataset_statistics[self.dataset]["n_kb"])  # get how many facts in total
            bm25_score_recon[result_dict_bm25_test["top_facts"][sample_idx]] = softmax(result_dict_bm25_test["top_scores"][sample_idx])

            useqa_score_recon =np.zeros(dataset_statistics[self.dataset]["n_kb"])
            useqa_score_recon[result_dict_useqa_test["top_facts"][sample_idx]] = softmax(result_dict_useqa_test["top_scores"][sample_idx])

            rel_doc_bool_idx_bm25 = np.zeros(dataset_statistics[self.dataset]["n_kb"]).astype(bool)
            rel_doc_bool_idx_bm25[result_dict_bm25_test["top_facts"][sample_idx]] = True

            rel_doc_bool_idx_useqa = np.zeros(dataset_statistics[self.dataset]["n_kb"]).astype(bool)
            rel_doc_bool_idx_useqa[result_dict_useqa_test["top_facts"][sample_idx]] = True

            rel_doc_bool_idx_combined = rel_doc_bool_idx_bm25+rel_doc_bool_idx_useqa

            combined_scores = bm25_score_recon+useqa_score_recon  # This is still the full size.

            combined_scores = combined_scores[rel_doc_bool_idx_combined]
            rel_doc_idx_combined = np.arange(dataset_statistics[self.dataset]["n_kb"])[rel_doc_bool_idx_combined]

            sorted_facts = rel_doc_idx_combined[np.flip(np.argsort(combined_scores))].tolist()  # sort in the order of descending score
            end_time = time.time()
            time_list.append(end_time-start_time)

            mrr = 1/(1+sorted_facts.index(gold_label)) if gold_label in sorted_facts else 0
            mrr_list.append(mrr)

        print("="*20)
        print("mrr bm25:", np.mean(result_dict_bm25_test["mrr"]))
        print("mrr useqa:", np.mean(result_dict_useqa_test["mrr"]))
        print("mrr unsupervised sum socre:", sum(mrr_list)/len(mrr_list))
        print("average sum and reranking time:", sum(time_list)/len(time_list))

        return np.array(mrr_list), np.array(time_list)

    def reranking_with_sum_score_deprecated(self, result_dict_bm25_test, result_dict_useqa_test):
        '''
        Ranking the full list.
        :param result_dict_bm25_test:
        :param result_dict_useqa_test:
        :return:
        '''
        mrr_list = []
        time_list = []

        start_time = time.time()
        for sample_idx in range(len(result_dict_bm25_test["mrr"])):
            gold_label = result_dict_useqa_test["gold_fact_index"][sample_idx]

            bm25_score_recon = np.zeros(dataset_statistics[self.dataset]["n_kb"])  # get how many facts in total
            bm25_score_recon[result_dict_bm25_test["top_facts"][sample_idx]] = softmax(result_dict_bm25_test["top_scores"][
                                                                                   sample_idx] )

            useqa_score_recon = np.zeros(dataset_statistics[self.dataset]["n_kb"])
            useqa_score_recon[result_dict_useqa_test["top_facts"][sample_idx]] = softmax(result_dict_useqa_test["top_scores"][
                                                                                     sample_idx] )

            combined_scores = bm25_score_recon + useqa_score_recon

            sorted_facts = np.flip(np.argsort(combined_scores)).tolist()  # sort in the order of descending score

            mrr = 1 / (1 + sorted_facts.index(gold_label)) if gold_label in sorted_facts else 0
            mrr_list.append(mrr)

        print("=" * 20)
        print("unsupervised sum ranker")
        print("time:", time.time()-start_time)
        print("mrr bm25:", np.mean(result_dict_bm25_test["mrr"]))
        print("mrr useqa:", np.mean(result_dict_useqa_test["mrr"]))
        print("mrr unsupervised sum socre:", sum(mrr_list) / len(mrr_list))

    def _get_max_score_bm25(self, result_dict):
        max_score = 0
        for top_scores in result_dict["top_scores"]:
            if np.max(top_scores)>max_score:
                max_score = np.max(top_scores)

        return max_score

    def _get_max_score_useqa(self, result_dict):
        return np.max(result_dict["top_scores"])

class UnsupervisedThreshold():
    def __init__(self, result_dict_bm25_dev,result_dict_useqa_dev,  dataset):
        self.max_dev_score_bm25 = self._get_max_score_bm25(result_dict_bm25_dev)  # max scores must be obtained on dev.
        self.dataset = dataset

        self.best_threshold = self._find_best_threshold_on_dev(result_dict_bm25_dev,result_dict_useqa_dev)

    def _get_max_score_bm25(self, result_dict):
        max_score = 0
        for top_scores in result_dict["top_scores"]:
            if np.max(top_scores)>max_score:
                max_score = np.max(top_scores)

        return max_score

    def _find_best_threshold_on_dev(self, bm25_result_dict, useqa_result_dict):

        best_mrr = 0
        best_threshold = 0

        for threshold in np.arange(0.1, 1.1, 0.1):
            mrr_array, _, _ = self._compute_mrr_given_threshold(bm25_result_dict, useqa_result_dict, threshold)
            if np.mean(mrr_array)>best_mrr:
                best_mrr = np.mean(mrr_array)
                best_threshold = threshold

        return best_threshold

    def _compute_mrr_given_threshold(self, bm25_result_dict, useqa_result_dict, threshold):


        # this also uses the softmax over all of the top 2000 facts.
        mrr_list = []
        neural_used = []
        time_list = []
        for query_idx in range(len(bm25_result_dict["mrr"])):
            start_time = time.time()
            if softmax(bm25_result_dict["top_scores"][query_idx])[0]>=threshold:
                mrr_list.append(bm25_result_dict["mrr"][query_idx])
                neural_used.append(0)
            else:
                mrr_list.append(useqa_result_dict["mrr"][query_idx])
                neural_used.append(1)
            end_time = time.time()
            time_list.append(end_time-start_time)

        return np.array(mrr_list), np.array(neural_used), np.array(time_list)

    def reranking_by_threshold(self, bm25_test_result_dict, useqa_test_result_dict):

        hybrid_mrr, neural_used, time_used = self._compute_mrr_given_threshold(bm25_test_result_dict,useqa_test_result_dict, self.best_threshold)

        print("="*20)
        print("threshold hybrid classifier")
        print("best threshold:", self.best_threshold)
        print("hybrid mrr:", np.mean(hybrid_mrr))
        print("n neural used:", np.sum(neural_used))
        print("average time:", np.mean(time_used))
        print("=" * 20)
        return hybrid_mrr, neural_used, time_used

class LogisticRegressionRouter():
    def __init__(self, result_dict_bm25_dev, result_dict_useqa_dev, dataset, n_feature):
        self.max_dev_score_bm25 = self._get_max_score_bm25(result_dict_bm25_dev)  # max scores must be obtained on dev.
        self.dataset = dataset
        self.n_feature = n_feature

        self.train_array, self.labels = self._build_data_for_LR( result_dict_bm25_dev, result_dict_useqa_dev)

        self.reg = LogisticRegression(solver= 'lbfgs')
        self.reg.fit(self.train_array, self.labels)

        print(self.reg.coef_)
        print(self.reg.intercept_)
        # for i in range(len(train_labels_pred)):
        #     print("="*20)
        #     print(self.train_array[i,:])
        #     print("label:", self.labels[i], " pred:", train_labels_pred[i])
        #
        #     input("A")

    def reranking_by_lr_router(self, bm25_result_dict, useqa_result_dict):
        n_top_facts = 2**(self.n_feature-1)
        mrr_list = []
        neural_used = []
        time_list = []
        for query_idx, top_scores in enumerate(bm25_result_dict["top_scores"]):
            start_time = time.time()
            input_array = np.zeros((1,self.n_feature))

            # Be careful not to modify the input data.
            top_scores_ = top_scores
            if len(top_scores)<(n_top_facts):
                top_scores_ = np.concatenate([top_scores_, np.zeros(n_top_facts-len(top_scores_))])
            else:
                top_scores_ = top_scores_[:n_top_facts]
            top_scores_ = softmax(top_scores_)


            for i in range(self.n_feature):
                input_array[0, i] = np.mean(top_scores_[:2 ** i])

            prediction = self.reg.predict(input_array)
            end_time = time.time()
            time_list.append(end_time-start_time)

            # print("=====================")
            # print(input_array)
            # print(prediction)
            # input("A")

            if prediction>0.5:
                mrr_list.append(useqa_result_dict["mrr"][query_idx])
                neural_used.append(1)
            else:
                mrr_list.append(bm25_result_dict["mrr"][query_idx])
                neural_used.append(0)

        print("="*20)
        print("run logistic regression router")
        print("hybrid mrr:", sum(mrr_list)/len(mrr_list))
        print("n neural used:", sum(neural_used))
        print("average time:", sum(time_list)/len(time_list))
        print("="*20)

        return np.array(mrr_list), np.array(neural_used), np.array(time_list)

    def _get_max_score_bm25(self, result_dict):
        max_score = 0
        for top_scores in result_dict["top_scores"]:
            if np.max(top_scores)>max_score:
                max_score = np.max(top_scores)

        return max_score

    def _build_data_for_LR(self, bm25_result_dict, useqa_result_dict):

        # first normalize the top 2000 BM25 scores with softmax.
        top_scores_list_copy = []
        for query_idx, top_scores in enumerate(bm25_result_dict["top_scores"]):
            top_scores_list_copy.append(top_scores)

        # be careful not to modify the original data.
        cleaned_array = []
        for query_idx in range(len(bm25_result_dict["mrr"])):
            if len(top_scores_list_copy[query_idx])< 2**(self.n_feature-1):
                score_len = len(top_scores_list_copy[query_idx])
                cleaned_array.append(np.concatenate([top_scores_list_copy[query_idx], np.zeros(2**(self.n_feature-1)-score_len)]).reshape(2**(self.n_feature-1)))
            else:
                cleaned_array.append(top_scores_list_copy[query_idx][:2**(self.n_feature-1)])

            cleaned_array[query_idx] = softmax(cleaned_array[query_idx])

        cleaned_array = np.array(cleaned_array)

        train_array = np.zeros((len(bm25_result_dict["mrr"]), self.n_feature))
        for i in range(self.n_feature):
            train_array[:, i] = np.mean(cleaned_array[:, :2 ** i], axis=1)

        labels = np.zeros(len(bm25_result_dict["mrr"]))
        labels[useqa_result_dict["mrr"]>bm25_result_dict["mrr"]]=1

        return train_array, labels

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

    unsupervised_sum = UnsupervisedSum(dataset_results.result_dev_bm25, dataset_results.result_dev_useqa, dataset)
    unsupervised_sum.reranking_with_sum_score(dataset_results.result_test_bm25, dataset_results.result_test_useqa)

    thresholdClassifier = UnsupervisedThreshold(dataset_results.result_dev_bm25, dataset_results.result_dev_useqa, dataset)
    thresholdClassifier.reranking_by_threshold(dataset_results.result_test_bm25, dataset_results.result_test_useqa)

    lrClassifier = LogisticRegressionRouter(dataset_results.result_dev_bm25, dataset_results.result_dev_useqa, dataset, 7)
    lrClassifier.reranking_by_lr_router(dataset_results.result_test_bm25, dataset_results.result_test_useqa)


#debug_experiment("openbook")
#get_data_statistics()

class ExperimentOpenbook():
    def __init__(self):
        self.dataset = "openbook"
        self.dataset_results = LoadRawData(self.dataset, True)

    def run_all_exp(self):

        # linear_sum = LinearSum(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa,
        #                        self.dataset)
        # _ = linear_sum.reranking_by_linear_combination(self.dataset_results.result_dev_bm25,
        #                                                self.dataset_results.result_dev_useqa)
        # linear_sum_mrr, linear_sum_time = linear_sum.reranking_by_linear_combination(
        #     self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        # TODO: change this back after experiment
        # # do unsupervised sum with warm ups
        unsupervised_sum = UnsupervisedSum(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa, self.dataset)
        _ = unsupervised_sum.reranking_with_sum_score(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa)
        unsupervised_sum_mrr, unsupervised_sum_time = unsupervised_sum.reranking_with_sum_score(self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        # do hyrbid threshold with warm ups
        thresholdClassifier = UnsupervisedThreshold(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa,
                                                    self.dataset)
        _ = thresholdClassifier.reranking_by_threshold(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa)
        hybrid_threshold_mrr, hybrid_threshold_neural_used, hybrid_threshold_time = thresholdClassifier.reranking_by_threshold(self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        # do hyrbid logistic regression with warm ups
        lrClassifier = LogisticRegressionRouter(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa,
                                                self.dataset, 7)
        _ =lrClassifier.reranking_by_lr_router(self.dataset_results.result_dev_bm25, self.dataset_results.result_dev_useqa)
        hybrid_lr_mrr, hybrid_lr_neural_used, hybrid_lr_time =lrClassifier.reranking_by_lr_router(self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)


        print(np.mean(unsupervised_sum_mrr))
        print(np.mean(hybrid_threshold_mrr))
        print(np.mean(hybrid_lr_mrr))

        print(np.sum(hybrid_threshold_neural_used))
        print(np.sum(hybrid_lr_neural_used))
        #
        final_result_dict = {
            #"linear_sum":{"mrr":linear_sum_mrr, "time":linear_sum_time},
            "unsupervised_sum":{"mrr":unsupervised_sum_mrr, "time":unsupervised_sum_time},
            "hybrid_threshold":{"mrr": hybrid_threshold_mrr, "router_output":hybrid_threshold_neural_used, "time":hybrid_threshold_time},
            "hybrid_lr":{"mrr": hybrid_lr_mrr, "router_output":hybrid_lr_neural_used, "time":hybrid_lr_time}
        }

        # with open(generated_data_path+"hybrid_classifier_result/openbook_hybrid_result.pickle", "wb") as handle:
        #     pickle.dump(final_result_dict, handle)

class ExperimentSquad():
    def __init__(self):
        self.dataset = "squad"
        self.dataset_results = LoadRawData(self.dataset, True)

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


        # TODO: change this back after experiment
        # # do unsupervised sum with warm ups
        # unsupervised_sum = UnsupervisedSum(bm25_dev_split, useqa_dev_split, self.dataset)
        # _ = unsupervised_sum.reranking_with_sum_score(bm25_dev_split, useqa_dev_split)
        # unsupervised_sum_mrr, unsupervised_sum_time = unsupervised_sum.reranking_with_sum_score(
        #     self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        # do hyrbid threshold with warm ups
        thresholdClassifier = UnsupervisedThreshold(bm25_dev_split, useqa_dev_split,self.dataset)
        _ = thresholdClassifier.reranking_by_threshold(bm25_dev_split, useqa_dev_split)
        hybrid_threshold_mrr, hybrid_threshold_neural_used, hybrid_threshold_time = thresholdClassifier.reranking_by_threshold(
            self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        # do hyrbid logistic regression with warm ups
        lrClassifier = LogisticRegressionRouter(bm25_dev_split, useqa_dev_split, self.dataset,7)
        _ = lrClassifier.reranking_by_lr_router(bm25_dev_split, useqa_dev_split)
        hybrid_lr_mrr, hybrid_lr_neural_used, hybrid_lr_time = lrClassifier.reranking_by_lr_router(
            self.dataset_results.result_test_bm25, self.dataset_results.result_test_useqa)

        #print(np.mean(unsupervised_sum_mrr))
        print(np.mean(hybrid_threshold_mrr))
        print(np.mean(hybrid_lr_mrr))

        print(np.sum(hybrid_threshold_neural_used))
        print(np.sum(hybrid_lr_neural_used))
        #
        final_result_dict = {
            #"linear_sum": {"mrr": linear_sum_mrr, "time": linear_sum_time},
            "unsupervised_sum": {"mrr": 0, "time": 0},
            "hybrid_threshold": {"mrr": hybrid_threshold_mrr, "router_output": hybrid_threshold_neural_used,
                                  "time": hybrid_threshold_time},
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
            print("="*30)
            print("running split "+str(i))
            final_result_dict = self.run_one_split(dev_split[0],dev_split[1])
            all_results.append(final_result_dict)

            # TODO: change this back after experiment
            unsupervised_sum.append(final_result_dict["unsupervised_sum"]["mrr"])
            hybrid_threshold.append(final_result_dict["hybrid_threshold"]["mrr"])
            hybrid_lr.append(final_result_dict["hybrid_lr"]["mrr"])


        print(np.mean(np.array(unsupervised_sum)))
        print(np.mean(np.array(hybrid_threshold)))
        print(np.mean(np.array(hybrid_lr)))


        with open(generated_data_path+"hybrid_classifier_result/squad_hybrid_ann_result.pickle", "wb") as handle:
            pickle.dump(all_results, handle)

        return 0

class ExperimentNQ():
    def __init__(self):
        self.dataset = "nq"
        self.dataset_results = LoadRawData(self.dataset, True)

    def _split_dev_data(self):
        dev_splits = []
        print("start sampling dev for split test ...")
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
        print("splitting dataset finished! ")

        return dev_splits

    def run_one_split(self, bm25_dev_split, useqa_dev_split, bm25_test_split, useqa_test_split):
        final_result_dict = {}

        # linear_sum = LinearSum(bm25_dev_split, useqa_dev_split,self.dataset)
        # _ = linear_sum.reranking_by_linear_combination(bm25_dev_split, useqa_dev_split,)
        # linear_sum_mrr, linear_sum_time = linear_sum.reranking_by_linear_combination(bm25_test_split, useqa_test_split)

        # TODO: change this back after experiment

        # # do unsupervised sum with warm ups
        # unsupervised_sum = UnsupervisedSum(bm25_dev_split, useqa_dev_split, self.dataset)
        # _ = unsupervised_sum.reranking_with_sum_score(bm25_dev_split, useqa_dev_split)
        # unsupervised_sum_mrr, unsupervised_sum_time = unsupervised_sum.reranking_with_sum_score(
        #     bm25_test_split, useqa_test_split)
        #unsupervised_sum_mrr, unsupervised_sum_time = 0,0

        # do hyrbid threshold with warm ups
        thresholdClassifier = UnsupervisedThreshold(bm25_dev_split, useqa_dev_split, self.dataset)
        _ = thresholdClassifier.reranking_by_threshold(bm25_dev_split, useqa_dev_split)
        hybrid_threshold_mrr, hybrid_threshold_neural_used, hybrid_threshold_time = thresholdClassifier.reranking_by_threshold(
            bm25_test_split, useqa_test_split)

        # do hyrbid logistic regression with warm ups
        lrClassifier = LogisticRegressionRouter(bm25_dev_split, useqa_dev_split, self.dataset,7)
        _ = lrClassifier.reranking_by_lr_router(bm25_dev_split, useqa_dev_split)
        hybrid_lr_mrr, hybrid_lr_neural_used, hybrid_lr_time = lrClassifier.reranking_by_lr_router(
            bm25_test_split, useqa_test_split)

        #print(np.mean(unsupervised_sum_mrr))
        print(np.mean(hybrid_threshold_mrr))
        print(np.mean(hybrid_lr_mrr))

        print(np.sum(hybrid_threshold_neural_used))
        print(np.sum(hybrid_lr_neural_used))
        #
        final_result_dict = {
            #"linear_sum": {"mrr": linear_sum_mrr, "time": linear_sum_time},
            "unsupervised_sum": {"mrr": 0, "time": 0},
            "hybrid_threshold": {"mrr": hybrid_threshold_mrr, "router_output": hybrid_threshold_neural_used,
                                  "time": hybrid_threshold_time},
            "hybrid_lr": {"mrr": hybrid_lr_mrr, "router_output": hybrid_lr_neural_used, "time": hybrid_lr_time},
            "dev_index_in_all_list":useqa_dev_split["dev_index_in_all_list"],
            "test_index_in_all_list": useqa_test_split["test_index_in_all_list"],
        }

        return final_result_dict

    def run_all_splits(self):

        dev_splits = self._split_dev_data()
        # remove warm up by running on the dev set.
        print("start running nq experiments ..")

        all_results = []
        unsupervised_sum = []
        hybrid_threshold = []
        hybrid_lr = []
        for i, dev_split in enumerate(dev_splits):
            print("="*30)
            print("running split "+str(i))
            final_result_dict = self.run_one_split(dev_split[0],dev_split[1], dev_split[2],dev_split[3])
            all_results.append(final_result_dict)

            # TODO: change this back

            unsupervised_sum.append(final_result_dict["unsupervised_sum"]["mrr"])
            hybrid_threshold.append(final_result_dict["hybrid_threshold"]["mrr"])
            hybrid_lr.append(final_result_dict["hybrid_lr"]["mrr"])


        print(np.mean(np.array(unsupervised_sum)))
        print(np.mean(np.array(hybrid_threshold)))
        print(np.mean(np.array(hybrid_lr)))


        # TODO: change this back after experiment.
        # with open(generated_data_path+"hybrid_classifier_result/nq_hybrid_ann_result.pickle", "wb") as handle:
        #     pickle.dump(all_results, handle)

        return 0

#experimentOpenbook = ExperimentOpenbook()
#experimentOpenbook.run_all_exp()

experimentSquad = ExperimentSquad()
experimentSquad.run_all_splits()
#
experimentNQ = ExperimentNQ()
experimentNQ.run_all_splits()