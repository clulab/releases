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
import random

from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def debug_dataprocessor(dataset = "openbook"):
    datasetResults = LoadRawData(dataset)

    if dataset=="nq":
        print("basic statistics:")
        print("\tbm25 top scores shape:",datasetResults.result_test_bm25["top_scores"].shape)
        print("\tbm25 top facts shape:", datasetResults.result_test_bm25["top_facts"].shape)
        print("\tuseqa top scores shape:", datasetResults.result_test_useqa["top_scores"].shape)
        print("\tuseqa top facts shape:", datasetResults.result_test_useqa["top_facts"].shape)

        print("bm25 test mrr:", np.mean(datasetResults.result_test_bm25["mrr"]))
        print("useqa test mrr:", np.mean(datasetResults.result_test_useqa["mrr"]))
    else:
        print("basic statistics:")
        print("\tbm25 top scores shape:", datasetResults.result_dev_bm25["top_scores"].shape)
        print("\tbm25 top facts shape:", datasetResults.result_dev_bm25["top_facts"].shape)
        print("\tuseqa top scores shape:", datasetResults.result_dev_useqa["top_scores"].shape)
        print("\tuseqa top facts shape:", datasetResults.result_dev_useqa["top_facts"].shape)

        print("bm25 dev mrr:", np.mean(datasetResults.result_dev_bm25["mrr"]))
        print("useqa dev mrr:", np.mean(datasetResults.result_dev_useqa["mrr"]))
        print("bm25 test mrr:", np.mean(datasetResults.result_test_bm25["mrr"]))
        print("useqa test mrr:", np.mean(datasetResults.result_test_useqa["mrr"]))

    idx_bm25_better = datasetResults.result_test_bm25["mrr"]>=datasetResults.result_test_useqa["mrr"]
    idx_useqa_better = datasetResults.result_test_bm25["mrr"]<datasetResults.result_test_useqa["mrr"]

    ceiling_mrr = datasetResults.result_test_bm25["mrr"][idx_bm25_better].tolist()+datasetResults.result_test_useqa["mrr"][idx_useqa_better].tolist()

    print(dataset+" ceiling mrr:"+str(sum(ceiling_mrr)/len(ceiling_mrr)))
    # good news: at least on openbook, nq and squad, the ceiling mrr is much better than both bm25 and

class LoadRawData():
    def __init__(self, dataset, use_ann = False):
        if dataset=="openbook":
            self.result_dev_bm25 = self._bm25_tsv_scores_to_dict(bm25_folder+"output_scores/openbook/dev/", generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_dev_bm25_result_dict.pickle")
            self.result_test_bm25 = self._bm25_tsv_scores_to_dict(bm25_folder+"output_scores/openbook/test/", generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_test_bm25_result_dict.pickle")

            if use_ann:
                self.result_dev_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/openbook_dev_useqa_ann_result_dict.pickle")
                self.result_test_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/openbook_test_useqa_ann_result_dict.pickle")

            else:
                self.result_dev_useqa = self._load_useqa_result_dict(generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_dev_useqa_result_dict.pickle")
                self.result_test_useqa = self._load_useqa_result_dict(generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_test_useqa_result_dict.pickle")

        if dataset=="squad":
            self.result_dev_bm25 = self._bm25_tsv_scores_to_dict(bm25_folder + "output_scores/squad/dev/",
                generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_dev_bm25_result_dict.pickle")

            self.result_test_bm25 = self._bm25_tsv_scores_to_dict(bm25_folder + "output_scores/squad/test/",
                generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_test_bm25_result_dict.pickle")

            if use_ann :
                self.result_dev_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_dev_useqa_ann_result_dict.pickle")

                self.result_test_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_test_useqa_ann_result_dict.pickle")

            else:
                self.result_dev_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_dev_useqa_result_dict.pickle")

                self.result_test_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_test_useqa_result_dict.pickle")

        if dataset=="nq":
            self.result_test_bm25 = self._bm25_tsv_scores_to_dict(bm25_folder + "output_scores/nq/test/",
                generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/nq_test_bm25_result_dict.pickle")

            if use_ann:
                self.result_test_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/nq_test_useqa_ann_result_dict.pickle")

            else:
                self.result_test_useqa = self._load_useqa_result_dict(
                    generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/nq_test_useqa_result_dict.pickle")

    def _bm25_tsv_scores_to_dict(self, folder_name, pickle_filename):
        # read the tsv files into result dicts. The format is the same as what is generated in
        if os.path.exists(pickle_filename):
            print(" loading bm25 scores from pickle ...")

            with open(pickle_filename, "rb") as handle:
                bm25_results_dict = pickle.load(handle)

        else:
            print(" loading bm25 scores from tsv ...")

            all_files_list = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]

            bm25_results_dict = {"mrr":[], "top_facts":[], "top_scores":[], "search_time":[], "write_time":[], "total_time":[]}

            start_time = time.time()
            for query_idx in range(len(all_files_list)):
                if (query_idx+1)%5000==0:
                    print("\tprocessing query "+str(query_idx+1)+" ... time:"+str(time.time()-start_time))
                    start_time = time.time()

                with open(folder_name+"query_" + str(query_idx) + "_scores.tsv", "r") as handle:
                    score_file = list(csv.reader(handle, delimiter="\t"))

                bm25_results_dict["mrr"].append(float(score_file[-1][0]))
                bm25_results_dict["search_time"].append(float(score_file[-1][1]))
                bm25_results_dict["write_time"].append(float(score_file[-1][2]))
                bm25_results_dict["total_time"].append(float(score_file[-1][3]))

                temp_array = np.array(score_file[:-3])
                bm25_results_dict["top_facts"].append(temp_array[:,0].astype(np.int32))
                bm25_results_dict["top_scores"].append(temp_array[:,1].astype(np.float32))


            bm25_results_dict["mrr"] = np.array(bm25_results_dict["mrr"]).astype(np.float32)
            bm25_results_dict["search_time"] = np.array(bm25_results_dict["search_time"]).astype(np.float32)
            bm25_results_dict["write_time"] = np.array(bm25_results_dict["write_time"]).astype(np.float32)
            bm25_results_dict["total_time"] = np.array(bm25_results_dict["total_time"]).astype(np.float32)

            bm25_results_dict["top_facts"] = np.array(bm25_results_dict["top_facts"])  # numpy allow element arrays to have different lengths
            bm25_results_dict["top_scores"] = np.array(bm25_results_dict["top_scores"])

            with open(pickle_filename, "wb") as handle:
                pickle.dump(bm25_results_dict, handle)

        return bm25_results_dict

    def _load_useqa_result_dict(self, filename):
        print(" loading useqa results from pickle ...")
        with open(filename, "rb") as handle:
            useqa_results_dict = pickle.load(handle)

        return useqa_results_dict

class ResultChecker():
    def __init__(self):
        self.result_pickle_names = {
            "openbook_dev":{"bm25": generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_dev_bm25_result_dict.pickle",
                            "useqa":generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_dev_useqa_result_dict.pickle"},
            "openbook_test":{"bm25":generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_test_bm25_result_dict.pickle",
                            "useqa":generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_test_useqa_result_dict.pickle"},
            "squad_dev":{"bm25": generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_dev_bm25_result_dict.pickle",
                         "useqa":generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_dev_useqa_result_dict.pickle"},
            "squad_test":{"bm25":generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_test_bm25_result_dict.pickle",
                         "useqa":generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/squad_test_useqa_result_dict.pickle"},
            "nq_test": {"bm25":generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/nq_test_bm25_result_dict.pickle",
                        "useqa":generated_data_path + "useQA_results_mrr_time_openbook_squad_nq/nq_test_useqa_result_dict.pickle"},
        }


        self.raw_data_pickle_names = {
            "openbook": generated_data_path+"/openbook_useqa/openbook_useqa_retrieval_data.pickle",
            "squad": generated_data_path+"/squad_useqa/squad_retrieval_data.pickle",
            "nq": generated_data_path+"/nq_retrieval_raw/nq_retrieval_data.pickle",
        }

    def check_mrrs(self):
        # This check is passed.

        for dataset in self.result_pickle_names.keys():
            print("="*20)
            print("checking dataset "+str(dataset))
            bm25_filename = self.result_pickle_names[dataset]["bm25"]
            useqa_filename = self.result_pickle_names[dataset]["useqa"]

            with open(bm25_filename,"rb") as handle:
                bm25_result_dict= pickle.load(handle)

            with open(useqa_filename,"rb") as handle:
                useqa_result_dict= pickle.load(handle)

            bm25_mrr_recon = []
            for i, gold_fact in enumerate(useqa_result_dict["gold_fact_index"]):
                if gold_fact in bm25_result_dict["top_facts"][i]:
                    bm25_mrr_recon.append(1/(1+bm25_result_dict["top_facts"][i].tolist().index(gold_fact)))
                else:
                    bm25_mrr_recon.append(0)

            useqa_mrr_recon = []
            for i, gold_fact in enumerate(useqa_result_dict["gold_fact_index"]):
                if gold_fact in useqa_result_dict["top_facts"][i]:
                    useqa_mrr_recon.append(1 / (1 + useqa_result_dict["top_facts"][i].tolist().index(gold_fact)))
                else:
                    useqa_mrr_recon.append(0)

            print("bm25 mrr:", np.mean(bm25_mrr_recon), " bm25 mrr recon:", sum(bm25_mrr_recon)/len(bm25_mrr_recon))
            print("useqa mrr:", np.mean(useqa_mrr_recon), " useqa mrr recon:", sum(useqa_mrr_recon) / len(useqa_mrr_recon))

    def check_retrieved_facts(self, dataset):

        if dataset=="openbook":
            self.openbook_check_retrieved_facts_maually("dev", "bm25")
            self.openbook_check_retrieved_facts_maually("dev", "useqa")
            self.openbook_check_retrieved_facts_maually("test", "bm25")
            self.openbook_check_retrieved_facts_maually("test", "useqa")
        if dataset=="squad":
            self.squad_check_retrieved_facts_maually("bm25")
            self.squad_check_retrieved_facts_maually("useqa")
        if dataset == "nq":
            self.nq_check_retrieved_facts_maually("bm25")
            self.nq_check_retrieved_facts_maually("useqa")

    def openbook_check_retrieved_facts_maually(self, partition, model_name):
        with open(self.raw_data_pickle_names["openbook"] , "rb") as handle:
            raw_data = pickle.load(handle)

        with open(self.result_pickle_names["openbook_"+partition][model_name] , "rb") as handle:
            result_dict = pickle.load(handle)

        instances_list = raw_data[partition+"_list"]
        kb = raw_data["kb"]

        indices_to_check = random.sample(range(len(instances_list)), 10)

        print(raw_data.keys())
        print(instances_list[0])

        print("="*20)
        print("check openbook "+partition+" "+model_name)
        for idx in indices_to_check:
            print("\t"+"-"*20)
            print("\tidx:"+str(idx)+" "+instances_list[idx]["text"])
            if model_name=="useqa":
                print("\tgold label:"+ str(instances_list[idx]["label"])+" saved label:"+str(result_dict["gold_fact_index"][idx]))
            else:
                print("\tgold label:" + str(instances_list[idx]["label"]))
            print("\tgold fact:"+ kb[instances_list[idx]["label"]])
            print("\tmrr:"+str(result_dict["mrr"][idx]))
            print("\ttop facts:")
            for top_fact in result_dict["top_facts"][idx][:min(10, len(result_dict["top_facts"][idx]))]:
                print("\t\t"+kb[top_fact])
        input("=============")

    def squad_check_retrieved_facts_maually(self, model_name):
        with open(self.raw_data_pickle_names["squad"] , "rb") as handle:
            raw_data = pickle.load(handle)

        with open(self.result_pickle_names["squad_test"][model_name] , "rb") as handle:
            result_dict = pickle.load(handle)

        print(raw_data.keys())

        instances_list = raw_data["dev_list"]

        print(instances_list[0])


        sent_list = raw_data["sent_list"]
        resp_list = raw_data["resp_list"]
        indices_to_check = random.sample(range(len(instances_list)), 10)

        print("="*20)
        print("check squad test "+model_name)
        for idx in indices_to_check:
            print("\t"+"-"*20)
            print("\tidx:"+str(idx)+" "+instances_list[idx]["question"])
            if model_name=="useqa":
                print("\tgold label:"+ str(instances_list[idx]["response"])+" saved label:"+str(result_dict["gold_fact_index"][idx]))
            else:
                print("\tgold label:" + str(instances_list[idx]["response"]))
            print("\tgold fact:"+ sent_list[int(resp_list[instances_list[idx]["response"]][0])])
            print("\tmrr:"+str(result_dict["mrr"][idx]))
            print("\ttop facts:")
            for top_fact in result_dict["top_facts"][idx][:min(10, len(result_dict["top_facts"][idx]))]:
                print("\t\t"+sent_list[int(resp_list[top_fact][0])])
        input("=============")

    def nq_check_retrieved_facts_maually(self, model_name):
        with open(self.raw_data_pickle_names["nq"] , "rb") as handle:
            raw_data = pickle.load(handle)

        with open(self.result_pickle_names["nq_test"][model_name] , "rb") as handle:
            result_dict = pickle.load(handle)

        print(raw_data.keys())

        instances_list = raw_data["train_list"]

        print(instances_list[0])


        sent_list = raw_data["sent_list"]
        resp_list = raw_data["resp_list"]
        indices_to_check = random.sample(range(len(instances_list)), 10)

        print("="*20)
        print("check nq test "+model_name)
        for idx in indices_to_check:
            print("\t"+"-"*20)
            print("\tidx:"+str(idx)+" "+instances_list[idx]["question"])
            if model_name=="useqa":
                print("\tgold label:"+ str(instances_list[idx]["response"])+" saved label:"+str(result_dict["gold_fact_index"][idx]))
            else:
                print("\tgold label:" + str(instances_list[idx]["response"]))
            print("\tgold fact:"+ sent_list[int(resp_list[instances_list[idx]["response"]][0])])
            print("\tmrr:"+str(result_dict["mrr"][idx]))
            print("\ttop facts:")
            for top_fact in result_dict["top_facts"][idx][:min(10, len(result_dict["top_facts"][idx]))]:
                print("\t\t"+sent_list[int(resp_list[top_fact][0])])
        input("=============")

    def debug_check_bm25_score_distribution(self, dataset, partition):
        with open(self.result_pickle_names[dataset+"_"+partition]["bm25"] , "rb") as handle:
            result_dict = pickle.load(handle)

        all_scores = np.concatenate(result_dict["top_scores"])

        histo, _ = np.histogram(all_scores, bins = np.arange(0,240,5))

        print("checking "+dataset+ " "+partition)
        print(histo)
        print("min score:", np.min(all_scores))
        print("max score:", np.max(all_scores))
        print("all scores:", np.sum(histo))

    def debug_check_bm25_score_distribution_get_figure(self, dataset, partition, normalization="none", n_features = 7, softmax_alpha = 1.0):

        def normalize_score(x):
            # This is to prevent overflow of large bm25 scores when using softmax.
            # MAX_FLOAT is 3.402823466 E + 38
            if np.max(x)>(85/softmax_alpha):   # this value is calculated by: log(MAX_FLOAT/64), where MAX_FLOAT is the maximum number of float64.
                x = x/np.max(x)*(85/softmax_alpha)
            return np.exp(x)/np.sum(np.exp(x))

        with open(self.result_pickle_names[dataset+"_"+partition]["bm25"] , "rb") as handle:
            result_dict_bm25 = pickle.load(handle)

        with open(self.result_pickle_names[dataset+"_"+partition]["useqa"] , "rb") as handle:
            result_dict_useqa = pickle.load(handle)

        avg_top64_scores = np.zeros((len(result_dict_bm25["top_scores"]), n_features))
        for i, top_score in enumerate(result_dict_bm25["top_scores"]):
            if len(top_score)<2**(n_features-1):
                top_score = np.concatenate([top_score, np.zeros(2**(n_features-1)-len(top_score))])
            else:
                top_score = top_score[:2**(n_features-1)]

            if normalization=="softmax":
                top_score = normalize_score(top_score)

            for j in range(n_features):
                avg_top64_scores[i,j] = np.mean(top_score[:2**j])

        if normalization=="softmax":
            bins=np.arange(0, 1.1, 0.1)
            x_range = np.arange(10)
        else:
            bins = np.arange(0, 105, 5)
            x_range = np.arange(20)

        plt.figure()
        for i in range(n_features):
            plt.subplot(2,4,i+1)
            plt.title("avg of top "+str(2**i)+" bm25 scores\n(norm:"+ normalization+")")
            scores_operating = avg_top64_scores[:,i]
            scores_bm25_better = scores_operating[result_dict_bm25["mrr"]>=result_dict_useqa["mrr"]]
            scores_useqa_better = scores_operating[result_dict_bm25["mrr"] < result_dict_useqa["mrr"]]

            histo_bm25_better ,_ = np.histogram(scores_bm25_better, bins)
            histo_useqa_better, _ = np.histogram(scores_useqa_better, bins)
            plt.bar(x_range-0.25, histo_bm25_better, 0.25, color = "r",  label = "bm25 better")
            plt.bar(x_range+0.25, histo_useqa_better, 0.25, color = "b", label = "useqa better")
            plt.xlabel("bm25 scores")
            plt.ylabel("number of samples")
            plt.legend()

        print("n bm25 better:", np.sum(result_dict_bm25["mrr"]>=result_dict_useqa["mrr"]))
        print("n useqa better:", np.sum(result_dict_bm25["mrr"] < result_dict_useqa["mrr"]))

        #plt.tight_layout()
        plt.show()



#checker = ResultChecker()

#checker.openbook_check_retrieved_facts_maually("dev","bm25")
#checker.check_retrieved_facts("nq")
#checker.debug_check_bm25_score_distribution_get_figure("squad", "dev", "softmax", 5, 2.0)