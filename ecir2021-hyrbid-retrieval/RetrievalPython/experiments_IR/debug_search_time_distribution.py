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
import math
import time
import matplotlib.pyplot as plt
import csv

def plot_useQA_time(dataset="squad", partition="test"):

    with open(dataset +"_" +partition +"_useqa_result_dict.pickle" ,"rb") as handle:
        result_dict_pickle = pickle.load(handle)

    plt.figure()
    plt.plot(result_dict_pickle["search_time"], color = "r")
    plt.plot(result_dict_pickle["sort_time"], color="b")
    plt.plot(result_dict_pickle["search_time"]+result_dict_pickle["sort_time"], color="g")
    plt.title("numpy: "+dataset+ " "+ partition)

    return 0

def plot_bm25_time(dataset, partition):
    n_ques_dict = {
        "openbook":{"dev":500, "test":500},
        "squad":{"dev":10000, "test":11426},
        "nq":{"test":74097}
    }

    search_time = []
    write_time = []
    total_time = []
    for query_idx in range(n_ques_dict[dataset][partition]):
        with open(bm25_folder+"output_scores/"+dataset+"/"+str(partition)+"/query_"+str(query_idx)+"_scores.tsv","r") as handle:
            score_file = list(csv.reader(handle, delimiter="\t"))[-1] # get the last line

        search_time.append(float(score_file[1]))
        write_time.append(float(score_file[2]))
        total_time.append(float(score_file[3]))

    plt.figure()
    plt.plot(search_time, color="r")
    plt.plot(write_time, color="b")
    plt.plot(total_time, color="g")
    plt.title("bm25: "+dataset + " " + partition)


plot_useQA_time("openbook", "dev")
plot_useQA_time("openbook", "test")
# plot_useQA_time("squad", "dev")
# plot_useQA_time("squad", "test")
# plot_useQA_time("nq", "test")

# plot_bm25_time("openbook", "dev")
# plot_bm25_time("openbook", "test")
# plot_bm25_time("squad", "dev")
# plot_bm25_time("squad", "test")
# plot_bm25_time("nq", "test")
plt.show()