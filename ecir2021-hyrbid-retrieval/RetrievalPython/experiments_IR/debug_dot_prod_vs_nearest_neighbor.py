import sys
from pathlib import Path
import argparse
from simpleneighbors import SimpleNeighbors


parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
generated_data_path = parent_folder_path+"/data_generated/"
sys.path+=[parent_folder_path, datasets_folder_path, generated_data_path]

import numpy as np
import pickle
import math
import time

def squad_test_nearest_neighbor(top_n_facts = 2000, embd_dimension = 512):
    with open(generated_data_path + "squad_useqa/squad_retrieval_data.pickle", "rb") as handle:
        instances_list = pickle.load(handle)

    query_embds_test = np.load(generated_data_path + "squad_useqa/ques_dev_embds.npy")
    fact_embds = np.load(generated_data_path+"squad_useqa/sents_embds.npy")


    query_embds_test = query_embds_test.tolist()
    fact_embds = fact_embds.tolist()
    fact_embds = [(idx, fact_embd) for idx, fact_embd in enumerate(fact_embds)]


    sim = SimpleNeighbors(embd_dimension)  # This 3 is the dimension of data.
    # the tree should be how many clucsters we want to use.

    build_index_start  =time.time()
    sim.feed(fact_embds)  # each color vector has three int elements, ranging from 0 to 255.
    sim.build()  # here we can control the size of the tree, it controls the speed and accuracy of search.

    build_index_end = time.time()
    print("build index time:", build_index_end-build_index_start)

    search_time = []
    mrr = []
    for query_idx, query_embd in enumerate(query_embds_test):
        search_time_start = time.time()
        nearest_neighbor = list(sim.nearest(query_embd, top_n_facts))
        search_time_end = time.time()

        gold_label = instances_list["dev_list"][query_idx]["response"]

        sample_mrr = 1/(nearest_neighbor.index(gold_label)+1) if gold_label in nearest_neighbor else 0
        mrr.append(sample_mrr)
        search_time.append(search_time_end-search_time_start)

    print("mrr:", sum(mrr)/len(mrr))
    print("avg search time:", sum(search_time)/len(search_time))

squad_test_nearest_neighbor()
