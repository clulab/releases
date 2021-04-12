import numpy as np
import pickle
from debug_dataloader import LoadRawData
import matplotlib.pyplot as plt

def softmax(x):
    # This is to prevent overflow of large bm25 scores when using softmax.
    # MAX_FLOAT is 3.402823466 E + 38
    if np.max(x) > 85:  # this value is calculated by: log(MAX_FLOAT/64), where MAX_FLOAT is the maximum number of float64.
        x = x / np.max(x) * 85
    return np.exp(x) / np.sum(np.exp(x))

def plot_histogram():
    openbook_result = LoadRawData("openbook")
    squad_result = LoadRawData("squad")

    width =0.35

    plt.figure()
    plt.subplot(1,2,1)
    # first get the dev result on openbook.
    n_bm25_better_openbook = openbook_result.result_dev_bm25["mrr"]>=openbook_result.result_dev_useqa["mrr"]
    n_useqa_better_openbook = openbook_result.result_dev_bm25["mrr"]<openbook_result.result_dev_useqa["mrr"]

    top_scores_bm25_better_openbook = [softmax(top_score)[0] for top_score in openbook_result.result_dev_bm25["top_scores"][n_bm25_better_openbook]]
    top_scores_useqa_better_openbook = [softmax(top_score)[0] for top_score in openbook_result.result_dev_bm25["top_scores"][n_useqa_better_openbook]]
    histo_bm25_better_openbook, bins = np.histogram(top_scores_bm25_better_openbook, np.arange(0,1.2,0.2))
    histo_useqa_better_openbook,_ = np.histogram(top_scores_useqa_better_openbook, np.arange(0,1.2,0.2))

    plt.bar(np.arange(len(bins)-1)+width/2, histo_bm25_better_openbook, width, label = "BM25 no worse than USE-QA")
    plt.bar(np.arange(len(bins)-1)+width*1.5, histo_useqa_better_openbook, width, label = "USE-QA better")

    plt.ylabel("number of queries")
    plt.xlabel("top BM25 scores")
    plt.xticks(np.arange(len(bins)),["0","0.2", "0.4","0.6", "0.8", "1.0"])

    plt.subplot(1, 2, 2)
    #  get the dev result on squad.
    n_bm25_better_squad = squad_result.result_dev_bm25["mrr"] >= squad_result.result_dev_useqa["mrr"]
    n_useqa_better_squad = squad_result.result_dev_bm25["mrr"] < squad_result.result_dev_useqa["mrr"]

    top_scores_bm25_better_squad = [softmax(top_score)[0] for top_score in
                                    squad_result.result_dev_bm25["top_scores"][n_bm25_better_squad]]
    top_scores_useqa_better_squad = [softmax(top_score)[0] for top_score in
                                     squad_result.result_dev_bm25["top_scores"][n_useqa_better_squad]]
    histo_bm25_better_squad, bins = np.histogram(top_scores_bm25_better_squad, np.arange(0, 1.2, 0.2))
    histo_useqa_better_squad, _ = np.histogram(top_scores_useqa_better_squad, np.arange(0, 1.2, 0.2))

    plt.bar(np.arange(len(bins) - 1) + width / 2, histo_bm25_better_squad, width, label="BM25 no worse than USE-QA")
    plt.bar(np.arange(len(bins) - 1) + width * 1.5, histo_useqa_better_squad, width, label="USE-QA better")

    plt.legend()
    plt.ylabel("number of queries")
    plt.xlabel("top BM25 scores")
    plt.xticks(np.arange(len(bins)), ["0","0.2", "0.4","0.6", "0.8", "1.0"])

    print("n bm25 better openbook:", np.sum(n_bm25_better_openbook), " total:", len(n_bm25_better_openbook))
    print("n bm25 better squad:", np.sum(n_bm25_better_squad), " total:", len(n_bm25_better_squad))


    plt.legend()
    plt.tight_layout()

    plt.show()

plot_histogram()