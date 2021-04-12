import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
generated_data_path = parent_folder_path+"/data_generated/"
sys.path+=[parent_folder_path, datasets_folder_path, generated_data_path]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # used for compute cosine similarity for sparse matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import openbook_retrieval
import os
import pickle

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def eval_tfidf(instances_list, tfidf_vectorizer, doc_matrix, kb,  saved_file_name):


    correct_count = 0
    justification_hit_ratio = list([])
    mrr = 0
    list_to_save = list([])
    count_top10 = 0
    for i, instance in enumerate(instances_list):
        query = instance["query"]
        query_matrix = tfidf_vectorizer.transform(query)

        cosine_similarities = linear_kernel(query_matrix, doc_matrix).flatten()
        rankings = list(reversed(np.argsort(cosine_similarities).tolist()))  # rankings of facts, from most relevant

        mrr+=1/(1+rankings.index(instance["documents"][0]))

        list_to_save.append({"id":instance["id"], "mrr": 1/(1+rankings.index(instance["documents"][0])), "top_score":np.max(cosine_similarities)})


    return list_to_save

def load_bert_scores(file_path):
    with open(file_path, "rb") as handle:
        result_dict = pickle.load(handle)

    return result_dict["mrr"]

def performance_comparison(tfidf_result, bert_mrr, threshold):
    tfidf_mrr = [single_dict["mrr"] for single_dict in tfidf_result]

    hybrid_mrr = []
    for i, tfidf_dict in enumerate(tfidf_result):
        if tfidf_dict["top_score"]<threshold:
            hybrid_mrr.append(bert_mrr[i])
        else:
            hybrid_mrr.append(tfidf_mrr[i])

    print("="*20)
    print("threshold ", threshold)
    print("tfidf mrr:", sum(tfidf_mrr)/len(tfidf_mrr))
    print("bert mrr:", sum(bert_mrr)/len(bert_mrr))
    print("hybrid mrr:", sum(hybrid_mrr)/len(hybrid_mrr))

    return 0


def main():
    train_list, dev_list, test_list, sci_kb = openbook_retrieval.construct_retrieval_dataset_openbook(5, 1)

    stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                       "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                       "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                       "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                       "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                       "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                       "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
                       "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
                       "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                       "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                       "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list, tokenizer=LemmaTokenizer())
    doc_matrix = tfidf_vectorizer.fit_transform(
        sci_kb)

    tfidf_dev_result = eval_tfidf(dev_list, tfidf_vectorizer, doc_matrix, sci_kb, "dev_scores.pickle")
    tfidf_test_result =  eval_tfidf(test_list, tfidf_vectorizer, doc_matrix, sci_kb, "test_scores.pickle")

    # bert_dev_result = load_bert_scores(generated_data_path+"openbook_retrieval_seed_0_2020-05-16_0011/dev_dict.pickle")
    # bert_test_result = load_bert_scores(generated_data_path+"openbook_retrieval_seed_0_2020-05-16_0011/test_dict.pickle")

    # bert_dev_result = load_bert_scores(
    #     generated_data_path + "openbook_retrieval_seed_1_2020-05-21_2225/dev_dict.pickle")
    # bert_test_result = load_bert_scores(
    #     generated_data_path + "openbook_retrieval_seed_1_2020-05-21_2225/test_dict.pickle")

    bert_dev_result = load_bert_scores(
        generated_data_path + "openbook_retrieval_seed_0_2020-05-14_1944/dev_dict.pickle")
    bert_test_result = load_bert_scores(
        generated_data_path + "openbook_retrieval_seed_0_2020-05-14_1944/test_dict.pickle")

    performance_comparison(tfidf_dev_result, bert_dev_result, 0.5)
    performance_comparison(tfidf_test_result, bert_test_result, 0.5)

    return 0

main()


