import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
generated_data_path = parent_folder_path+"/data_generated/"
sys.path+=[parent_folder_path, datasets_folder_path, generated_data_path]

import numpy as np
import pickle
import math
import time

class EvalBase():
    def __init__(self, batch_size, embd_dim_size, answer_field, n_top_doc_save):
        '''
        defines some functions that are useful to all evaluation tasks
        '''
        self.batch_size = 1 # This is hard coded to 1 to facilitate time measuring.
        self.embd_dim_size = embd_dim_size
        self.answer_field = answer_field
        self.n_top_doc_save = n_top_doc_save

        self.t0 = 0   # time before matmul
        self.t1 = 0   # time after matmul
        self.t2 = 0   # time before argsort
        self.t3 = 0   # time after argsort

    def _fill_results_dict(self, gold_facts_indices_list, query_embds_batch, fact_embds, result_dict):
        for i in range(20):
            np.argsort(np.random.rand(2000))
            np.matmul(np.random.rand(50,50), np.random.rand(50,50))
        # Things to return:
        batch_size = len(query_embds_batch)

        gold_facts_indices = np.array(gold_facts_indices_list).reshape((batch_size, 1))  # size: n_query * 1

        # batch_scores = softmax(np.matmul(query_embds_batch, fact_embds))  # size: n_query * n_facts
        self.t0 = time.time()
        batch_scores = np.matmul(query_embds_batch, fact_embds)
        self.t1 = time.time()

        sorted_scores = np.flip(np.sort(batch_scores, axis=1), axis=1)

        self.t2 = time.time()
        sorted_facts_ = np.argsort(batch_scores, axis=1)
        self.t3 = time.time()

        sorted_facts = np.flip(sorted_facts_  , axis=1)

        gold_fact_rankings_indices_row, gold_fact_rankings = np.where(sorted_facts == gold_facts_indices)

        result_dict["gold_fact_index"].extend(gold_facts_indices.flatten().tolist())
        result_dict["gold_fact_ranking"].extend(gold_fact_rankings.tolist())  # get the gold fact ranking of each query
        result_dict["gold_fact_score"].extend(
            sorted_scores[gold_fact_rankings_indices_row, gold_fact_rankings].tolist())
        result_dict["mrr"].extend((1 / (1 + gold_fact_rankings)).tolist())

        result_dict["top_facts"].extend(sorted_facts[:, :self.n_top_doc_save].tolist())
        result_dict["top_scores"].extend(sorted_scores[:, :self.n_top_doc_save].tolist())

        result_dict["search_time"].append(self.t1-self.t0)
        result_dict["sort_time"].append(self.t3-self.t2)

        return 0

    def _eval(self, query_embds, fact_embds, instances_list):
        result_dict = {"gold_fact_index":[], "gold_fact_ranking":[], "gold_fact_score":[], "mrr":[], "top_facts":[], "top_scores":[], "search_time":[], "sort_time":[]}
        n_batch = math.ceil(len(instances_list)/self.batch_size)

        fact_embds_t = np.transpose(fact_embds) # transpose for better calculation
        for i in range(n_batch):
            if ((i+1)%100)==0:
                print("eval batch "+str(i+1)+" out of "+str(n_batch) + " batches ....")

            if i!=n_batch-1:
                instances_labels_list = [instance[self.answer_field] for instance in instances_list[i*self.batch_size:(i+1)*self.batch_size]]
            else:
                instances_labels_list = [instance[self.answer_field] for instance in instances_list[i*self.batch_size:]]

            exact_batch_size = len(instances_labels_list)

            query_embds_batch = query_embds[i*self.batch_size:i*self.batch_size+exact_batch_size,:].reshape((exact_batch_size, self.embd_dim_size))

            self._fill_results_dict(instances_labels_list, query_embds_batch, fact_embds_t, result_dict)

        return result_dict

    def _trim_result_dict(self, result_dict, reduce_precision = False):

        for stat_field in result_dict.keys():
            result_dict[stat_field] = np.array(result_dict[stat_field])

        if reduce_precision:
            result_dict["top_scores"] = result_dict["top_scores"].astype(np.float32)
            result_dict["gold_fact_score"] = result_dict["gold_fact_score"].astype(np.float32)
            result_dict["mrr"] = result_dict["mrr"].astype(np.float32)

            result_dict["top_facts"] = result_dict["top_facts"].astype(np.int32)
            result_dict["gold_fact_index"] = result_dict["gold_fact_index"].astype(np.int32)
            result_dict["gold_fact_ranking"] = result_dict["gold_fact_ranking"].astype(np.int32)

            result_dict["search_time"] = result_dict["search_time"].astype(np.float32)
            result_dict["sort_time"] = result_dict["sort_time"].astype(np.float32)

        print(result_dict["top_scores"][0,0].dtype)
        print(result_dict["top_facts"][0,0].dtype)

        return result_dict

class EvalOpenbook(EvalBase):

    def __init__(self):
        super(EvalOpenbook, self).__init__(batch_size = 5, embd_dim_size = 512, answer_field="label", n_top_doc_save=1326)

        self.fact_embds = np.load(generated_data_path+"openbook_useqa/openbook_sents_embds.npy")

        with open(generated_data_path+"openbook_useqa/openbook_useqa_retrieval_data.pickle","rb") as handle:
            self.instances_list = pickle.load(handle)

    def eval_dev(self):
        # evaluate dev:
        query_embds_dev = np.load(generated_data_path+"openbook_useqa/openbook_ques_dev_embds.npy")
        dev_result_dict = self._eval(query_embds_dev, self.fact_embds, self.instances_list["dev_list"])

        dev_result_dict = self._trim_result_dict(dev_result_dict, reduce_precision = True)
        with open("openbook_dev_useqa_result_dict.pickle", "wb") as handle:
            pickle.dump(dev_result_dict, handle)

        mrr_list = dev_result_dict["mrr"]

        print("openbook useqa dev mrr:", sum(mrr_list) / len(mrr_list))

    def eval_test(self):
        # evaluate dev:
        query_embds_test = np.load(generated_data_path + "openbook_useqa/openbook_ques_test_embds.npy")
        test_result_dict = self._eval(query_embds_test, self.fact_embds, self.instances_list["test_list"])

        test_result_dict = self._trim_result_dict(test_result_dict, reduce_precision=True)
        with open("openbook_test_useqa_result_dict.pickle", "wb") as handle:
            pickle.dump(test_result_dict, handle)

        mrr_list = test_result_dict["mrr"]

        print("openbook useqa test mrr:", sum(mrr_list) / len(mrr_list))

class EvalSQuAD(EvalBase):

    def __init__(self):
        super(EvalSQuAD, self).__init__(batch_size = 5, embd_dim_size = 512, answer_field="response", n_top_doc_save=2000)

        self.fact_embds = np.load(generated_data_path+"squad_useqa/sents_embds.npy")

        with open(generated_data_path+"squad_useqa/squad_retrieval_data.pickle","rb") as handle:
            self.instances_list = pickle.load(handle)

    def _get_dev_indexes(self):
        query_id2idx_dict = {}
        for i, instance in enumerate(self.instances_list["train_list"]):
            query_id2idx_dict[instance["id"]] = i

        # Load squad dev query id
        with open("/Users/zhengzhongliang/NLP_Research/2020_HybridRetrieval/IR_BM25/data/squad/raw_data/squad_dev_id.txt","r") as handle:
            query_id_list = handle.read().split("\n")[:-1]  # [-1] is to remove the last line, which is empty

        assert(len(query_id_list)==10000)

        with open("/Users/zhengzhongliang/NLP_Research/2020_HybridRetrieval/IR_BM25/data/squad/raw_data/squad_dev_label.txt","r") as handle:
            query_label_list = handle.read().split("\n")[:-1]

        assert(len(query_label_list)==10000)

        dev_query_idx_in_train_list = []
        for i, query_id in enumerate(query_id_list):
            dev_query_idx_in_train_list.append(query_id2idx_dict[query_id])
            assert(int(query_label_list[i])==self.instances_list["train_list"][dev_query_idx_in_train_list[-1]]["response"])

        return dev_query_idx_in_train_list


    def eval_dev(self):
        dev_query_idx_in_train_list = self._get_dev_indexes()

        dev_list = [self.instances_list["train_list"][idx] for idx in dev_query_idx_in_train_list]

        # evaluate dev:
        query_embds_dev = np.load(generated_data_path+"squad_useqa/ques_train_embds.npy")
        query_embds_dev = np.array([query_embds_dev[idx, :] for idx in dev_query_idx_in_train_list])
        assert(len(query_embds_dev)==10000)
        dev_result_dict = self._eval(query_embds_dev, self.fact_embds, dev_list)

        dev_result_dict["dev_index_in_train_list"] = dev_query_idx_in_train_list

        dev_result_dict = self._trim_result_dict(dev_result_dict, reduce_precision=True)
        with open("squad_dev_useqa_result_dict.pickle", "wb") as handle:
            pickle.dump(dev_result_dict, handle)

        mrr_list = dev_result_dict["mrr"]

        print("squad useqa dev mrr:", sum(mrr_list)/len(mrr_list))

    def eval_test(self):
        # evaluate dev:
        query_embds_test = np.load(generated_data_path + "squad_useqa/ques_dev_embds.npy")
        test_result_dict = self._eval(query_embds_test, self.fact_embds, self.instances_list["dev_list"])

        test_result_dict = self._trim_result_dict(test_result_dict, reduce_precision=True)
        with open("squad_test_useqa_result_dict.pickle", "wb") as handle:
            pickle.dump(test_result_dict, handle)

        mrr_list = test_result_dict["mrr"]

        print("squad useqa test mrr:", sum(mrr_list) / len(mrr_list))

class EvalNQ(EvalBase):

    def __init__(self):
        super(EvalNQ, self).__init__(batch_size = 5, embd_dim_size = 512, answer_field="response", n_top_doc_save=2000)

        self.fact_embds = np.load(generated_data_path+"nq_retrieval_raw/nq_sents_embds.npy")

        with open(generated_data_path+"nq_retrieval_raw/nq_retrieval_data.pickle","rb") as handle:
            self.instances_list = pickle.load(handle)

    def eval_test(self):
        # evaluate dev:
        query_embds_test = np.load(generated_data_path + "nq_retrieval_raw/nq_ques_train_embds.npy")
        test_result_dict = self._eval(query_embds_test, self.fact_embds, self.instances_list["train_list"])

        test_result_dict = self._trim_result_dict(test_result_dict, reduce_precision=True)
        with open("nq_test_useqa_result_dict.pickle", "wb") as handle:
            pickle.dump(test_result_dict, handle)

        mrr_list = test_result_dict["mrr"]

        print("nq useqa test mrr:", sum(mrr_list) / len(mrr_list))

def check_result_dict_pickle(dataset, partition, debug=True):

    with open(dataset+"_"+partition+"_useqa_result_dict.pickle" ,"rb") as handle:
        result_dict_pickle = pickle.load(handle)

    if debug==True:
        print("=" * 20)
        print("checking result dict "+dataset+" "+partition)
        for result_field in result_dict_pickle.keys():
            print("="*20)
            print(result_field, result_dict_pickle[result_field].shape)
            print(result_dict_pickle[result_field][0])
    else:
        print('='*20)
        print(dataset +"\t"+partition)
        print("\tmrr:", np.mean(result_dict_pickle["mrr"]))
        print("\tavg search time:", np.mean(result_dict_pickle["search_time"]))
        print("\tavg sort time:", np.mean(result_dict_pickle["sort_time"]))

def get_statistics_from_saved_pickle():
    check_result_dict_pickle("openbook", "dev", debug=False)
    check_result_dict_pickle("openbook", "test", debug=False)
    check_result_dict_pickle("squad", "dev", debug=False)
    check_result_dict_pickle("squad", "test", debug=False)
    #check_result_dict_pickle("openbook", "dev")



def main(dataset):
    if dataset=="openbook":
        evalOpenbook = EvalOpenbook()
        evalOpenbook.eval_dev()
        evalOpenbook.eval_test()

        check_result_dict_pickle(dataset, "dev")
        check_result_dict_pickle(dataset, "test")

    if dataset=="squad":
        evalSquad = EvalSQuAD()
        evalSquad.eval_dev()
        evalSquad.eval_test()

        check_result_dict_pickle(dataset, "dev")
        check_result_dict_pickle(dataset, "test")

    if dataset=="nq":
        evalNQ = EvalNQ()
        evalNQ.eval_test()

        check_result_dict_pickle(dataset, "test")

#get_statistics_from_saved_pickle()
main("openbook")




