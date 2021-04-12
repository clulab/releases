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
from debug_dataloader import LoadRawData

def check_with_bootstrap_resampling(baseline_mrrs, hybrid_mrrs):
    print("check with bootstrap resampling ...")
    print("\tresampling size:", len(baseline_mrrs), len(hybrid_mrrs), " baseline shape:", baseline_mrrs.shape, " control shape:", hybrid_mrrs.shape)
    print("\tbaseline mrr:", np.mean(baseline_mrrs), " control mrr:", np.mean(hybrid_mrrs))

    count=0
    for i in range(10000):
        sampled_indx = np.random.choice(len(baseline_mrrs), len(hybrid_mrrs))

        baseline_mrr_all = np.sum(baseline_mrrs[sampled_indx])
        hybrid_mrr_all = np.sum(hybrid_mrrs[sampled_indx])

        if hybrid_mrr_all>baseline_mrr_all:
            count+=1

        # if i%1000==0:
        #     print("resampling ", i, " ... tfidf score:",tfdif_mrr_all, " bert score:", bert_mrr_all, " hybrid score:", hybrid_mrr_all )

    return count/10000

class OpenbookBootstrap():
    def __init__(self, use_ann = True):
        self.dataset_result = LoadRawData("openbook")

        self.bm25_test_mrrs = self.dataset_result.result_test_bm25["mrr"]
        self.useqa_test_mrrs = self.dataset_result.result_test_useqa["mrr"]

        if use_ann:
            with open(generated_data_path+"/hybrid_classifier_result/openbook_hybrid_ann_result.pickle", "rb") as handle:
                self.hybrid_models_result = pickle.load(handle)
        else:
            with open(generated_data_path+"/hybrid_classifier_result/openbook_hybrid_result.pickle", "rb") as handle:
                self.hybrid_models_result = pickle.load(handle)

        self.hybrid_threshold_test_mrrs = self.hybrid_models_result["hybrid_threshold"]["mrr"]
        self.hybrid_lr_test_mrrs = self.hybrid_models_result["hybrid_lr"]["mrr"]

        # load squad test data.
        with open(generated_data_path + "openbook_useqa/openbook_useqa_retrieval_data.pickle", "rb") as handle:
            openbook_retrieval_raw = pickle.load(handle)

        self.test_list = openbook_retrieval_raw["test_list"]
        self.kb = openbook_retrieval_raw["kb"]

        # load the array indicating which uses ueural methods.
        self.router_pred = self.hybrid_models_result["hybrid_lr"]["router_output"]

    def run_bootstrap(self):

        bs_hybridlr_hybridt = check_with_bootstrap_resampling(self.hybrid_lr_test_mrrs, self.hybrid_threshold_test_mrrs)  # should be less than 0.95

        bs_hybridlr_useqa = check_with_bootstrap_resampling(self.hybrid_lr_test_mrrs, self.useqa_test_mrrs) # should be larger than 0.95
        bs_hybridt_useqa = check_with_bootstrap_resampling(self.hybrid_threshold_test_mrrs, self.useqa_test_mrrs) # should be larger than 0.95

        bs_bm25_hybridlr = check_with_bootstrap_resampling(self.bm25_test_mrrs, self.hybrid_lr_test_mrrs)  # should be larger than 0.95
        bs_bm25_hybridt = check_with_bootstrap_resampling(self.bm25_test_mrrs, self.hybrid_threshold_test_mrrs)  # should be larger than 0.95

        print(bs_hybridlr_hybridt, bs_hybridlr_useqa, bs_hybridt_useqa, bs_bm25_hybridlr, bs_bm25_hybridt)
        # result = 0.8907 0.8884 0.3311 1.0 1.0 which means hyrbid lr is as good as threshold and useqa.

    def debug_get_router_statistics(self):
        router_pred_bool = self.router_pred.astype(bool)
        print("total num test:", len(router_pred_bool))
        print("total number neural:", np.sum(self.router_pred))
        print("bm 25 imporved by neural:", np.sum(self.bm25_test_mrrs[router_pred_bool]<self.hybrid_lr_test_mrrs[router_pred_bool]))
        print("bm 25 harmed by neural:", np.sum(self.bm25_test_mrrs[router_pred_bool]>self.hybrid_lr_test_mrrs[router_pred_bool]))
        print("original bm 25 mrr in neural:", np.mean(self.bm25_test_mrrs[router_pred_bool]))
        print("neural mrr in neural:", np.mean(self.hybrid_lr_test_mrrs[router_pred_bool]))

        mrrs_bm25_better = self.bm25_test_mrrs[self.bm25_test_mrrs>=self.useqa_test_mrrs]
        mrrs_useqa_better = self.useqa_test_mrrs[self.useqa_test_mrrs>self.bm25_test_mrrs]

        print("ceiling:", np.mean(np.concatenate([mrrs_bm25_better, mrrs_useqa_better])), " size:", len(mrrs_useqa_better)+len(mrrs_bm25_better))


    def manual_check(self):
        all_instances = self.test_list
        for idx, pred in enumerate(self.router_pred):
            print("="*20)
            print("\tquery:", all_instances[idx]["question"], " pred:", pred)
            print("\tbm25 mrr:", self.bm25_test_mrrs[idx], "useqa mrr:", self.useqa_test_mrrs[idx])
            print("\t"+'-'*20 )
            print("\ttop bm25 facts:")
            for top_fact in self.dataset_result.result_test_bm25["top_facts"][idx][:5]:
                print("\t\t", self.kb[top_fact])
            print("\ttop useqa facts:")
            for top_fact in self.dataset_result.result_test_useqa["top_facts"][idx][:5]:
                print("\t\t", self.kb[top_fact])

            input("A")

    def get_formal_statistics_in_paper(self):
        print("n routed bm25:", np.sum(self.router_pred==0))
        print("n routed to useQA:", np.sum(self.router_pred))
        print("n improved vs bm25:", np.sum(self.hybrid_lr_test_mrrs>self.bm25_test_mrrs))
        print("n harmed vs bm25:", np.sum(self.hybrid_lr_test_mrrs<self.bm25_test_mrrs))
        print("n improved vs useqa:", np.sum(self.hybrid_lr_test_mrrs > self.useqa_test_mrrs))
        print("n harmed vs useqa:", np.sum(self.hybrid_lr_test_mrrs < self.useqa_test_mrrs))

    def get_time_statistics(self):
        self.useqa_encoding_time = np.load(generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/openbook_test_query_time.npy")

        bm25_time = self.dataset_result.result_test_bm25["search_time"]
        useqa_time = self.dataset_result.result_test_useqa["search_time"]+self.dataset_result.result_test_useqa["sort_time"]+self.useqa_encoding_time

        threshold_router_pred_pool = self.hybrid_models_result["hybrid_threshold"]["router_output"].astype(bool)
        lr_router_pred_pool = self.hybrid_models_result["hybrid_lr"]["router_output"].astype(bool)

        print("bm25 time:", np.sum(bm25_time))   # should be only parsing, search and sort time
        print("useqa time:", np.sum(useqa_time))
        print("unsupervised sum time:", np.sum(bm25_time+useqa_time+self.hybrid_models_result["unsupervised_sum"]["time"]))
        print("hybrid threshold time:", np.sum(bm25_time+self.hybrid_models_result["hybrid_threshold"]["time"])+np.sum(useqa_time[threshold_router_pred_pool]))
        print("hybrid lr time:", np.sum(bm25_time+self.hybrid_models_result["hybrid_lr"]["time"])+np.sum(useqa_time[lr_router_pred_pool]))

class SquadBootstrap():
    def __init__(self, use_ann = True):
        self.dataset_result = LoadRawData("squad")

        self.bm25_test_mrrs = np.concatenate([self.dataset_result.result_test_bm25["mrr"] for i in range(5)])
        self.useqa_test_mrrs = np.concatenate([self.dataset_result.result_test_useqa["mrr"] for i in range(5)])

        self.use_ann = use_ann

        if use_ann:
            with open(generated_data_path+"/hybrid_classifier_result/squad_hybrid_ann_result.pickle", "rb") as handle:
                self.hybrid_models_result = pickle.load(handle)
        else:
            with open(generated_data_path+"/hybrid_classifier_result/squad_hybrid_result.pickle", "rb") as handle:
                self.hybrid_models_result = pickle.load(handle)

        self.hybrid_threshold_test_mrrs = np.concatenate([single_seed["hybrid_threshold"]["mrr"] for single_seed in self.hybrid_models_result])
        self.hybrid_lr_test_mrrs =np.concatenate([single_seed["hybrid_lr"]["mrr"] for single_seed in self.hybrid_models_result])

        # load squad test data.
        with open(generated_data_path+"squad_useqa/squad_retrieval_data.pickle", "rb") as handle:
            squad_retrieval_raw = pickle.load(handle)

        self.test_list = squad_retrieval_raw["dev_list"]
        self.sent_list = squad_retrieval_raw["sent_list"]
        self.doc_list = squad_retrieval_raw["doc_list"]
        self.resp_list = squad_retrieval_raw["resp_list"]

        # load the array indicating which uses ueural methods.
        self.router_pred = np.concatenate([single_seed["hybrid_lr"]["router_output"] for single_seed in self.hybrid_models_result])

    def run_bootstrap(self):
        bs_hybridlr_hybridt = check_with_bootstrap_resampling(self.hybrid_lr_test_mrrs,
                                                              self.hybrid_threshold_test_mrrs)  # should be less than 0.95

        bs_useqa_hybridlr = check_with_bootstrap_resampling(self.useqa_test_mrrs, self.hybrid_lr_test_mrrs)  # should be larger than 0.95
        bs_useqa_hybridt = check_with_bootstrap_resampling(self.useqa_test_mrrs, self.hybrid_threshold_test_mrrs)  # should be larger than 0.95

        bs_bm25_hybridlr = check_with_bootstrap_resampling(self.bm25_test_mrrs,
                                                           self.hybrid_lr_test_mrrs)  # should be larger than 0.95
        bs_bm25_hybridt = check_with_bootstrap_resampling(self.bm25_test_mrrs,
                                                          self.hybrid_threshold_test_mrrs)  # should be larger than 0.95

        print(bs_hybridlr_hybridt, bs_useqa_hybridlr, bs_useqa_hybridt, bs_bm25_hybridlr, bs_bm25_hybridt)

    def debug_get_router_statistics(self):
        router_pred_bool = self.router_pred.astype(bool)
        print("total num test:", len(router_pred_bool))
        print("total number neural:", np.sum(self.router_pred))
        print("bm 25 imporved by neural:", np.sum(self.bm25_test_mrrs[router_pred_bool]<self.hybrid_lr_test_mrrs[router_pred_bool]))
        print("bm 25 harmed by neural:", np.sum(self.bm25_test_mrrs[router_pred_bool]>self.hybrid_lr_test_mrrs[router_pred_bool]))
        print("original bm 25 mrr in neural:", np.mean(self.bm25_test_mrrs[router_pred_bool]))
        print("neural mrr in neural:", np.mean(self.hybrid_lr_test_mrrs[router_pred_bool]))

        mrrs_bm25_better = self.bm25_test_mrrs[self.bm25_test_mrrs>=self.useqa_test_mrrs]
        mrrs_useqa_better = self.useqa_test_mrrs[self.useqa_test_mrrs>self.bm25_test_mrrs]

        print("ceiling:", np.mean(np.concatenate([mrrs_bm25_better, mrrs_useqa_better])), " size:", len(mrrs_useqa_better)+len(mrrs_bm25_better))


    def manual_check(self):
        all_instances = self.test_list*4
        for idx, pred in enumerate(self.router_pred):
            if pred==1:
                print("="*20)
                print("\tquery:", all_instances[idx]["question"])
                print("\tbm25 mrr:", self.bm25_test_mrrs[idx], "useqa mrr:", self.useqa_test_mrrs[idx])
                print("\t"+'-'*20 )
                print("\ttop bm25 facts:")
                for top_fact in self.dataset_result.result_test_bm25["top_facts"][idx][:5]:
                    print("\t\t", self.sent_list[int(self.resp_list[top_fact][0])])
                print("\ttop useqa facts:")
                for top_fact in self.dataset_result.result_test_useqa["top_facts"][idx][:5]:
                    print("\t\t", self.sent_list[int(self.resp_list[top_fact][0])])

                input("A")

    def get_formal_statistics_in_paper(self):
        print("n routed bm25:", np.sum(self.router_pred==0))
        print("n routed to useQA:", np.sum(self.router_pred))
        print("n improved vs bm25:", np.sum(self.hybrid_lr_test_mrrs>self.bm25_test_mrrs))
        print("n harmed vs bm25:", np.sum(self.hybrid_lr_test_mrrs<self.bm25_test_mrrs))
        print("n improved vs useqa:", np.sum(self.hybrid_lr_test_mrrs > self.useqa_test_mrrs))
        print("n harmed vs useqa:", np.sum(self.hybrid_lr_test_mrrs < self.useqa_test_mrrs))

    def get_time_statistics(self):
        self.useqa_encoding_time = np.load(generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/squad_test_query_time.npy")

        bm25_time = self.dataset_result.result_test_bm25["search_time"]
        useqa_time = self.dataset_result.result_test_useqa["search_time"]+self.dataset_result.result_test_useqa["sort_time"]+self.useqa_encoding_time

        bm25_time = np.concatenate([bm25_time for i in range(5)])
        useqa_time = np.concatenate([useqa_time for i in range(5)])

        threshold_router_pred_pool = np.concatenate([self.hybrid_models_result[i]["hybrid_threshold"]["router_output"] for i in range(5)]).astype(bool)
        lr_router_pred_pool = np.concatenate([self.hybrid_models_result[i]["hybrid_lr"]["router_output"] for i in range(5)]).astype(bool)

        if self.use_ann == False:
            unsupervsied_meta_time = np.concatenate([self.hybrid_models_result[i]["unsupervised_sum"]["time"] for i in range(5)])
        hybrid_threshold_meta_time = np.concatenate([self.hybrid_models_result[i]["hybrid_threshold"]["time"] for i in range(5)])
        hybrid_lr_meta_time = np.concatenate(([self.hybrid_models_result[i]["hybrid_lr"]["time"] for i in range(5)]))

        print("bm25 time:", np.sum(bm25_time))   # should be only parsing, search and sort time
        print("useqa time:", np.sum(useqa_time))
        if self.use_ann ==  False:
            print("unsupervised sum time:", np.sum(bm25_time+useqa_time+unsupervsied_meta_time))
        print("hybrid threshold time:", np.sum(bm25_time+hybrid_threshold_meta_time)+np.sum(useqa_time[threshold_router_pred_pool]))
        print("hybrid lr time:", np.sum(bm25_time+hybrid_lr_meta_time)+np.sum(useqa_time[lr_router_pred_pool]))


class NqBootstrap():
    def __init__(self, use_ann = True):
        self.dataset_result = LoadRawData("nq")

        self.bm25_test_mrrs_raw = self.dataset_result.result_test_bm25
        self.useqa_test_mrrs_raw = self.dataset_result.result_test_useqa

        self.use_ann = use_ann

        if use_ann:
            with open(generated_data_path+"/hybrid_classifier_result/nq_hybrid_ann_result.pickle", "rb") as handle:
                self.hybrid_models_result = pickle.load(handle)
        else:
            with open(generated_data_path+"/hybrid_classifier_result/nq_hybrid_result.pickle", "rb") as handle:
                self.hybrid_models_result = pickle.load(handle)

        all_test_indices = [single_seed["test_index_in_all_list"] for single_seed in self.hybrid_models_result]

        self.bm25_test_mrrs = np.concatenate([self.bm25_test_mrrs_raw["mrr"][test_split] for test_split in all_test_indices])
        self.useqa_test_mrrs = np.concatenate([self.useqa_test_mrrs_raw["mrr"][test_split] for test_split in all_test_indices])

        self.hybrid_threshold_test_mrrs = np.concatenate([single_seed["hybrid_threshold"]["mrr"] for single_seed in self.hybrid_models_result])
        self.hybrid_lr_test_mrrs =np.concatenate([single_seed["hybrid_lr"]["mrr"] for single_seed in self.hybrid_models_result])

        # load nq test data.
        with open(generated_data_path + "nq_retrieval_raw/nq_retrieval_data.pickle", "rb") as handle:
            nq_retrieval_raw = pickle.load(handle)

        print(nq_retrieval_raw.keys())

        self.test_list = nq_retrieval_raw["train_list"]
        self.sent_list = nq_retrieval_raw["sent_list"]
        self.doc_list = nq_retrieval_raw["doc_list"]
        self.resp_list = nq_retrieval_raw["resp_list"]

        # load the array indicating which uses ueural methods.
        self.router_pred = np.concatenate(
            [single_seed["hybrid_lr"]["router_output"] for single_seed in self.hybrid_models_result])

    def run_bootstrap(self):
        bs_hybridlr_hybridt = check_with_bootstrap_resampling(self.hybrid_lr_test_mrrs,
                                                              self.hybrid_threshold_test_mrrs)  # should be less than 0.95

        bs_useqa_hybridlr = check_with_bootstrap_resampling(self.useqa_test_mrrs,
                                                            self.hybrid_lr_test_mrrs)  # should be larger than 0.95
        bs_useqa_hybridt = check_with_bootstrap_resampling(self.useqa_test_mrrs,
                                                           self.hybrid_threshold_test_mrrs)  # should be larger than 0.95

        bs_bm25_hybridlr = check_with_bootstrap_resampling(self.bm25_test_mrrs,
                                                           self.hybrid_lr_test_mrrs)  # should be larger than 0.95
        bs_bm25_hybridt = check_with_bootstrap_resampling(self.bm25_test_mrrs,
                                                          self.hybrid_threshold_test_mrrs)  # should be larger than 0.95

        print(bs_hybridlr_hybridt, bs_useqa_hybridlr, bs_useqa_hybridt, bs_bm25_hybridlr, bs_bm25_hybridt)

    def debug_get_router_statistics(self):
        router_pred_bool = self.router_pred.astype(bool)
        print("total num test:", len(router_pred_bool))
        print("total number neural:", np.sum(self.router_pred))
        print("bm 25 imporved by neural:", np.sum(self.bm25_test_mrrs[router_pred_bool]<self.hybrid_lr_test_mrrs[router_pred_bool]))
        print("bm 25 harmed by neural:", np.sum(self.bm25_test_mrrs[router_pred_bool]>self.hybrid_lr_test_mrrs[router_pred_bool]))
        print("original bm 25 mrr in neural:", np.mean(self.bm25_test_mrrs[router_pred_bool]))
        print("neural mrr in neural:", np.mean(self.hybrid_lr_test_mrrs[router_pred_bool]))

        mrrs_bm25_better = self.bm25_test_mrrs[self.bm25_test_mrrs>=self.useqa_test_mrrs]
        mrrs_useqa_better = self.useqa_test_mrrs[self.useqa_test_mrrs>self.bm25_test_mrrs]

        print("ceiling:", np.mean(np.concatenate([mrrs_bm25_better, mrrs_useqa_better])), " size:", len(mrrs_useqa_better)+len(mrrs_bm25_better))

    def manual_check(self):
        all_instances = self.test_list*4
        for idx, pred in enumerate(self.router_pred):
            if pred==1:
                print("="*20)
                print("\tquery:", all_instances[idx]["question"])
                print("\tbm25 mrr:", self.bm25_test_mrrs[idx], "useqa mrr:", self.useqa_test_mrrs[idx])
                print("\t"+'-'*20 )
                print("\ttop bm25 facts:")
                for top_fact in self.dataset_result.result_test_bm25["top_facts"][idx][:5]:
                    print("\t\t", self.sent_list[int(self.resp_list[top_fact][0])])
                print("\ttop useqa facts:")
                for top_fact in self.dataset_result.result_test_useqa["top_facts"][idx][:5]:
                    print("\t\t", self.sent_list[int(self.resp_list[top_fact][0])])

                input("A")

    def get_formal_statistics_in_paper(self):
        print("n routed bm25:", np.sum(self.router_pred==0))
        print("n routed to useQA:", np.sum(self.router_pred))
        print("n improved vs bm25:", np.sum(self.hybrid_lr_test_mrrs>self.bm25_test_mrrs))
        print("n harmed vs bm25:", np.sum(self.hybrid_lr_test_mrrs<self.bm25_test_mrrs))
        print("n improved vs useqa:", np.sum(self.hybrid_lr_test_mrrs > self.useqa_test_mrrs))
        print("n harmed vs useqa:", np.sum(self.hybrid_lr_test_mrrs < self.useqa_test_mrrs))

    def get_time_statistics(self):
        self.useqa_encoding_time = np.load(generated_data_path+"useQA_results_mrr_time_openbook_squad_nq/nq_test_query_time.npy")

        bm25_time = self.dataset_result.result_test_bm25["search_time"]
        useqa_time = self.dataset_result.result_test_useqa["search_time"]+self.dataset_result.result_test_useqa["sort_time"]+self.useqa_encoding_time

        bm25_time = np.concatenate([bm25_time[self.hybrid_models_result[i]["test_index_in_all_list"]] for i in range(5)])
        useqa_time = np.concatenate([useqa_time[self.hybrid_models_result[i]["test_index_in_all_list"]] for i in range(5)])

        threshold_router_pred_pool = np.concatenate([self.hybrid_models_result[i]["hybrid_threshold"]["router_output"] for i in range(5)]).astype(bool)
        lr_router_pred_pool = np.concatenate([self.hybrid_models_result[i]["hybrid_lr"]["router_output"] for i in range(5)]).astype(bool)

        if self.use_ann == False:
            unsupervsied_meta_time = np.concatenate([self.hybrid_models_result[i]["unsupervised_sum"]["time"] for i in range(5)])
        hybrid_threshold_meta_time = np.concatenate([self.hybrid_models_result[i]["hybrid_threshold"]["time"] for i in range(5)])
        hybrid_lr_meta_time = np.concatenate(([self.hybrid_models_result[i]["hybrid_lr"]["time"] for i in range(5)]))

        print("bm25 time:", np.sum(bm25_time))   # should be only parsing, search and sort time
        print("useqa time:", np.sum(useqa_time))

        if self.use_ann == False:
            print("unsupervised sum time:", np.sum(bm25_time+useqa_time+unsupervsied_meta_time))
        print("hybrid threshold time:", np.sum(bm25_time+hybrid_threshold_meta_time)+np.sum(useqa_time[threshold_router_pred_pool]))
        print("hybrid lr time:", np.sum(bm25_time+hybrid_lr_meta_time)+np.sum(useqa_time[lr_router_pred_pool]))



# openbookBS = OpenbookBootstrap()
# openbookBS.run_bootstrap()
#openbookBS.debug_get_router_statistics()
#openbookBS.manual_check()
#openbookBS.get_formal_statistics_in_paper()
#openbookBS.get_time_statistics()

squadBS = SquadBootstrap(True)
squadBS.run_bootstrap()
#squadBS.debug_get_router_statistics()
#squadBS.manual_check()
#squadBS.get_formal_statistics_in_paper()
squadBS.get_time_statistics()

nqBS = NqBootstrap(True)
nqBS.run_bootstrap()
#nqBS.debug_get_router_statistics()
#nqBS.get_formal_statistics_in_paper()
nqBS.get_time_statistics()
