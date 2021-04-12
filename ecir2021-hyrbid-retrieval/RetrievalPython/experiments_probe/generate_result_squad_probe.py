import pickle
import numpy as np
def check_with_bootstrap_resampling(baseline_mrrs, hybrid_mrrs):
    print("check with bootstrap resampling ...")
    print(" baseline shape:", baseline_mrrs.shape, " control shape:", hybrid_mrrs.shape)
    print("\tbaseline val:", np.mean(baseline_mrrs), " control val:", np.mean(hybrid_mrrs))

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


def generate_probe_result_squad():

    root_foler_path = "data_generated/squad/probe_experiment_2020-05-31_002033/"

    exp_folder_paths = {
        "useqa_embd_gold_label": "query_useqa_embd_gold_result_seed_",
        "rand_embd_gold_label": "query_random_embd_gold_result_seed_",
        "tfidf_embd_gold_label": "query_tfidf_embd_gold_result_seed_",
        "useqa_embd_rand_label": "query_useqa_embd_ques_shuffle_result_seed_"
    }

    all_results = {}
    for exp_name in ["useqa_embd_gold_label", "tfidf_embd_gold_label" , "rand_embd_gold_label", "useqa_embd_rand_label"]:
        query_map = []
        query_ppl = []
        target_map = []
        target_ppl = []
        for seed in range(5):
            result_dict_name = root_foler_path+exp_folder_paths[exp_name]+str(seed)+"/best_epoch_result.pickle"

            with open(result_dict_name,"rb") as handle:
                result_dict = pickle.load(handle)

            query_map.append(np.mean(result_dict["query map:"]))
            query_ppl.append(np.mean(result_dict["query ppl:"]))
            target_map.append(np.mean(result_dict["target map:"]))
            target_ppl.append(np.mean(result_dict["target ppl:"]))

            #print(result_dict["query map:"])

            #print(np.std(result_dict["target map:"]))

        all_results[exp_name] = np.array(target_ppl)

        print("="*20)
        print(len(query_map), len(query_ppl), len(target_map), len(target_ppl))
        print(exp_name)
        print("query map\tquery ppl\ttarget map\ttarget ppl")
        print("%.3f {\\tiny $\\pm%.3f$} & %.3f {\\tiny $\\pm%.3f$} & %.3f {\\tiny $\\pm%.3f$} & %.3f {\\tiny $\\pm%.3f$} \\\\" % (np.mean(query_map).item(), np.std(query_map).item(), np.mean(query_ppl).item(), np.std(query_ppl).item(),np.mean(target_map).item(), np.std(target_map).item(), np.mean(target_ppl).item(), np.std(target_ppl).item()))
        # print(np.mean(np.array(query_map)), np.mean(np.array(query_ppl)), np.mean(np.array(target_map)), np.mean(np.array(target_ppl)))
        #

    return all_results


results = generate_probe_result_squad()

#print(check_with_bootstrap_resampling(results["tfidf_embd_gold_label"], results["useqa_embd_gold_label"]))
# useqa target map vs tfidf target map: 1.0 (this is good. )
# useqa target ppl vs tfidf target ppl: 1.0 (means useqa target ppl is larger than tfidf, which is not good)


