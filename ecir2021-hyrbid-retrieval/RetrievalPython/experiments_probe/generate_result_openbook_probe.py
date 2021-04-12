import pickle
import numpy as np

def generate_probe_result_openbook():

    root_foler_path = "data_generated/openbook/probe_experiment_2020-05-30_125749/"

    exp_folder_paths = {
        "useqa_embd_gold_label": "query_useqa_embd_gold_result_seed_",
        "rand_embd_gold_label": "query_random_embd_gold_result_seed_",
        "tfidf_embd_gold_label": "query_tfidf_embd_gold_result_seed_",
        "useqa_embd_rand_label": "query_useqa_embd_ques_shuffle_result_seed_"
    }

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

            # print(result_dict["query map:"])

            # print(np.std(result_dict["target map:"]))


        print("=" * 20)
        print(len(query_map), len(query_ppl), len(target_map), len(target_ppl))
        print(exp_name)
        print("query map\tquery ppl\ttarget map\ttarget ppl")
        print(
            "%.3f {\\tiny $\\pm%.3f$} & %.3f {\\tiny $\\pm%.3f$} & %.3f {\\tiny $\\pm%.3f$} & %.3f {\\tiny $\\pm%.3f$} \\\\" % (
            np.mean(query_map).item(), np.std(query_map).item(), np.mean(query_ppl).item(), np.std(query_ppl).item(),
            np.mean(target_map).item(), np.std(target_map).item(), np.mean(target_ppl).item(),
            np.std(target_ppl).item()))


generate_probe_result_openbook()

