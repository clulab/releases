import pickle
import sys

exp_num = sys.argv[1]
depth = sys.argv[2]

with open("saved_models/20201118_t5_small/exp_"+exp_num+"_depth_"+depth+".pickle", "rb") as handle:
    results = pickle.load(handle)

print(#"pred_hit_list:", results["pred_hit_list"],
        "\nacc:", results["acc"],
        #"\ntime_list:", results["time_list"],
        "\navg_time:", results["avg_time"],
        "\nproof_acc:", results["proof_acc"],
      "\nnum of results:", len(results["pred_hit_list"]))

instance_outputs= results["instance_output"]

while 1:
    print("-"*20)
    instance_num = input("input instance to check:")
    print(instance_outputs[int(instance_num)])

