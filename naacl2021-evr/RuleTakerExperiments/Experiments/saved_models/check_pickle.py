import pickle

def check_by_exp_series():
    exp_num = 11

    exp_num_depth_map = {"7":5, "8":4, "9":4, "10":4, "11":4}

    # 7: EVR1 train on DU5, test on DU5
    # 8: EVR1 train on DU5, test on BE

    # 9: EVR2 train on DU1, test on BE
    # 10: EVR3 train on DU1, test on BE

    for d in range(exp_num_depth_map[str(exp_num)]+1):
        with open("exp_eval_results/exp_" + str(exp_num) + "_depth_" + str(d) + ".pickle", "rb") as handle:
            a = pickle.load(handle)

            print(a["acc"])


def check_by_single_file():
    with open("exp_eval_results/exp_10_depth_0.pickle", "rb") as handle:
        a = pickle.load(handle)

        print(a["acc"])

def check_by_single_file_show_all():
    with open("exp_eval_results/exp_8_depth_2.pickle", "rb") as handle:
        a = pickle.load(handle)

    hits_list = a["pred_hit_list"]
    miss_idx = []
    hit_idx = []
    for i, hit in enumerate(hits_list):
        if hit == 0:
            miss_idx.append(i)
        else:
            hit_idx.append(i)

    print(hit_idx)
    print("="*20)
    print(miss_idx)


#check_by_single_file()
check_by_exp_series()
#check_by_single_file_show_all()