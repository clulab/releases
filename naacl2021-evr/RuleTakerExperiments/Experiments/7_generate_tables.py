import pickle
import numpy as np
import json

def be_depth_to_dataset_partition(debug_flag = False):
    data_file_path = "/Users/zhengzhongliang/NLP_Research/2020_ThinkInNaturalLanguage/ThinkInNaturalLanguage/Data/rule-reasoning-dataset-V2020.2.4/birds-electricity/meta-test.jsonl"

    with open(data_file_path, "r") as f:
        raw_jsons = list(f)

    instances_all_depth = []

    for depth in range(5):  # BE depth is from 0 to 4.

        instances_current_depth = []

        for raw_json in raw_jsons:
            item = json.loads(raw_json)
            item_id = item["id"]

            if "AttPosElectricity" in item_id:
                if "RB1" in item_id:
                    partition = "RB1"
                elif "RB2" in item_id:
                    partition = "RB2"
                elif "RB3" in item_id:
                    partition = "RB3"
                else:
                    partition = "RB4"
            else:
                if "Var1" in item_id:
                    partition = "Bird1"
                else:
                    partition = "Bird2"

            question_tuples = list(item["questions"].items())

            for question_tuple in question_tuples:

                # print(int(question_tuple[1]["QDep"]))
                if int(question_tuple[1]["QDep"]) == int(depth):

                    instances_current_depth.append(partition)

        instances_all_depth.append(instances_current_depth)

    if debug_flag:
        print([len(instances_one_depth) for instances_one_depth in instances_all_depth])

        all_parition_dict = {}

        for instances_one_depth in instances_all_depth:
            for instance in instances_one_depth:
                if instance not in all_parition_dict:
                    all_parition_dict[instance] = 1
                else:
                    all_parition_dict[instance] += 1


        print(all_parition_dict)

    return instances_all_depth


def generate_result_table_1():

    print("="*40)

    evr_names = ["evr1 du1", "evr2 du1", "evr3 du1", "evr1 du5"]

    for idx, exp in enumerate([1,2,3,7]):

        table_results_list = []
        all_dep_hit_list = []
        for dep in range(6):
            file_name = "saved_models/exp_eval_results/exp_"+str(exp)+"_depth_"+str(dep)+".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            table_results_list.append(result_dict["pred_hit_list"])
            all_dep_hit_list.extend(result_dict["pred_hit_list"])

        table_results_list.append(all_dep_hit_list)

        acc_line = ["{:.1f}".format(sum(hit_list)/len(hit_list)*100) for hit_list in table_results_list]
        table_row_to_show =  evr_names[idx] + " EVR"+str(exp)+" & "+" & ".join(acc_line)
        print(table_row_to_show)

def generate_result_table_2():

    print("=" * 40)

    evr_names = ["evr1 du1", "evr2 du1", "evr3 du1", "evr1 du5"]

    for idx, exp in enumerate([1,2,3,7]):

        list_all_possible_proof = []
        list_supported_proof = []
        list_correct_proof = []

        for dep in range(6):
            file_name = "saved_models/exp_eval_results/exp_"+str(exp)+"_depth_"+str(dep)+".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            result_dict_list = result_dict["instance_output"]

            count_all_possible_proof = 0
            count_supported_proof = 0
            count_correct_proof = 0

            for instance_result_dict in result_dict_list:
                if instance_result_dict["strategy"]=="proof" \
                    or instance_result_dict["strategy"]=="inv-proof":

                    count_all_possible_proof += 1
                    if (instance_result_dict["strategy"]=="proof" and
                        instance_result_dict["label"]==True) or (instance_result_dict["strategy"]=="inv-proof" and
                        instance_result_dict["label"]==False):
                        count_supported_proof+=1
                        if instance_result_dict["proof_flag"]==1:
                            count_correct_proof+=1

            list_all_possible_proof.append(count_all_possible_proof)
            list_supported_proof.append(count_supported_proof)
            list_correct_proof.append(count_correct_proof)

        max_proof_accs = [list_supported_proof[i]/list_all_possible_proof[i]
                      for i in range(len(list_all_possible_proof))] + \
                        [sum(list_supported_proof)/sum(list_all_possible_proof)]
        proof_accs = [list_correct_proof[i]/list_all_possible_proof[i]
                      for i in range(len(list_all_possible_proof))]+ \
                        [sum(list_correct_proof)/sum(list_all_possible_proof)]

        line_to_show = ["{:.1f}".format(proof_acc*100) for proof_acc in proof_accs]
        print(evr_names[idx] + " EVR"+str(exp)+" & "+" & ".join(line_to_show))

    print("cnt "+ " & ".join([str(cnt) for cnt in list_all_possible_proof]+[str(sum(list_all_possible_proof))]))

def generate_result_table_3():
    bird_e_count = [40,40,162,180,624, 4224]

    rt_du1_baseline_acc = [100,100,88.9,80.0, 93.9, 97.5]

    rt_du5_baseline_acc = [97.5, 100, 96.9, 98.3, 91.8, 76.7]

    pr_baseline_acc = [95.0,95.0, 100, 100, 89.7, 84.8 ]

    for exp in [6, 9, 10, 8, 11]:
        # exp 6: EVR1 trained on DU1
        # exp 9: EVR2 trained on DU1
        # exp 10: EVR3 trained on DU1
        # exp 8: EVR1 trained on DU5
        # exp 11: EVR1, controlled on DU1, rf on DU5.

        table_results_list = []
        all_dep_hit_list = []
        for dep in range(5):
            file_name = "saved_models/exp_eval_results/exp_" + str(exp) + "_depth_" + str(dep) + ".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            table_results_list.append(result_dict["pred_hit_list"])
            all_dep_hit_list.extend(result_dict["pred_hit_list"])

        table_results_list.append(all_dep_hit_list)

        acc_line = ["{:.1f}".format(sum(hit_list) / len(hit_list) * 100) for hit_list in table_results_list]
        table_row_to_show = "EVR1 & " + " & ".join(acc_line)
        print(table_row_to_show)

    print("RT du1:", sum(np.array(rt_du1_baseline_acc)*np.array(bird_e_count))/sum(bird_e_count))
    print("RT du5:", sum(np.array(rt_du5_baseline_acc)*np.array(bird_e_count))/sum(bird_e_count))

    print("PR du5:", sum(np.array(pr_baseline_acc)*np.array(bird_e_count))/sum(bird_e_count))

    cnt_list= [len(hit_list) for hit_list in table_results_list]
    print("cnt " + " & ".join([str(cnt) for cnt in cnt_list]))

def generate_result_table_4():

    for exp in range(6,7,1):

        list_all_possible_proof = []
        list_supported_proof = []
        list_correct_proof = []

        for dep in range(5):
            file_name = "saved_models/exp_eval_results/exp_"+str(exp)+"_depth_"+str(dep)+".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            result_dict_list = result_dict["instance_output"]

            count_all_possible_proof = 0
            count_supported_proof = 0
            count_correct_proof = 0

            for instance_result_dict in result_dict_list:
                if instance_result_dict["strategy"]=="proof" \
                    or instance_result_dict["strategy"]=="inv-proof":

                    count_all_possible_proof += 1
                    if (instance_result_dict["strategy"]=="proof" and
                        instance_result_dict["label"]==True) or (instance_result_dict["strategy"]=="inv-proof" and
                        instance_result_dict["label"]==False):
                        count_supported_proof+=1
                        if instance_result_dict["proof_flag"]==1:
                            count_correct_proof+=1

            list_all_possible_proof.append(count_all_possible_proof)
            list_supported_proof.append(count_supported_proof)
            list_correct_proof.append(count_correct_proof)

        max_proof_accs = [list_supported_proof[i]/list_all_possible_proof[i]
                      for i in range(len(list_all_possible_proof))] + \
                        [sum(list_supported_proof)/sum(list_all_possible_proof)]
        proof_accs = [list_correct_proof[i]/list_all_possible_proof[i]
                      for i in range(len(list_all_possible_proof))]+ \
                        [sum(list_correct_proof)/sum(list_all_possible_proof)]

        line_to_show = ["{:.1f}".format(proof_acc*100) for proof_acc in proof_accs]
        line_to_show_max = ["{:.1f}".format(proof_acc*100) for proof_acc in max_proof_accs]
        print("MAX & "+" & ".join(line_to_show_max))
        print("EVR1 & "+" & ".join(line_to_show))
        print("cnt & "+ " & ".join([str(cnt) for cnt in  list_all_possible_proof]+ [str(sum(list_all_possible_proof))]))


def generate_result_table_3_alt():
    # 3 alt and 4 alt groups entries by dataset partition, as the other papers does.

    # the sequence: bird1, bird2, electric1, 2, 3, 4
    bird_e_count = [40,40,162,180,624, 4224]

    rt_du1_baseline_acc = [100.0,100.0,88.9,80.0, 93.9, 97.5]

    rt_du5_baseline_acc = [97.5, 100.0, 96.9, 98.3, 91.8, 76.7]

    pr_baseline_acc = [95.0,95.0, 100.0, 100.0, 89.7, 84.8 ]

    instances_partitions_all_depth = be_depth_to_dataset_partition()

    exp_names = ["EVR1 DU1", "EVR2 DU1", "EVR3 DU1", "EVR1 DU5", "EVR1 DU1-5"]
    for exp_idx, exp in enumerate([6, 9, 10, 8, 11]):
        # exp 6: EVR1 trained on DU1
        # exp 9: EVR2 trained on DU1
        # exp 10: EVR3 trained on DU1
        # exp 8: EVR1 trained on DU5
        # exp 11: EVR1, controlled on DU1, rf on DU5.

        partition_names = ["Bird1", "Bird2", "RB1", "RB2", "RB3", "RB4"]
        all_partitions_result_dict = {"Bird1":[], "Bird2":[], "RB1":[], "RB2":[], "RB3":[], "RB4":[], "all": []}

        table_results_list = []
        all_dep_hit_list = []
        for dep in range(5):
            file_name = "saved_models/exp_eval_results/exp_" + str(exp) + "_depth_" + str(dep) + ".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            assert(len(instances_partitions_all_depth[dep]) == len(result_dict["pred_hit_list"]))
            for ins_idx in range(len(result_dict["pred_hit_list"])):
                all_partitions_result_dict[instances_partitions_all_depth[dep][ins_idx]].append(result_dict["pred_hit_list"][ins_idx])

        for partition_name in partition_names:
            all_partitions_result_dict["all"].extend(all_partitions_result_dict[partition_name])

        acc_line = ["{:.1f}".format(sum(all_partitions_result_dict[p_name]) / len(all_partitions_result_dict[p_name]) * 100) for p_name in partition_names+["all"]]
        table_row_to_show = exp_names[exp_idx] + " & " + " & ".join(acc_line)
        print(table_row_to_show)

    rt_du1_all = sum(np.array(rt_du1_baseline_acc) * np.array(bird_e_count)) / sum(bird_e_count)
    rt_du5_all = sum(np.array(rt_du5_baseline_acc) * np.array(bird_e_count)) / sum(bird_e_count)

    pr_du5_all = sum(np.array(pr_baseline_acc) * np.array(bird_e_count)) / sum(bird_e_count)

    print("RT du1:", " & ".join([str(x) for x in rt_du1_baseline_acc] + [str(rt_du1_all)]))
    print("RT du5:", " & ".join([str(x) for x in rt_du5_baseline_acc] + [str(rt_du5_all)]))

    print("PR du5:", " & ".join([str(x) for x in pr_baseline_acc] + [str(pr_du5_all)]))

    print("cnt list:", " & ".join([str(x) for x in bird_e_count] + [str(sum(bird_e_count))]))

def generate_result_table_4_alt():

    instances_partitions_all_depth = be_depth_to_dataset_partition()
    exp_names = ["EVR1 DU1", "EVR2 DU1", "EVR3 DU1", "EVR1 DU5", "EVR1 DU1-5"]

    for exp_idx, exp in enumerate([6, 9, 10, 8, 11]):

        partition_names = ["Bird1", "Bird2", "RB1", "RB2", "RB3", "RB4"]
        all_proofs_dict = {"Bird1": [], "Bird2": [], "RB1": [], "RB2": [], "RB3": [], "RB4": [], "all": []}
        supported_proofs_dict = {"Bird1": [], "Bird2": [], "RB1": [], "RB2": [], "RB3": [], "RB4": [], "all": []}
        correct_proofs_dict = {"Bird1": [], "Bird2": [], "RB1": [], "RB2": [], "RB3": [], "RB4": [], "all": []}

        list_all_possible_proof = []
        list_supported_proof = []
        list_correct_proof = []

        for dep in range(5):
            file_name = "saved_models/exp_eval_results/exp_"+str(exp)+"_depth_"+str(dep)+".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            result_dict_list = result_dict["instance_output"]

            count_all_possible_proof = 0
            count_supported_proof = 0
            count_correct_proof = 0

            for ins_idx, instance_result_dict in enumerate(result_dict_list):
                if instance_result_dict["strategy"]=="proof" \
                    or instance_result_dict["strategy"]=="inv-proof":

                    all_proofs_dict[instances_partitions_all_depth[dep][ins_idx]].append(1)

                    if instance_result_dict["proof_flag"]==1:
                        correct_proofs_dict[instances_partitions_all_depth[dep][ins_idx]].append(1)
                    else:
                        correct_proofs_dict[instances_partitions_all_depth[dep][ins_idx]].append(0)
                    # count_all_possible_proof += 1
                    # if (instance_result_dict["strategy"]=="proof" and
                    #     instance_result_dict["label"]==True) or (instance_result_dict["strategy"]=="inv-proof" and
                    #     instance_result_dict["label"]==False):
                    #     count_supported_proof+=1
                    #     if instance_result_dict["proof_flag"]==1:
                    #         count_correct_proof+=1

                else:
                    all_proofs_dict[instances_partitions_all_depth[dep][ins_idx]].append(0)

        for partition_name in partition_names:
            all_proofs_dict["all"].extend(all_proofs_dict[partition_name])
            correct_proofs_dict["all"].extend(correct_proofs_dict[partition_name])

        proof_accs = [sum(correct_proofs_dict[partition_name])/sum(all_proofs_dict[partition_name]) for partition_name in partition_names + ["all"]]
        cnt = [sum(all_proofs_dict[partition_name]) for partition_name in partition_names + ["all"]]

        line_to_show = ["{:.1f}".format(proof_acc*100) for proof_acc in proof_accs]

        # line_to_show_max = ["{:.1f}".format(proof_acc*100) for proof_acc in max_proof_accs]
        # print("MAX & "+" & ".join(line_to_show_max))
        print(exp_names[exp_idx] +" & "+" & ".join(line_to_show))

    print("cnt & "+ " & ".join([str(x) for x in cnt]))

def generate_result_table_5():

    for exp in [4,5,1]:

        table_results_list = []
        all_dep_hit_list = []
        for dep in range(6):
            file_name = "saved_models/exp_eval_results/exp_"+str(exp)+"_depth_"+str(dep)+".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            table_results_list.append(result_dict["pred_hit_list"])
            all_dep_hit_list.extend(result_dict["pred_hit_list"])

        table_results_list.append(all_dep_hit_list)

        acc_line = ["{:.1f}".format(sum(hit_list)/len(hit_list)*100) for hit_list in table_results_list]
        table_row_to_show =  "EVR"+str(exp)+" & "+" & ".join(acc_line)
        print(table_row_to_show)

def generate_result_table_6():

    for exp in [4,5,1]:

        list_all_possible_proof = []
        list_supported_proof = []
        list_correct_proof = []

        for dep in range(6):
            file_name = "saved_models/exp_eval_results/exp_"+str(exp)+"_depth_"+str(dep)+".pickle"
            with open(file_name, "rb") as handle:
                result_dict = pickle.load(handle)

            result_dict_list = result_dict["instance_output"]

            count_all_possible_proof = 0
            count_supported_proof = 0
            count_correct_proof = 0

            for instance_result_dict in result_dict_list:
                if instance_result_dict["strategy"]=="proof" \
                    or instance_result_dict["strategy"]=="inv-proof":

                    count_all_possible_proof += 1
                    if (instance_result_dict["strategy"]=="proof" and
                        instance_result_dict["label"]==True) or (instance_result_dict["strategy"]=="inv-proof" and
                        instance_result_dict["label"]==False):
                        count_supported_proof+=1
                        if instance_result_dict["proof_flag"]==1:
                            count_correct_proof+=1

            list_all_possible_proof.append(count_all_possible_proof)
            list_supported_proof.append(count_supported_proof)
            list_correct_proof.append(count_correct_proof)

        max_proof_accs = [list_supported_proof[i]/list_all_possible_proof[i]
                      for i in range(len(list_all_possible_proof))] + \
                        [sum(list_supported_proof)/sum(list_all_possible_proof)]
        proof_accs = [list_correct_proof[i]/list_all_possible_proof[i]
                      for i in range(len(list_all_possible_proof))]+ \
                        [sum(list_correct_proof)/sum(list_all_possible_proof)]

        line_to_show = ["{:.1f}".format(proof_acc*100) for proof_acc in proof_accs]
        print("EVR"+str(exp)+" & "+" & ".join(line_to_show))

    print("cnt "+ " & ".join([str(cnt) for cnt in list_all_possible_proof]+[str(sum(list_all_possible_proof))]))

#generate_result_table_1()
#generate_result_table_2()

#generate_result_table_3_alt()
generate_result_table_4_alt()

#generate_result_table_4()
#generate_result_table_5()
#generate_result_table_3()
#be_depth_to_dataset_partition(True)





