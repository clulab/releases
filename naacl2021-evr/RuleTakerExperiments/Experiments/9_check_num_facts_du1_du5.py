import pickle
import json
import numpy as np

def check_fact_buffers_in_train(debug_flag = False):
    data_file_paths = ["/Users/zhengzhongliang/NLP_Research/2020_ThinkInNaturalLanguage/ThinkInNaturalLanguage/Data/rule-reasoning-dataset-V2020.2.4/depth-1/meta-train.jsonl",
                       "/Users/zhengzhongliang/NLP_Research/2020_ThinkInNaturalLanguage/ThinkInNaturalLanguage/Data/rule-reasoning-dataset-V2020.2.4/depth-5/meta-train.jsonl"]

    for data_file_path in data_file_paths:
        with open(data_file_path, "r") as f:
            raw_jsons = list(f)

        count_dict = {}
        count_dict_facts = {}

        for raw_json in raw_jsons:
            item = json.loads(raw_json)

            n_fact = int(item["NFact"])


            n_fact_buffer = int(n_fact/5) + 1

            question_tuples = list(item["questions"].items())

            for question_tuple in question_tuples:
                if n_fact_buffer not in count_dict:
                    count_dict[n_fact_buffer] = 1
                else:
                    count_dict[n_fact_buffer] += 1

                if n_fact not in count_dict_facts:
                    count_dict_facts[n_fact] = 1

        print(count_dict)
        print(count_dict_facts)



check_fact_buffers_in_train()