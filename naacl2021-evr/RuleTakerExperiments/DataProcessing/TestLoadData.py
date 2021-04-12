from LoadData import loadAsSingleTasks, loadAsMultiTask
from LoadDataStandard import RuleTakerParsedInstances
import random
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from PrepareBatch import RuleTakerDataset, PadCollate, RuleTakerDatasetT5Standard, PadCollateT5Standard

import sys
from pathlib import Path
import argparse
import json

def print_json_item(json_item):

    all_facts = [triple[0]+": "+triple[1]["text"] for triple in list(json_item["triples"].items())]
    all_rules = [rule[0]+": "+rule[1]["text"] for rule in list(json_item["rules"].items())]

    all_questions = [q[0]+": "+q[1]["question"]+"  "+ str(q[1]["answer"])+"\n\t\t"+q[1]["proofs"] for q in list(json_item["questions"].items())]

    # Proofs is a string.
    #sample_proofs = list(json_item["questions"].items())[2][1]["proofs"]
    #print(select_proof_with_one_step_inference(sample_proofs))

    #question = json_item["questions"]["Q1"]["question"]

    #proofs = json_item["questions"]["Q1"]["proofs"]

    print("="*20)
    print("facts:")
    for fact in all_facts:
        print("\t"+fact)

    print("rules:")
    for rule in all_rules:
        print("\t" + rule)

    print("questions:")
    for question in all_questions:
        print("\t"+question)

    return 0


def testLoadAsSingleTasks():
    random.seed(0)

    instances = loadAsSingleTasks(fact_buffer_size = 5, rule_buffer_size = 3, train_amount_option = "70k", train_depth = "5")

    n_sample_count = 0
    for i in range(1,13,1):
        n_sample_count+=len(instances["train"]["pattern"+str(i)])

    print("total number of training samples:", n_sample_count)

    longest_input = 0
    for split in ["train", "dev"]: #["train", "dev", "test"]:
        #for pattern in ["pattern1", "pattern2", "pattern3", "pattern4"]:
        for pattern in ["pattern"+str(i) for i in range(4, 5, 1)]:
            print("number of instance in "+ pattern +"  "+split+":"+str(len(instances[split][pattern])))
            #for selected_instance in random.sample(instances[split][pattern], 2):
            for instance_idx, selected_instance in enumerate(instances[split][pattern]):
                # if " or " in selected_instance["output"].lower() or \
                #         " and " in selected_instance["output"].lower() or \
                #         " not " in selected_instance["input"].split(". ")[1]:
                # if " and " in selected_instance["output"].lower() and \
                #     " not " in selected_instance["input"].split(". ")[1]:
                #if " )and( " in selected_instance["output"].lower():
                #if "something" in selected_instance["input"] or "someone" in selected_instance["input"]:
                #if "according to rule" in selected_instance["output"]:
                if instance_idx<50:
                    print_json_item(selected_instance['item'])
                    print("-"*20)
                    print("split: %s, pattern: %s" % (split, pattern))
                    print("input: %s" % (selected_instance["input"]))
                    print("output: %s" % (selected_instance["output"]))
                    input("----------")
                else:
                    break

            #     input_length = len(selected_instance["input"].split(" "))
            #     if input_length>longest_input:
            #         longest_input = input_length
            #
            # print("longest input of ", pattern, ": ", longest_input)

    return 0

def testLoadMultiTask():
    instances = loadAsMultiTask()

    for split in ["train", "dev", "test"]:
        if split=="train":
            for selected_instance in random.sample(instances[split], 10):
                print("=" * 20)
                print("split: %s" % (split))
                print("input: %s" % (selected_instance["input"]))
                print("output: %s" % (selected_instance["output"]))
                input("-" * 20)
        else:
            for pattern in ["pattern1", "pattern2", "pattern3", "pattern4"]:
                for selected_instance in random.sample(instances[split][pattern], 2):
                    print("=" * 20)
                    print("split: %s, pattern: %s" % (split, pattern))
                    print("input: %s" % (selected_instance["input"]))
                    print("output: %s" % (selected_instance["output"]))
                    input("-" * 20)

    return 0


def test_prepare_batch():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    instances = loadAsSingleTasks()
    train_pairs = []
    for pattern in ["pattern" + str(i) for i in range(1, 9, 1)]:
        train_pairs.extend(instances['train'][pattern])

    ruletaker_train_dataset = RuleTakerDataset(train_pairs)
    ruletaker_train_dataloader  = DataLoader(ruletaker_train_dataset, batch_size=2,
                        shuffle=True, num_workers=1, collate_fn=PadCollate(tokenizer))

    for batch in ruletaker_train_dataloader:
        print("="*10)
        print(batch)
        input("A")

def test_load_parsed_instances():
    testDataset = RuleTakerParsedInstances()
    for depth in [0, 1, 2, 3, 5]:
        for split in ["train", "dev", "test"]:
            for instance in testDataset.instances["depth-" + str(depth)][split]:
                print(instance)
                input("-" * 20)

def test_prepare_batch_t5_standard():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    ruleTakerInstances = RuleTakerParsedInstances()

    ruletaker_train_dataset = RuleTakerDatasetT5Standard(ruleTakerInstances.instances["depth-0"]["train"])
    ruletaker_train_dataloader = DataLoader(ruletaker_train_dataset, batch_size=2,
                                            shuffle=True, num_workers=1, collate_fn=PadCollateT5Standard(tokenizer))

    for batch in ruletaker_train_dataloader:
        print("="*10)
        print(batch)
        input("A")

def check_du1_data():
    data_folder_path = str(Path('.').absolute().parent.parent) + "/Data"

    chaining_data_folder_path = data_folder_path + "/rule-reasoning-dataset-V2020.2.4"

    with open(chaining_data_folder_path + "/depth-5/meta-train.jsonl", "r") as f:
        raw_jsons = list(f)

    n_questions = 0
    for raw_json in raw_jsons:
        json_item = json.loads(raw_json)
        n_questions+=len(json_item["questions"])

    print("total number of training samples in du1:", n_questions)

def output_data_for_paper_submission():
    instances = loadAsSingleTasks(fact_buffer_size = 5, rule_buffer_size = 3, train_amount_option = "70k")

    with open('/Users/zhengzhongliang/NLP_Research/2020_ThinkInNaturalLanguage/'
              'ThinkInNaturalLanguage/RuleTakerExperiments/Experiments/saved_models/EVR_train_data.txt', 'w') \
            as outfile:
        json.dump(instances, outfile)

#print(output_data_for_paper_submission())


#check_du1_data()


testLoadAsSingleTasks()
#testLoadMultiTask()

#test_prepare_batch()

#test_prepare_batch_t5_standard()

