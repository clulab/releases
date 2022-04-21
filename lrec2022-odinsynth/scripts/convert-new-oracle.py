import odinson.ruleutils
import json
import os
from collections import defaultdict

# create the vocabulary from the sentences
def build_vocab(sentences, fields_included=["word", "tag", "lemma"]):
    # vocab = Dict[Text, List[Text]]
    vocab = defaultdict(set)
    # vocab[field].add(smth)
    for s in sentences:
        for field in s["fields"]:
            if field["name"] in fields_included:
                for token in field["tokens"]:
                    vocab[field["name"]].add(token)
    return vocab


 #
def build_steps_list(gold_path):
    steps = []
    for i in range(len(gold_path) - 1):
        tmp = {"current_rule": "", "next_correct_rule": "", "next_incorrect_rules": []}
        tmp["current_rule"] = str(gold_path[i])
        tmp["next_correct_rule"] = str(gold_path[i + 1])
        steps.append(json.dumps(tmp, ensure_ascii=False))
    return steps


files = os.listdir("specs")

for spec_name in files:
    # open specs file
    j = open("specs/" + spec_name)
    file_content = j.readlines()
    j.close()
    odinson_document = json.loads(file_content[1])
    sentences = odinson_document["sentences"]

    # generate vocab
    vocab = build_vocab(sentences)

    # open the same file for the step
    step_name = spec_name[:-10] + ".steps.json"
    # open steps file
    j = open("steps/" + step_name)
    steps = j.readlines()
    j.close()
    last_step = json.loads(steps[-1])

    # get the last line
    rule_generated = last_step["next_correct_rule"]
    parsed_rule = odinson.ruleutils.parse_odinson_query(rule_generated)
    # generate next steps with the oracle
    gold_paths = odinson.ruleutils.all_paths_from_root(parsed_rule, vocab)

    # print the steps to the file in a separate folder
    steps_s = [build_steps_list(gold_paths[i]) for i in range(len(gold_paths))]
    steps = [item for sublist in steps_s for item in sublist]
    # flatten steps

    # save the step lists to the file
    j = open("v2.0-steps/" + step_name, "w")
    j.write("\n".join(steps) + "\n")
    j.close()
