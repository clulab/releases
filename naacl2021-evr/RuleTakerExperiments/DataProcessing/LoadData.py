import sys
from pathlib import Path
import argparse

data_folder_path = str(Path('.').absolute().parent.parent)+"/Data"

import json
import random
import re
import math

class DataConstants():
    def __init__(self):
        self.animal_entities = {"bear":0, "mouse":0, "squirrel":0, "cow":0, "bald eagle":0, "lion":0, "dog":0, "tiger":0, "cat":0, "rabbit":0}

    def get_animal_entities(self):
        return self.animal_entities

def loadPattern1(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    # Model input example:
    # episodic buffer 1
    # episodic buffer 2 (goal)

    # Model output example:
    # generate subgoals

    # 1, get how many facts and rules in the data
    # 2, get the true answer
    # 3, convert the input to episodic buffer X + goal:

    n_fact_buffer = math.ceil(int(num_facts)/fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules)/rule_buffer_size)

    instance = {"input": "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+
                         " rule buffers. episodic buffer: i want to prove \""+ query[:-1] + "\". </s>",

                "output": "GENERATE_SUBGOALS </s>",

                "item":item}

    return instance

def loadPattern2(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    # Model input example:
    # episodic buffer 1
    # episodic buffer 2 (goal)
    # operator: generate subgoals

    # Model output example:
    # ouptut: I want to judge whether the facts can prove bob is green AND I want to judge whether the rules can prove bob is green

    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    if "not" not in query.split(" "):

        instance = {"input": "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+
                             " rule buffers. episodic buffer: i want to prove \""+ query[:-1] +
                             "\". operator: GENERATE_SUBGOALS </s>",

                    "output": "i want to judge whether the facts can prove \""+query[:-1]+
                              "\". OR i want to judge whether the rules can prove \""+query[:-1]+ "\". </s>",

                    "item":item}

    else:
        instance = {"input": "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " +
                             str(n_rule_buffer) + " rule buffers. episodic buffer: i want to prove \"" + query[:-1] +
                             "\". operator: GENERATE_SUBGOALS </s>",

                    "output": "i want to judge whether the facts do not contradict \"" + query[:-1] +
                              "\". AND i want to judge whether the rules do not contradict \"" + query[:-1]+ "\". </s>",

                    "item":item}

    return instance

def loadPattern3(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    # Model input example:
    # episodic buffer 1
    # episodic buffer 2: judge whether the facts can prove ...

    # Model output example:
    # ouptut: GET(FACTS_BUFFER) THEN RUN(EPISODIC_BUFFER)

    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    if "not" not in query.split(" "):
        instance = {"input":  "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+
                              " rule buffers. episodic buffer: i want to judge whether the facts can prove \"" +
                              query[:-1]+ "\". </s>",

                    "output": "GENERATE_SUBGOALS </s>",

                    "item":item}
    else:
        instance = {"input": "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " +
                             str(n_rule_buffer) +
                             " rule buffers. episodic buffer: i want to judge whether the facts do not contradict \"" +
                             query[:-1] + "\". </s>",

                    "output": "GENERATE_SUBGOALS </s>",

                    "item":item}

    return instance

def loadPattern4(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    if "not" not in query.split(" "):
        input_string = "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+ \
                       " rule buffers. episodic buffer: i want to judge whether the facts can prove \"" + query[:-1]+ \
                       "\". operator: GENERATE_SUBGOALS </s>"
        output_strings = []
        for fact_buffer_idx in range(n_fact_buffer):
            output_strings.append("i want to judge whether fact buffer "+str(fact_buffer_idx+1)+" can prove \"" +query[:-1]+ "\".")

        return {"input": input_string,
                "output": " OR ".join(output_strings) +" </s>" ,
                "item":item}
    else:
        input_string = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + str(n_rule_buffer) + \
                       " rule buffers. episodic buffer: i want to judge whether the facts do not contradict \"" + \
                       query[ :-1] + "\". operator: GENERATE_SUBGOALS </s>"

        output_strings = []
        for fact_buffer_idx in range(n_fact_buffer):
            output_strings.append(
                "i want to judge whether fact buffer " + str(
                            fact_buffer_idx+1) + " does not contradict \"" + query[:-1] + "\".")

        return {"input": input_string,
                "output": " AND ".join(output_strings)+" </s>",
                "item":item}

def loadPattern5(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    instances = []

    for fact_buffer_idx in range(n_fact_buffer):

        if "not" not in query.split(" "):
            instance = {"input":  "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+
                                  " rule buffers. episodic buffer: i want to judge whether fact buffer "+
                                  str(fact_buffer_idx+1)+" can prove \"" +query[:-1]+ "\". </s>",

                        "output": "GET(FACT_BUFFER_"+str(fact_buffer_idx+1)+") THEN RUN(EPISODIC_BUFFER, FACT_BUFFER_"+
                                  str(fact_buffer_idx+1)+") </s>",

                        "item":item}
        else:
            instance = {"input": "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " +
                                 str(n_rule_buffer) + " rule buffers. episodic buffer: i want to judge whether fact buffer " +
                                 str(fact_buffer_idx+1) + " does not contradict \"" + query[:-1] + "\". </s>",

                        "output": "GET(FACT_BUFFER_" + str(fact_buffer_idx+1) + ") THEN RUN(EPISODIC_BUFFER, FACT_BUFFER_" +
                                  str(fact_buffer_idx+1) + ") </s>",

                        "item":item}

        instances.append(instance)

    return instances

def loadPattern6(num_facts, num_rules, query_dict, item, data_constants, fact_buffer_size = 5, rule_buffer_size = 3):

    def search_by_entity_query_rep(query_rep, fact_reps, fact_buffer_start_index):
        if query_rep[-1] == "+":
            for idx, fact_rep in enumerate(fact_reps):
                if query_rep ==fact_rep:
                    return "true, this is confirmed by fact "+str(fact_buffer_start_index+idx+1)+". </s>"
            return "false, CWA. </s>"

        if query_rep[-1] in ["~", "-"]:
            for idx, fact_rep in enumerate(fact_reps):
                if query_rep[:-1] == fact_rep[:-1] and fact_rep[-1]=="+":
                    return "false, this is contradicted by fact "+str(fact_buffer_start_index+idx+1)+". </s>"
            return "true, NAF. </s>"

    def search_by_something_query_rep(query_rep, fact_reps, fact_buffer_start_index):
        if query_rep[-1] == "+":
            for idx, fact_rep in enumerate(fact_reps):
                if query_rep[1:] ==fact_rep[1:]:
                    return "true, this is confirmed by fact "+str(fact_buffer_start_index+idx+1)+". </s>"
            return "false, CWA. </s>"

        if query_rep[-1] in ["~", "-"]:
            return "true, NAF. </s>"

    def generate_text_from_rep(subj_in_text, precond_rep):
        if subj_in_text in data_constants.animal_entities:
            subj_in_text = "the " + subj_in_text

        verb_in_text = precond_rep[1]

        obj_in_text = "the " + precond_rep[2] if precond_rep[2] in data_constants.animal_entities else precond_rep[
            2]

        if precond_rep[-1] in ["~", "-"]:
            if precond_rep[1] == "is":
                text_to_add = subj_in_text + " is not " + obj_in_text.lower()
            else:
                text_to_add = subj_in_text + " does not " + verb_in_text + " " + obj_in_text.lower()
        else:
            text_to_add = subj_in_text + " " + verb_in_text.lower() + " " + obj_in_text.lower()

        return text_to_add

    all_facts = [triple[1]["text"].lower() for triple in item["triples"].items()]
    all_fact_reps = [triple[1]["representation"].lower()[2:-2].split("\" \"") for triple in item["triples"].items()]

    # in the rep of facts, + means positive and - means negative. no ~ appears in the reps.

    query_text = query_dict["question"].lower()
    query_rep = query_dict["representation"].lower()[2:-2].split("\" \"")

    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    instances = []

    for fact_buffer_idx in range(n_fact_buffer):

        fact_buffer_idx_start = fact_buffer_idx*fact_buffer_size
        fact_buffer_idx_end  = min(len(all_facts), (fact_buffer_idx+1)*fact_buffer_size)

        fact_buffer = " ".join(["fact "+str(fact_buffer_idx*fact_buffer_size+fact_idx+1)+": "+fact for fact_idx, fact in enumerate(all_facts[fact_buffer_idx_start:fact_buffer_idx_end])])
        if "not" not in query_text.split(" "):
            episodic_buffer = "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+ \
                              " rule buffers. episodic buffer: i want to judge whether fact buffer "+\
                              str(fact_buffer_idx+1)+" can prove \"" +query_text[:-1]+ "\". "


            output = search_by_entity_query_rep(query_rep,
                                                all_fact_reps[fact_buffer_idx_start:fact_buffer_idx_end],
                                                fact_buffer_idx_start)


        else:
            episodic_buffer = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + \
                              str(n_rule_buffer) + \
                              " rule buffers. episodic buffer: i want to judge whether fact buffer " + \
                              str(fact_buffer_idx+1) + " does not contradict \"" + query_text[:-1] + "\". "


            # If the positive literal is in the fact buffer, return false, else return true.
            output = search_by_entity_query_rep(query_rep,
                                                all_fact_reps[fact_buffer_idx_start:fact_buffer_idx_end],
                                                fact_buffer_idx_start)

        instance = {
            "input": episodic_buffer+fact_buffer+" operator: RUN </s>",
            "output": output,
            "item": item}

        instances.append(instance)


        # write script to handle something/someone in this pattern.
        if random.random()<0.05:
            new_subj = "something" if random.random()<0.5 else "someone"
            new_query_text = generate_text_from_rep(new_subj, query_rep)

            if "not" not in query_text.split(" "):
                episodic_buffer = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + str(
                    n_rule_buffer) + \
                                  " rule buffers. episodic buffer: i want to judge whether fact buffer " + \
                                  str(fact_buffer_idx + 1) + " can prove \"" + new_query_text + "\". "

                output = search_by_something_query_rep(query_rep,
                                                       all_fact_reps[fact_buffer_idx_start:fact_buffer_idx_end],
                                                       fact_buffer_idx_start)
            else:
                episodic_buffer = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + \
                                  str(n_rule_buffer) + \
                                  " rule buffers. episodic buffer: i want to judge whether fact buffer " + \
                                  str(fact_buffer_idx + 1) + " do not contradict \"" + new_query_text + "\". "

                output = search_by_something_query_rep(query_rep,
                                                       all_fact_reps[fact_buffer_idx_start:fact_buffer_idx_end],
                                                       fact_buffer_idx_start)

            instance = {
                "input": episodic_buffer + fact_buffer + " operator: RUN </s>",
                "output": output,
                "item": item}

            instances.append(instance)


    return instances

def loadPattern7(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    if "not" not in query.split(" "):
        instance = {"input": "episodic buffer: there are " + str(n_rule_buffer) + " rule buffers and " +
                             str(n_rule_buffer) +
                             " rule buffers. episodic buffer: i want to judge whether the rules can prove \"" +
                             query[:-1] + "\". </s>",

                    "output": "GENERATE_SUBGOALS </s>",

                    "item":item}

    else:
        instance = {"input": "episodic buffer: there are " + str(n_rule_buffer) + " rule buffers and " +
                             str(n_rule_buffer) +
                             " rule buffers. episodic buffer: i want to judge whether the rules do not contradict \"" +
                             query[:-1] + "\". </s>",

                    "output": "GENERATE_SUBGOALS </s>",

                    "item":item}

    return instance

def loadPattern8(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    if "not" not in query.split(" "):
        input_string = "episodic buffer: there are "+str(n_fact_buffer)+" fact buffers and "+str(n_rule_buffer)+ \
                       " rule buffers. episodic buffer: i want to judge whether the rules can prove \"" + query[:-1]+ \
                       "\". operator: GENERATE_SUBGOALS </s>"
        output_strings = []
        for rule_buffer_idx in range(n_rule_buffer):
            output_strings.append("i want to judge whether rule buffer "+str(rule_buffer_idx+1)+" can prove \"" +query[:-1]+ "\".")

        return {"input": input_string,
                "output": " OR ".join(output_strings)+" </s>",
                "item":item}
    else:
        input_string = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + str(n_rule_buffer) + \
                       " rule buffers. episodic buffer: i want to judge whether the rules do not contradict \"" + \
                       query[:-1] + "\". operator: GENERATE_SUBGOALS </s>"

        output_strings = []
        for rule_buffer_idx in range(n_rule_buffer):
            output_strings.append(
                "i want to judge whether rule buffer " + str(
                            rule_buffer_idx+1) + " does not contradict \"" + query[:-1] + "\".")

        return {"input": input_string,
                "output": " AND ".join(output_strings)+" </s>",
                "item":item}

def loadPattern9(num_facts, num_rules, query, item, fact_buffer_size = 5, rule_buffer_size = 3):
    n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
    n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

    instances = []

    for rule_buffer_idx in range(n_rule_buffer):

        if "not" not in query.split(" "):
            instance = {"input": "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + str(
                n_rule_buffer) + " rule buffers. episodic buffer: i want to judge whether rule buffer " + str(
                rule_buffer_idx+1) + " can prove \"" + query[:-1] + "\". </s>",
                        "output": "GET(RULE_BUFFER_" + str(
                            rule_buffer_idx+1) + ") THEN RUN(EPISODIC_BUFFER, RULE_BUFFER_" + str(
                            rule_buffer_idx+1) + ") </s>",
                        "item":item}
        else:
            instance = {"input": "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " +
                                 str(n_rule_buffer) +
                                 " rule buffers. episodic buffer: i want to judge whether rule buffer " +
                                 str(rule_buffer_idx+1) + " does not contradict \"" + query[:-1] + "\". </s>",
                        "output": "GET(RULE_BUFFER_" + str(
                            rule_buffer_idx+1) + ") THEN RUN(EPISODIC_BUFFER, RULE_BUFFER_" + str(
                            rule_buffer_idx+1) + ") </s>",
                        "item":item}

        instances.append(instance)

    return instances

def loadPattern10(num_facts, num_rules, json_item, data_constants, fact_buffer_size = 5, rule_buffer_size = 3):
    # Input: the episodic buffers (goal: whether the rules can prove Y), the phonological buffer with all rules, and the operator.
    # Output: No | Yes, according to rule X we need to prove.

    literal_extracting_pattern = r'"(.*?)"' # Anything in parenthesis.

    def parse_rule(rule_string):
        # This function is to parse the raw rule representation to a list of preconditions and an effect.
        preconditions_raw, effect_raw = rule_string.split("->")
        precondition_literals = [re.findall(literal_extracting_pattern, precondition_raw) for precondition_raw in preconditions_raw.split(") (")]
        effect_literal = re.findall(literal_extracting_pattern, effect_raw)

        return precondition_literals, effect_literal

    def generate_text_from_rep(subj_in_text, precond_rep):
        if subj_in_text in data_constants.animal_entities:
            subj_in_text = "the "+subj_in_text

        verb_in_text = precond_rep[1]

        obj_in_text = "the "+ precond_rep[2] if precond_rep[2] in data_constants.animal_entities else precond_rep[2]

        if precond_rep[-1] in ["~", "-"]:
            if precond_rep[1]=="is":
                text_to_add = subj_in_text+" is not "+ obj_in_text.lower()
            else:
                text_to_add = subj_in_text + " does not " + verb_in_text + " " + obj_in_text.lower()
        else:
            text_to_add = subj_in_text+" "+verb_in_text.lower()+" "+obj_in_text.lower()

        return text_to_add

    def return_matched_literals(question_features, rules_features, rule_buffer_idx_offset):

        matched_literals_all_rules = []

        # question_features: subj, verb, obj, polarity
        # rule_features: [preconditions, effect]
        if question_features[-1]=="+":
            for i, rule_features in enumerate(rules_features):
                preconditions_to_prove = []
                # Matched rule pattern 1: when the query start with something/someone, match the verb, object and polairty
                if question_features[0] in ["something", "someone"] and \
                    rule_features[-1][1:]==question_features[1:]:

                    for precondition in rule_features[0]:
                        preconditions_to_prove.append(generate_text_from_rep(precondition[0], precondition))
                    matched_literals_all_rules.append((rule_buffer_idx_offset+i, preconditions_to_prove))


                # Matched rule pattern 2: when the query start with entity, and the rule effect starts with something/someone,
                # match the verb, object and polarity.
                elif question_features[0] not in ["something", "someone"] and \
                    rule_features[-1][0] in ["something", "someone"] and \
                    rule_features[-1][1:] == question_features[1:]:

                    for precondition in rule_features[0]:
                        if precondition[0] in ["something", "someone"]:
                            preconditions_to_prove.append(generate_text_from_rep(question_features[0], precondition))
                        else:
                            preconditions_to_prove.append(generate_text_from_rep(precondition[0], precondition))
                    matched_literals_all_rules.append((rule_buffer_idx_offset+i, preconditions_to_prove))


                # Matched rule patter 3: when the query start with entity, and the rule effect starts with entity,
                # match the subj, verb, obj, and polarity
                elif question_features[0] not in ["something", "someone"] and \
                    rule_features[-1][0] not in ["something", "someone"] and \
                    rule_features[-1] == question_features:

                    for precondition in rule_features[0]:
                        preconditions_to_prove.append(generate_text_from_rep(precondition[0], precondition))
                    matched_literals_all_rules.append((rule_buffer_idx_offset+i, preconditions_to_prove))

        else:
            for i, rule_features in enumerate(rules_features):
                preconditions_to_prove = []
                # Matched rule pattern 1: when the query start with something/someone, match the verb, object and polairty
                if question_features[0] in ["something", "someone"] and \
                        rule_features[-1][1:-1] == question_features[1:-1] and \
                        rule_features[-1][-1]== "+":

                    for precondition in rule_features[0]:
                        preconditions_to_prove.append(generate_text_from_rep(precondition[0], precondition))
                    matched_literals_all_rules.append((rule_buffer_idx_offset+i, preconditions_to_prove))

                # Matched rule pattern 2: when the query start with entity, and the rule effect starts with something/someone,
                # match the verb, object and polarity.
                elif question_features[0] not in ["something", "someone"] and \
                        rule_features[-1][0] in ["something", "someone"] and \
                        rule_features[-1][1:-1] == question_features[1:-1] and \
                        rule_features[-1][-1]== "+":

                    for precondition in rule_features[0]:
                        if precondition[0] in ["something", "someone"]:
                            preconditions_to_prove.append(generate_text_from_rep(question_features[0], precondition))
                        else:
                            preconditions_to_prove.append(generate_text_from_rep(precondition[0], precondition))
                    matched_literals_all_rules.append((rule_buffer_idx_offset+i, preconditions_to_prove))


                # Matched rule patter 3: when the query start with entity, and the rule effect starts with entity,
                # match the subj, verb, obj, and polarity
                elif question_features[0] not in ["something", "someone"] and \
                        rule_features[-1][0] not in ["something", "someone"] and \
                        rule_features[-1][:-1] == question_features[:-1] and \
                        rule_features[-1][-1]== "+":

                    for precondition in rule_features[0]:
                        preconditions_to_prove.append(generate_text_from_rep(precondition[0], precondition))
                    matched_literals_all_rules.append((rule_buffer_idx_offset+i, preconditions_to_prove))

        return matched_literals_all_rules

    def generate_output_string(question_features, rules_features, rule_buffer_idx_offset):
        matched_literals = return_matched_literals(question_features, rules_features, rule_buffer_idx_offset)
        # Note that both "~" and "-" means negation.

        if len(matched_literals) == 0:
            if question_features[-1] in ["-","~"]:
                output_string = "true"
            else:
                output_string = "false"
        else:
            if question_features[-1] in ["-", "~"]:
                if question_features[0] in ["something", "someone"]:
                    output_string = "true"
                else:
                    output_string = " )AND( ".join(
                        ["according to rule " + str(rule_index + 1) + ", I need to prove " + " and ".join(
                            one_rule_matched_literals)
                         for (rule_index, one_rule_matched_literals) in matched_literals]) + "."
            else:
                output_string = " OR ".join(
                    ["according to rule " + str(rule_index+1) + ", I need to prove " + " and ".join(one_rule_matched_literals)
                     for (rule_index, one_rule_matched_literals) in matched_literals]) + "."

        return output_string, matched_literals

    facts_list = [triple[1]["text"].lower() for triple in json_item["triples"].items()]
    rules_list = [rule[1]["text"].lower() for rule in json_item["rules"].items()]

    rule_reps = [rule_tuple[1]["representation"] for rule_tuple in list(json_item["rules"].items())]

    rules_features = [parse_rule(rule_string) for rule_string in
                      rule_reps]  # each element is a (preconditions, effect) tuple.

    instances_list = []
    for i, question in enumerate(list(json_item["questions"].items())):
        query = question[1]["question"].lower()

        n_fact_buffer = math.ceil(int(num_facts) / fact_buffer_size)
        n_rule_buffer = math.ceil(int(num_rules) / rule_buffer_size)

        for rule_buffer_idx in range(n_rule_buffer):
            if "not" in query.split(" "):
                episodic_buffer_input = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + str(
                    n_rule_buffer) + " rule buffers. episodic buffer: i want to judge whether rule buffer " + \
                                        str(rule_buffer_idx + 1) + " does not contradict \"" + query[:-1] + "\". "
            else:
                episodic_buffer_input = "episodic buffer: there are " + str(n_fact_buffer) + " fact buffers and " + str(
                    n_rule_buffer) + " rule buffers. episodic buffer: i want to judge whether rule buffer "+\
                                        str(rule_buffer_idx+1)+" can prove \"" + query[:-1] + "\". "

            rule_buffer_idx_start = rule_buffer_idx * rule_buffer_size
            rule_buffer_idx_end = min(len(rules_list), (rule_buffer_idx + 1) * rule_buffer_size)

            rules_list_ = rules_list[rule_buffer_idx_start:rule_buffer_idx_end]
            rules_features_ = rules_features[rule_buffer_idx_start:rule_buffer_idx_end]

            phonological_buffer_input = " ".join(
                ["rule " + str(rule_buffer_idx*rule_buffer_size+r_idx+1) + ": " + rule + " " for r_idx, rule in enumerate(rules_list_)])
            operator_input = "operator: RUN"

            question_rep = question[1]["representation"]
            question_features = re.findall(literal_extracting_pattern, question_rep)
            output_string, matched_literals_raw = generate_output_string(question_features, rules_features_, rule_buffer_idx_start)

            #if len(matched_literals_raw)>0:
            instances_list.append({"input": episodic_buffer_input + phonological_buffer_input + operator_input+ " </s>",
                                   "output": output_string.lower() + " </s>", "output_raw":matched_literals_raw,
                                   "item":json_item, "query":query})

            if random.random()<0.05:
                subj_in_text = "something" if random.random()<0.5 else "someone"
                new_question_text = generate_text_from_rep(subj_in_text, question_features)
                new_question_features = question_features.copy()
                new_question_features[0] = subj_in_text

                output_string, matched_literals_raw = generate_output_string(new_question_features, rules_features_, rule_buffer_idx_start)

                if "not" in query.split(" "):
                    episodic_buffer_input = "episodic buffer: there are " + str(
                        n_fact_buffer) + " fact buffers and " + str(
                        n_rule_buffer) + " rule buffers. episodic buffer: i want to judge whether rule buffer " + \
                                            str(rule_buffer_idx + 1) + " does not contradict \"" + new_question_text + "\". "
                else:
                    episodic_buffer_input = "episodic buffer: there are " + str(
                        n_fact_buffer) + " fact buffers and " + str(
                        n_rule_buffer) + " rule buffers. episodic buffer: i want to judge whether rule buffer " + \
                                            str(rule_buffer_idx + 1) + " can prove \"" + new_question_text + "\". "

                # if len(matched_literals_raw)>0:
                instances_list.append(
                    {"input": episodic_buffer_input + phonological_buffer_input + operator_input + " </s>",
                     "output": output_string.lower() + " </s>", "output_raw": matched_literals_raw,
                     "item": json_item, "query": query})

    return instances_list

def loadPattern11(pattern10_instance):
    # Input: the episodic buffer (10 facts and 4 rules, and according to rule X ....)
    # Output: generate subgoals

    extract_episodic_pattern = "episodic.*?rule buffers\. "
    episodic_buffer_1 = re.findall(extract_episodic_pattern, pattern10_instance["input"])[0]
    episodic_buffer_2 = "episodic buffer: "+pattern10_instance["output"]
    output_string = "GENERATE_SUBGOALS"

    return {"input":episodic_buffer_1+episodic_buffer_2,
            "output":output_string+ " </s>",
            "item":pattern10_instance["item"]}

def loadPattern12(pattern10_instance):
    # Input: the episodic buffer (10 facts and 4 rules, and according to rule X ....) and generate subgoals
    # Output: the actual subgoal
    extract_episodic_pattern = "episodic.*?rule buffers\. "
    episodic_buffer_1 = re.findall(extract_episodic_pattern, pattern10_instance["input"])[0]
    episodic_buffer_2 = "episodic buffer: " + pattern10_instance["output"][:-4]
    operator = " operator: GENERATE_SUBGOALS"

    if "not" in pattern10_instance["query"].split(" "):
        output_string = " )AND( ".join([" AND ".join(["i want to prove \""+ matched_literal.lower()+"\"." for matched_literal in matched_literals]) for (rule_idx, matched_literals) in pattern10_instance["output_raw"]])
    else:
        output_string = " OR ".join([" AND ".join(["i want to prove \""+ matched_literal.lower()+"\"." for matched_literal in matched_literals]) for (rule_idx, matched_literals) in pattern10_instance["output_raw"]])

    return {"input":episodic_buffer_1+episodic_buffer_2+operator+ " </s>",
            "output":output_string+ " </s>",
            "item":pattern10_instance["item"]}

def loadAsSingleTasks(chaining_data_folder_path = data_folder_path+"/rule-reasoning-dataset-V2020.2.4",
                      fact_buffer_size = 5, rule_buffer_size = 3, train_amount_option = "70k", train_depth = "1"):

    assert(train_depth in ["0", "1", "2", "3", "5"])
    print("use training depth ", train_depth)

    data_constants = DataConstants()

    instances = {"train":{}, "dev":{}, "test":{}}
    for split in ["train","dev", "test"]:
        for pattern_num in range(1,13,1):
            instances[split]["pattern"+str(pattern_num)] = []

    for split in ["train", "dev", "test"]:
        with open(chaining_data_folder_path+"/depth-"+ train_depth + "/meta-"+split+".jsonl", "r") as f:
            raw_jsons = list(f)

        all_json_items = []
        if split == "train":
            if train_amount_option=="10k":
                random.shuffle(raw_jsons)
                n_sample_count = 0
                for raw_json in raw_jsons:
                    item = json.loads(raw_json)
                    all_json_items.append(item)
                    n_sample_count+=len(item["questions"])
                    if n_sample_count>10000:
                        break
                print("train amount option:", train_amount_option," number of original training samples:", n_sample_count)

            elif train_amount_option=="30k":
                random.shuffle(raw_jsons)
                n_sample_count = 0
                for raw_json in raw_jsons:
                    item = json.loads(raw_json)
                    all_json_items.append(item)
                    n_sample_count += len(item["questions"])
                    if n_sample_count > 30000:
                        break
                print("train amount option:", train_amount_option," number of original training samples:", n_sample_count)

            else:  # use all training data, i.e., 70k.
                n_sample_count = 0
                for raw_json in raw_jsons:
                    item = json.loads(raw_json)
                    all_json_items.append(item)
                    n_sample_count += len(item["questions"])
                print("train amount option:", train_amount_option," number of original training samples:", n_sample_count)

        else:
            for raw_json in raw_jsons:
                item = json.loads(raw_json)
                all_json_items.append(item)

        for item in all_json_items:
            num_facts = int(item["NFact"])
            num_rules = int(item["NRule"])
            for query_dict in item["questions"].values():
                query = query_dict["question"].lower()

                instances[split]["pattern1"].append(loadPattern1(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))
                instances[split]["pattern2"].append(loadPattern2(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))

                instances[split]["pattern3"].append(loadPattern3(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))
                instances[split]["pattern4"].append(loadPattern4(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))
                instances[split]["pattern5"].extend(loadPattern5(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))
                instances[split]["pattern6"].extend(loadPattern6(num_facts, num_rules, query_dict, item, data_constants, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))

                instances[split]["pattern7"].append(loadPattern7(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))
                instances[split]["pattern8"].append(loadPattern8(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))
                instances[split]["pattern9"].extend(loadPattern9(num_facts, num_rules, query, item, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))

            instances[split]["pattern10"].extend(loadPattern10(num_facts, num_rules, item, data_constants, fact_buffer_size = fact_buffer_size, rule_buffer_size = rule_buffer_size))

        for instance in instances[split]["pattern10"]:
            if instance["output"][:5]!="false" and instance["output"][:4]!="true":
                instances[split]["pattern11"].append(loadPattern11(instance))
                instances[split]["pattern12"].append(loadPattern12(instance))

    print("number of generated training samples:",sum([len(instances["train"]["pattern"+str(pattern_num)]) for pattern_num in range(1,13,1)]))
    print("number of generated dev samples:",sum([len(instances["dev"]["pattern"+str(pattern_num)]) for pattern_num in range(1,13,1)]))

    return instances

def loadAsMultiTask(chaining_data_folder_path = data_folder_path+"/rule-reasoning-dataset-V2020.2.4/depth-1"):
    instances = {"train": [],
                 "dev": {"pattern1": [], "pattern2": [], "pattern3": [], "pattern4": []},
                 "test": {"pattern1": [], "pattern2": [], "pattern3": [], "pattern4": []}}

    for split in ["train", "dev", "test"]:
        with open(chaining_data_folder_path + "/meta-" + split + ".jsonl", "r") as f:
            raw_jsons = list(f)

        for raw_json in raw_jsons:
            item = json.loads(raw_json)
            num_facts = int(item["NFact"])
            num_rules = int(item["NRule"])
            query = item["questions"]["Q1"]["question"].lower()

            if split=="train":
                instances[split].append(loadPattern1(num_facts, num_rules, query))
                instances[split].append(loadPattern2(num_facts, num_rules, query))
                instances[split].append(loadPattern3(num_facts, num_rules, query))
                instances[split].append(loadPattern4(num_facts, num_rules, query))
            else:
                instances[split]["pattern1"].append(loadPattern1(num_facts, num_rules, query))
                instances[split]["pattern2"].append(loadPattern2(num_facts, num_rules, query))
                instances[split]["pattern3"].append(loadPattern3(num_facts, num_rules, query))
                instances[split]["pattern4"].append(loadPattern4(num_facts, num_rules, query))

    return instances


