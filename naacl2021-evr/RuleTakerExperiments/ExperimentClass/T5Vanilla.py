import sys
from pathlib import Path
import argparse

import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import optim
import torch
import os

import csv
import editdistance
import re

class T5Vanilla():
    def __init__(self, learning_rate, device):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_model.to(device)

        # This learning rate 0.0001 is used in one of the tutorial, but might not be the best choice.
        self.adamOpt = optim.Adam(self.t5_model.parameters(), lr=learning_rate)
        self.device = device

        # we can add at most 28 addition tokens without resizing the default embedding layer.
        self.dsl_tokens = ["AND","OR" "GET","GENERATE_SUBGOALS","RUN" ,"THEN","EPISODIC_BUFFER",
                           "FACT_BUFFER_1", "FACT_BUFFER_2","FACT_BUFFER_3","FACT_BUFFER_4","FACT_BUFFER_5",
                           "RULE_BUFFER_1", "RULE_BUFFER_2", "RULE_BUFFER_3", "RULE_BUFFER_4", "RULE_BUFFER_5",
                           "NAF", "CWA"]
        self.tokenizer.add_tokens(self.dsl_tokens)

        self.max_generation_length = 100

    def train_iters(self, train_pairs, n_iters, print_every=100):
        self.t5_model.train()

        print_loss_total = 0  # Reset every print_every

        # Training data is checked to be correct.
        training_pair_indices = [random.choice(range(len(train_pairs))) for i in range(n_iters)]

        for iter, idx in enumerate(training_pair_indices):
            self.adamOpt.zero_grad()

            training_pair = train_pairs[idx]
            input_tensor = self.tokenizer.encode(training_pair["input"], return_tensors="pt").to(self.device)
            target_tensor = self.tokenizer.encode(training_pair["output"], return_tensors="pt").to(self.device)

            outputs = self.t5_model(input_ids=input_tensor, labels=target_tensor)
            loss, prediction_scores = outputs[:2]

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("iter ", iter, " average loss:",print_loss_avg)
                print_loss_total = 0

    def train_iters_batch(self, data_loader, n_iters, print_every = 100):
        self.t5_model.train()

        print_loss_total = 0  # Reset every print_every

        iter = 0
        for e in range(10):
            for batch in data_loader:
                self.adamOpt.zero_grad()

                outputs = self.t5_model(input_ids=batch["input"]["input_ids"].to(self.device), attention_mask=batch["input"]["attention_mask"].to(self.device),
                                       labels=batch["output"]["input_ids"].to(self.device))

                loss, prediction_scores = outputs[:2]

                loss.backward()
                self.adamOpt.step()

                print_loss_total += loss.detach().cpu().numpy()
                iter += 1

                batch_size = batch["input"]["input_ids"]
                if iter*len(batch_size) % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print("iter ", iter, " batch size:", len(batch_size)," average loss:", print_loss_avg)
                    print_loss_total = 0

                if iter*len(batch_size)>=n_iters:
                    return 0

    def evaluate_iters(self, test_pairs, print_flag = False):
        #test_pairs = test_pairs[:40]

        self.t5_model.eval()
        abnormal_count = 0
        with torch.no_grad():
            for i in range(len(test_pairs)):
                input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)

                predicted_tensor = self.t5_model.generate(input_tensor, max_length = self.max_generation_length)
                predicted_text = self.tokenizer.decode(predicted_tensor[0])
                edit_distance = editdistance.eval(predicted_text.replace(" ",""), test_pairs[i]["output"][:-5].replace(" ",""))

                #if edit_distance>=0:
                    #abnormal_count+=1
                if print_flag:
                    print("\t"+"-"*30)
                    print("\tinput sent:"+test_pairs[i]["input"])
                    print("\ttarget sent:"+test_pairs[i]["output"])
                    print("\tpredict sent:"+predicted_text)
                    print("\tedit distance:"+str(edit_distance))

                    input("\tpress enter for the next example")

        print("n cases where edit dist>3 (out of %d): %d" %(len(test_pairs), abnormal_count))


    def evaluate_iters_batch(self, test_pairs, print_every = 200):
        self.t5_model.eval()

        total_loss = 0
        edit_distances = []
        with torch.no_grad():
            for i, batch in enumerate(test_pairs):

                predicted_tensors = self.t5_model.generate(input_ids=batch["input"]["input_ids"].to(self.device),
                                                           attention_mask=batch["input"]["attention_mask"].to(self.device), max_length = self.max_generation_length)

                for j, predicted_tensor in enumerate(predicted_tensors):
                    edit_distance = editdistance.eval(self.tokenizer.decode(predicted_tensor).replace(" ", ""),
                                                      batch["output_strings"][j][:-5].replace(" ", ""))
                    edit_distances.append(edit_distance)

                if i % print_every == 0:
                    print("evaluating batch " + str(i) + " out of " + str(len(test_pairs)))

            # print("average loss:", total_loss/len(test_pairs), "avg edit dist:", sum(edit_distances)/len(edit_distances))
            print("avg edit dist:", sum(edit_distances) / len(edit_distances))

        return sum(edit_distances) / len(edit_distances)

    def evaluate_iters_batch_keyword(self, test_pairs, pattern, debug_flag = False, print_every = 20, constraint = True):
        self.t5_model.eval()

        total_loss = 0
        count_dict = {}
        if pattern in ["pattern1", "pattern3", "pattern5", "pattern7", "pattern9", "pattern11"]:
            count_dict = {"0":0}
        elif pattern == "pattern2":
            count_dict = {"0":0, "1":0}
        elif pattern in ["pattern4", "pattern6", "pattern8", "pattern12"]:
            count_dict = {"0":0, "1":0, "2":0, "3":0}
        elif pattern == "pattern10":
            count_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4":0}

        with torch.no_grad():
            for i, batch in enumerate(test_pairs):

                # When using constraints, the flow is like this:
                # The t5 generates a bunch of generations, and the generations are checked by some constraints (depending
                #   on the pattern. For example, for pattern 10, the generation should have word overlap with the query
                #   and the rule buffer.
                # I think this is only used in the forward of SSL currently, and not used for evaluating the training
                #   process.
                if constraint:
                    if debug_flag:
                        print("+"*80)
                        print("sample:", i)

                    num_beams = 50
                    predicted_tensors = self.t5_model.generate(input_ids=batch["input"]["input_ids"].to(self.device),
                                                               attention_mask=batch["input"]["attention_mask"].to(
                                                                   self.device),
                                                               max_length=self.max_generation_length,
                                                               num_beams=num_beams, num_return_sequences=num_beams)

                    for j in range(len(batch["output_strings"])):
                        n_correct_beam = 0
                        for k in range(num_beams):
                            input_string = batch["input_strings"][j][:-5]
                            target_text = batch["output_strings"][j]
                            pred_text = self.tokenizer.decode(predicted_tensors[j*num_beams + k])

                            constraint_satisfied_flag = self._check_generation_with_constraint(pattern, input_string, pred_text, target_text, debug_flag)

                            if constraint_satisfied_flag:
                                count_dict = self._get_pattern_eval_metric(pattern, count_dict, pred_text,
                                                                           target_text, debug_flag)
                                n_correct_beam += 1
                                if debug_flag:
                                    print("*"*40)

                            if n_correct_beam > 0:
                                break

                # This just generates a bunch of generations. For each generation, it tries to judge whether it matches
                #    the target. If it matches the target, the counter adds one.
                else:
                    if debug_flag:
                        print("+"*80)
                        print("sample:", i)

                    num_beams = 1
                    predicted_tensors = self.t5_model.generate(input_ids=batch["input"]["input_ids"].to(self.device),
                                                               attention_mask=batch["input"]["attention_mask"].to(
                                                                   self.device),
                                                               max_length=self.max_generation_length,
                                                               num_beams=num_beams, num_return_sequences=num_beams)

                    for j, predicted_tensor in enumerate(predicted_tensors):
                        # TODO: change this if the manual debugging process need to be changed.
                        #target_text_cleaned = batch["output_strings"][j][:-5].replace(" ", "")
                        target_text = batch["output_strings"][0]
                        pred_text = self.tokenizer.decode(predicted_tensor)
                        count_dict = self._get_pattern_eval_metric(pattern, count_dict, pred_text, target_text, debug_flag)

                    if i % print_every == 0:
                        print("evaluating batch " + str(i) + " out of " + str(len(test_pairs)))

            # print("average loss:", total_loss/len(test_pairs), "avg edit dist:", sum(edit_distances)/len(edit_distances))
            print("evalated ",pattern, " subpattern counts:", count_dict)

        return count_dict


    def evaluate_iters_and_get_loss(self, test_pairs, print_every = 200):
        self.t5_model.eval()

        total_loss = 0
        edit_distances = []
        with torch.no_grad():
            for i in range(len(test_pairs)):
                input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)
                # target_tensor = self.tokenizer.encode(test_pairs[i]["output"], return_tensors="pt").to(self.device)
                #
                # loss = self.t5_model(input_ids=input_tensor, labels=target_tensor)[0]
                # total_loss+=loss.detach().cpu().numpy()

                predicted_tensor = self.t5_model.generate(input_tensor, max_length = self.max_generation_length)
                edit_distance = editdistance.eval(self.tokenizer.decode(predicted_tensor[0]).replace(" ",""), test_pairs[i]["output"][:-5].replace(" ",""))
                edit_distances.append(edit_distance)


                if i%print_every==0:
                    print("evaluating "+ str(i)+  " out of " +str(len(test_pairs)) )

            #print("average loss:", total_loss/len(test_pairs), "avg edit dist:", sum(edit_distances)/len(edit_distances))
            print("avg edit dist:", sum(edit_distances)/len(edit_distances))

        return sum(edit_distances)/len(edit_distances)

    def evaluate_iters_and_save_output(self, test_pairs, avg_loss, output_tsv_file):
        print("\t"+"*"*30)
        print("\tsaving output ...")

        with open(output_tsv_file, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['input', 'target', 'output', 'avg loss:'+str(avg_loss)])

            self.t5_model.eval()
            with torch.no_grad():
                for i in range(len(test_pairs)):
                    input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)
                    predicted_tensor = self.t5_model.generate(input_tensor, max_length = self.max_generation_length)

                    if i<2:
                        print("\tinput: %s" % (test_pairs[i]["input"]))
                        print("\ttarget: %s" % (test_pairs[i]["output"]))
                        print("\tactual output: %s" % (self.tokenizer.decode(predicted_tensor[0])))

                    if i<50:
                        tsv_writer.writerow([test_pairs[i]["input"], test_pairs[i]["output"], self.tokenizer.decode(predicted_tensor[0])])
                    else:
                        break


    def save_tuned_model(self, save_model_name):

        torch.save(self.t5_model, save_model_name)

    def evaluate_iters_and_get_exact_match(self, test_pairs, print_every = 200):
        self.t5_model.eval()

        total_loss = 0
        edit_distances = []

        correct_count = 0
        with torch.no_grad():
            for i in range(len(test_pairs)):
                input_strings = "context: " + test_pairs["context"] + " question: " + test_pairs["question"]

                input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)
                # target_tensor = self.tokenizer.encode(test_pairs[i]["output"], return_tensors="pt").to(self.device)
                #
                # loss = self.t5_model(input_ids=input_tensor, labels=target_tensor)[0]
                # total_loss+=loss.detach().cpu().numpy()

                predicted_tensor = self.t5_model.generate(input_tensor, max_length=self.max_generation_length)
                predicted_text = self.tokenizer.decode(predicted_tensor[0]).replace(" ", "")
                if predicted_text==test_pairs["answer"].replace(" ",""):
                    correct_count+=1

                if i % print_every == 0:
                    print("evaluating " + str(i) + " out of " + str(len(test_pairs)))

            # print("average loss:", total_loss/len(test_pairs), "avg edit dist:", sum(edit_distances)/len(edit_distances))
            print("eval accuracy:", correct_count/len(test_pairs))

        return correct_count/len(test_pairs)

    @classmethod
    def _get_pattern_eval_metric(cls, pattern, count_dict, pred_text, target_text, debug_flag):
        '''
        :param pattern: the pattern to evaluate
        :param count_dict: the dict of hit count for each sub pattern.
        :param pred_text: model's predicted text, without space or </s>.
        :param target_text: target text without space
        :return: subpattern hit count: a dict, the keys are all subpatterns in this pattern, the values
        are the hit count of each subpattern.
        '''
        pred_text = pred_text.replace(" ","")
        target_text = target_text[:-5].replace(" ","")

        if debug_flag:
            print("-"*20)
            print("target text:", target_text)
            print("pred text:", pred_text)
            print("dict before update:", count_dict)

        # For these patterns, we can consider there are only 1 subpattern for each pattern.
        if pattern == "pattern1" or \
            pattern == "pattern3" or \
            pattern == "pattern5" or \
            pattern == "pattern7" or \
            pattern == "pattern9" or \
            pattern == "pattern11":

            if pred_text == target_text:
                count_dict["0"] += 1

        # For pattern2, there are 2 possible output sub patterns.
        elif pattern == "pattern2":
            if "canprove" in target_text and pred_text == target_text:
                count_dict["0"]+=1
            elif "donotcontradict" in target_text and pred_text == target_text:
                count_dict["1"]+=1

        # For pattern 4 and pattern 8:
        # There are 4 possible subpatterns for each pattern:
        #   subpattern 1: can prove, only 1 target literal (1 buffer)
        #   subpattern 2: can prove, multiple target literals (multiple buffers)
        #   subpattern 3: do not contradict, only 1 target literal (1 buffer)
        #   subpattern 4: do not contradict, multiple target literals (multiple buffers)
        elif pattern == "pattern4" or pattern == "pattern8":
            if "canprove" in target_text:
                if "OR" not in target_text and pred_text == target_text:
                    count_dict["0"] += 1
                if "OR" in target_text and pred_text == target_text:
                    count_dict["1"] += 1
            if "notcontradict" in target_text:
                if "AND" not in target_text and pred_text == target_text:
                    count_dict["2"] += 1
                if "AND" in target_text and pred_text == target_text:
                    count_dict["3"] += 1

        # For pattern 6:
        # there are 4 patterns:
        # true, confirmed by
        # true, NAF
        # false, contradicted by
        # false, CWA
        elif pattern == "pattern6":
            if "true,thisisconfirmedbyfact" in target_text and pred_text == target_text:
                count_dict["0"] += 1
            if "false,CWA" in target_text and pred_text == target_text:
                count_dict["1"] += 1
            if "false,thisiscontradictedbyfact" in target_text and pred_text == target_text:
                count_dict["2"] += 1
            if "true,NAF" in target_text and pred_text == target_text:
                count_dict["3"] += 1

        # For pattern 10:
        # There are 5 possible sub patterns
        elif pattern == "pattern10":
            if "true" in target_text and pred_text == target_text:
                count_dict["0"] += 1
            if "false" in target_text and pred_text == target_text:
                count_dict["1"] += 1
            if "accordingtorule" in target_text:
                if "oraccordingto" in target_text and pred_text == target_text:
                    count_dict["2"] += 1
                elif ")and(accordingto" in target_text and pred_text == target_text:
                    count_dict["3"] += 1
                elif pred_text == target_text:
                    count_dict["4"] += 1

        elif pattern == "pattern12":
            if ")AND(" in target_text and pred_text == target_text:
                count_dict["3"] += 1
            elif "AND" in target_text and pred_text == target_text:
                count_dict["1"] += 1
            elif "OR" in target_text and pred_text == target_text:
                count_dict["2"] += 1
            elif pred_text == target_text:
                count_dict["0"] += 1

        if debug_flag:
            print("dict after update:", count_dict)

        return count_dict

    @classmethod
    def _check_generation_with_constraint(cls, pattern, input_text, pred_text, target_text, debug_flag):
        '''
        :param pattern: should only compare 5,6,9,10
        :param pred_text: the decoded generation
        :param target_text: target text
        :param debug_flag: true or false. If true, some debugging information is printed.
        :return: true or false
        '''
        # if debug_flag:
        #     print("target:", target_text, " pred:", pred_text)
        pred_text_cleaned = pred_text.replace(" ","")
        #target_text_cleaned = target_text[:-5].replace(" ","")

        if pattern in ["pattern1", "pattern2","pattern3","pattern4",
                       "pattern7","pattern8","pattern11","pattern12"]:
            return True
        elif pattern in ["pattern5", "pattern9"]:
            get_keyword_matches = re.findall(r"GET\((.*?\d)\)", pred_text_cleaned)
            run_keyword_matches = re.findall(r"RUN\((.*?\d)\)", pred_text_cleaned)

            if len(get_keyword_matches)!=0 and len(run_keyword_matches)!=0:
                get_argument = get_keyword_matches[0].replace("_", " ").lower()
                if len(run_keyword_matches[0].split(","))==2:
                    run_argument = run_keyword_matches[0].split(",")[1].replace("_"," ").lower()
                    if debug_flag:
                        print(pattern," get argument:", get_argument, " run argument:", run_argument)
                    if get_argument == run_argument and get_argument in input_text and run_argument in input_text:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

        elif pattern == "pattern6":
            query_extracted = re.findall(r"i want to judge whether .*\"(.*?)\"", input_text)[0]
            fact_buffer_extracted = re.findall(r"(fact \d+:.*\.)+", input_text)[0]
            fact_list = [fact[8:]for fact in fact_buffer_extracted[:-1].split(". ")]

            if debug_flag:
                print(pattern, " query:", query_extracted, " target:", target_text, " pred:", pred_text)
                print(fact_buffer_extracted)
                print(fact_list)

            if "true,NAF" in pred_text_cleaned:
                # the query should contain "not"
                if "not" in query_extracted:
                    query_positive = query_extracted.replace("not", "") if "does" not in query_extracted else query_extracted.replace("does not", "")
                    edit_distances = [editdistance.eval(query_positive.replace(" ",""), fact.replace(" ","")) for fact in fact_list]
                    if min(edit_distances)<3:  # There is a positive fact in the fact buffer to negative the query.
                        return False
                    else:
                        return True
                else:
                    return False
            elif "false,CWA" in pred_text_cleaned:
                # the query should not contain "not"
                if "not" not in query_extracted:
                    if query_extracted in fact_buffer_extracted:
                        return False
                    else:
                        return True
                else:
                    return False
            elif "true,thisisconfirmedby" in pred_text_cleaned:
                try:
                    # the fact should be the same as the query.
                    fact_num = re.findall(r"fact(\d+)", pred_text_cleaned)[0]
                    facts_extracted = re.findall(r"fact " + fact_num + ":(.*?)\.", input_text)

                    if debug_flag:
                        print(pattern, " query:",query_extracted, " fact:", facts_extracted)

                    if len(facts_extracted)>0 and query_extracted == facts_extracted[0]:
                        return True
                    else:
                        return False
                except:
                    return False

            elif "false,thisiscontradictedby" in pred_text_cleaned:
                try:
                    # the fact should have high overlap with the query
                    fact_num = re.findall(r"fact(\d+)", pred_text_cleaned)[0]
                    facts_extracted = re.findall(r"fact " + fact_num + ":(.*?)\.", input_text)

                    if debug_flag:
                        print(pattern, " query:",query_extracted, " fact:", facts_extracted)

                    if len(facts_extracted) > 0 and len(set(query_extracted.split(" ")).intersection(set(facts_extracted[0].split(" "))))>=3:
                        return True
                    else:
                        return False
                except:
                    return False

            else:
                return False

        elif pattern == "pattern10":
            query_extracted = re.findall(r"i want to judge whether .*\"(.*?)\"", input_text)[0]
            rule_buffer_extracted = re.findall(r"(rule \d+:.*\.)+", input_text)[0]
            rule_dict = {re.findall(r"(\d+)", rule)[0]: re.findall(r": (.+)", rule)[0]
                         for rule in rule_buffer_extracted[:-1].split(". ")}

            if debug_flag:
                print(" ")
                print(pattern, " query:", query_extracted)
                print("target:", target_text, " pred:", pred_text)
                print(rule_buffer_extracted)
                print(rule_dict)

            if pred_text_cleaned == "true":
                if "not" not in query_extracted:  # for positive query, the output should never be "true"
                    return False
                else:
                    # for negative query: if the pred answer is true, then it means there is no rule to prove the positive version of the negative query.
                    # This means: the rules either (1) the effect is a negative literal or (2) it does not contain the target entity.
                    # From this, we can conclude that:
                    return True

            elif pred_text_cleaned == "false":
                if "not" in query_extracted:
                    return False
                else:
                    return True  # There should be some heuristics, but maybe hard to compose.

            else: # The complex ones, generated proofs.
                # The patterns that can be generated:
                # according to rule X, Y and Y or according to rule X, Y and Y.
                # according to rule X, Y and Y )and( according to rule X, Y and Y.
                if "according to" not in pred_text:
                    return False

                if " or " in pred_text:
                    if " not " in query_extracted:
                        return False

                if " )and( " in pred_text:
                    if " not " not in query_extracted:
                        return False

                try:
                    literals = pred_text[:-1].split(" or ") if " or " in pred_text else pred_text.split(" )and( ")
                    literals_parsed = [(re.findall(r"according to rule (\d+)" , literal)[0], re.findall(r"i need to prove (.+)" , literal)[0].split(" "))
                                       for literal in literals]
                except:
                    return False

                query_tokens = query_extracted.split(" ")

                if debug_flag:
                    print("literal parsed:", [(literal_tuple[0], " ".join(literal_tuple[1])) for literal_tuple in literals_parsed])
                for literal_tuple in literals_parsed:
                    # print("\trule num matched:", literal_tuple[0], " not in dict false?", literal_tuple[0] not in rule_dict)
                    # print("\trule last two tokens:", rule_dict[literal_tuple[0]].split(" ")[-2:],
                    #       " not false?", "not" in rule_dict[literal_tuple[0]].split(" ")[-2:])
                    # print("\trule last token:", rule_dict[literal_tuple[0]].split(" ")[-1],
                    #       " query last token:", query_tokens[-1], " last token differ false?", rule_dict[literal_tuple[0]].split(" ")[-1] != query_tokens[-1])
                    #
                    # matched_rule_tokens = rule_dict[literal_tuple[0]].split(" ")
                    # print("\trule:",rule_dict[literal_tuple[0]], " literal:", " ".join(literal_tuple[1]),
                    #       " intersection empty false:", len(set(matched_rule_tokens).intersection(set(literal_tuple[1])))==0)
                    #
                    # print("\tquery literal intersection:", set(query_tokens).intersection(set(literal_tuple[1])),
                    #         " non empty false?", len(set(query_tokens).intersection(set(literal_tuple[1])))==0)
                    # input("----waiting----")

                    # check 0: the rule number predicted should be in the rule buffer.
                    if literal_tuple[0] not in rule_dict:
                        return False

                    # check 1: if the matched rules have "not" in the last two tokens: then it should not happen.
                    if "not" in rule_dict[literal_tuple[0]].split(" ")[-2:]:
                        return False

                    # check 2: the query's last token should match the rules' last tokens
                    if rule_dict[literal_tuple[0]].split(" ")[-1] != query_tokens[-1]:
                        return False

                    # check 3: the literal and the matched rule should have overlap
                    matched_rule_tokens = rule_dict[literal_tuple[0]].split(" ")
                    if len(set(matched_rule_tokens).intersection(set(literal_tuple[1])))==0:
                        return False

                    # check 4: the query and the literal should have overlap
                    if len(set(query_tokens).intersection(set(literal_tuple[1])))==0:
                        return False

                # if all checks pass, return True.
                return True

        else:
            return True

    @classmethod
    def check_generation_with_constraint(cls, pattern, input_text, pred_text, target_text, debug_flag):
        return cls._check_generation_with_constraint(pattern, input_text, pred_text, target_text, debug_flag)
