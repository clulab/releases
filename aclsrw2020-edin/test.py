import argparse
import random
import pickle
import json
import os
import glob

from model_dy import *
from bio_utils import *

from nltk.translate.bleu_score import corpus_bleu

from collections import defaultdict

if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('datadir')
    parser.add_argument('dev_datadir')
    parser.add_argument('label')
    args = parser.parse_args()
    input_lang = Lang("input")
    pl1 = Lang("position")
    char = Lang("char")
    rule_lang = Lang("rule")
    raw_train = list()
    input_lang, pl1, char, rule_lang, raw_train = pickle.load(open("indexes_ge_2K2.pickle", "rb"))
    # input_lang, pl1, char, rule_lang, raw_train = prepare_data(args.datadir,args.label,input_lang, pl1, char, rule_lang, raw_train)
    # input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed_loc",args.label,input_lang, pl1, char, rule_lang, raw_train, "valids_loc.json")
    input2_lang, pl2, char2, rule_lang2, raw_test = prepare_data(args.dev_datadir,args.label, valids="valids_new.json")
    # input_lang, pl1, char, rule_lang, raw_train = parse_json_data(input_lang, pl1, char, raw_train)
    model = LSTMLM.load(args.model)
    i = j = 0
    # for fname in glob.glob(os.path.join(args.dev_datadir, '*.a1')):
    #     root = os.path.splitext(fname)[0]
    #     output = "preds/"+root.split("/")[-1]+ '.a2'
    #     test = []
    #     raw_test, tcount = prepare_data(fname, False, "valids.json")
    #     for datapoint in raw_test:
    #         if datapoint[4]:
    #             i += 1
    #             test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
    #                 datapoint[1],
    #                 datapoint[2],
    #                 datapoint[3], 
    #                 input_lang.label2id[datapoint[4]], 
    #                 [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-2]]+[0],
    #                 [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
    #                 datapoint[-3]))
    #         else:
    #             j += 1
    #             test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
    #                 datapoint[1],
    #                 datapoint[2],
    #                 datapoint[3], 0, 
    #                 [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-2]]+[0],
    #                 [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
    #                 datapoint[-3]))
    #     predict = 0.0
    #     label_correct = 0.0
    #     trigger_correct = 0.0
    #     both_correct = 0.0
    #     references = []
    #     candidates = []
    #     phosphos = dict()
    #     events = defaultdict(list)
    test = list()
    i = j = 0
    for datapoint in raw_test:
        if datapoint[3][0] != -1:
            i += len(datapoint[3])
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],#entity
                datapoint[2],#entity position
                datapoint[3],#trigger position
                [input_lang.label2id[l] for l in datapoint[4]],#trigger label
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],#positions
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
                [[rule_lang.word2index[p] for p in rule + ["EOS"]] for rule in datapoint[6]],
                ))
        else:
            j += 1
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],
                datapoint[2],
                datapoint[3], [0], 
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
                [rule_lang.word2index["EOS"]],
                ))
    print(i,j)

    predict = 0.0
    label_correct = 0.0
    trigger_correct = 0.0
    both_correct = 0.0
    tp = 0
    s = 0
    r = 582
    references = []
    candidates = []

    references1 = []
    candidates1 = []

    references2 = []
    candidates2 = []

    references3 = []
    candidates3 = []
    for datapoint in test:
        triggers = datapoint[3]  
        rules = datapoint[7]
        # tags = datapoint[8]

        sentence = datapoint[0]
        entity = datapoint[2]
        pos = datapoint[5]
        chars = datapoint[6]

        pred_triggers, score, contexts, hidden, pred_rules = model.get_pred(sentence, pos, chars, entity)
        if len(pred_triggers) != 0:
            s += len(pred_triggers)
        if len(pred_triggers) != 0 or triggers[0] != -1:
            # if len(pred_triggers) != 0:
            #     s += len(pred_triggers)
            # if triggers[0] != -1:
            #     r += len(triggers)
            for k, t in enumerate(pred_triggers):
                with open("rules_phos.txt", "a") as f:
                    f.write(' '.join([rule_lang.index2word[id] for id in pred_rules[k]]))
                    candidates1.append(pred_rules[k])
                    candidates2.append(pred_rules[k])
                    if t in triggers:
                        tp += 1
                        j = triggers.index(t)
                        if rules[j][0] != 0:
                            f.write('-$|$-'+' '.join([rule_lang.index2word[id] for id in rules[j][:-1]]))
                            references.append([rules[j][:-1]])
                            references1.append([rules[j][:-1]])
                            references2.append([rules[j][:-1]])
                            candidates.append(pred_rules[k])
                            references3.append([rules[j][:-1]])
                            candidates3.append(pred_rules[k])
                        else:
                            references1.append([[]])
                            references2.append([[]])
                    else:
                        references1.append([[]])
                        references2.append([[]])
                    f.write('\n')
            for t in triggers:
                if t != -1 and t not in pred_triggers:
                    j = triggers.index(t)
                    references1.append([rules[j][:-1]])
                    candidates1.append([])
                    references3.append([rules[j][:-1]])
                    candidates3.append([])
    explained = 0
    for i, sent in enumerate(references):
        # print (sent[0])
        # print (candidates[i])
        if sent[0] == candidates[i]:
            explained += 1
    print (explained)
    print (len(references))
    print (len(references3))
    print (len(candidates1))
    print ("%.4f"%(explained/len(candidates)))
    precision = tp/s if s!= 0 else 0
    recall = tp/r
    f1 = 2*(precision*recall)/(recall+precision) if recall+precision != 0 else 0
    bleu = corpus_bleu(references, candidates)
    bleu1 = corpus_bleu(references1, candidates1)
    bleu2 = corpus_bleu(references2, candidates2)
    bleu3 = corpus_bleu(references3, candidates3)
    print (tp, r, s)
    print ("Recall: %.4f Precision: %.4f F1: %.4f BLEU: %.4f BLEU1: %.4f BLEU2: %.4f BLEU3: %.4f"%(recall, precision, f1, bleu, bleu1, bleu2, bleu3))
        
