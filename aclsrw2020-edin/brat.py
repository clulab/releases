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
    parser.add_argument("--dev", default=False, action="store_true" , help="Flag to do something")
    args = parser.parse_args()

    input_lang = Lang("input")
    pl1 = Lang("position")
    char = Lang("char")
    rule_lang = Lang("rule")
    raw_train = list()

    if os.path.exists(args.datadir):
        input_lang, pl1, char, rule_lang, raw_train   = pickle.load(open('%s/train'%args.datadir, "rb"))
        if args.dev:
            input2_lang, pl2, char2, rule_lang2, raw_test = pickle.load(open('%s/test1'%args.datadir, "rb"))
        else:
            input2_lang, pl2, char2, rule_lang2, raw_test = pickle.load(open('%s/test2'%args.datadir, "rb"))
    else:
        print ('Data file not found!')
        exit()

    model = LSTMLM.load(args.model)
    test = list()
    for datapoint in raw_test:
        test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
            datapoint[1],
            datapoint[2],
            [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[3]]+[0],
            [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
            datapoint[4],
            datapoint[5]))

    print (raw_test[0])
    prev_root = raw_test[0][-1]
    tcount = int(raw_test[0][-2][1:])
    phosphos = dict()
    events = defaultdict(set)
    total = 0
    for i, datapoint in enumerate(test):
        sentence = datapoint[0]
        eid = datapoint[1]
        entity = datapoint[2]
        pos = datapoint[3]
        chars = datapoint[4]
        starts = datapoint[5]
        ends = datapoint[6]
        text = raw_test[i][0]
        root = raw_test[i][-1]
        pred_triggers, score, contexts, hidden, pred_rules = model.get_pred(sentence, pos, chars, entity)
        total += len(pred_triggers)
        if root == prev_root:
            if len(pred_triggers) != 0:
                for pred_trigger in pred_triggers:
                    if str(starts[pred_trigger[0]])+" "+str(ends[pred_trigger[0]])+"\t"+text[pred_trigger[0]] not in phosphos:
                        tcount += 1
                        phosphos[str(starts[pred_trigger[0]])+" "+str(ends[pred_trigger[0]])+"\t"+text[pred_trigger[0]]] = "T"+str(tcount)+"\t"+input_lang.labels[pred_trigger[1]]
                    tid = phosphos[str(starts[pred_trigger[0]])+" "+str(ends[pred_trigger[0]])+"\t"+text[pred_trigger[0]]].split("\t")[0]
                    events[input_lang.labels[pred_trigger[1]]+":"+tid].add(eid)
        else:
            with open(prev_root+".a2p", "a") as f:
                for k in phosphos:
                    f.write(phosphos[k]+" "+k+"\n")
                ecount = 1
                for tid in events:
                    for e in events[tid]:
                        f.write("E"+str(ecount)+"\t"+tid+" Theme:"+e+"\n")
                        ecount += 1
            prev_root = root
            phosphos = dict()
            tcount = int(raw_test[i][-2][1:])
            events = defaultdict(set)
    with open(prev_root+".a2p", "a") as f:
        for k in phosphos:
            f.write(phosphos[k]+"\t"+" "+k+"\n")
        ecount = 1
        for tid in events:
            for e in events[tid]:
                f.write("E"+str(ecount)+"\t"+tid+" Theme:"+e+"\n")
                ecount += 1
    print (total)