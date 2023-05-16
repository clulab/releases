import json
from utils import scorer
from collections import defaultdict, Counter
import random
import sys

def load_data(gold, pred):
    j = json.load(open(gold))
    golds = list()
    predictions_per_rule = dict()

    for i, item in enumerate(open(pred)):
        item = item.strip()
        label = j[i]['relation']
        golds.append(label)
        for d in item.split('|'):
            d = d.split('\t')
            rule = d[-1]
            if rule+"|"+d[0] not in predictions_per_rule:
                predictions_per_rule[rule+"|"+d[0]] = ['no_relation' for x in range(len(j))]
            predictions_per_rule[rule+"|"+d[0]][i] = d[0]

    return golds, predictions_per_rule

def eval_w_threshold(threshold, predictions_per_rule, golds):
    rules = predictions_per_rule.keys()

    kept = defaultdict(list)
    for rule in rules:
        rule2, label = rule.split('|')
        prec, recall, f1 = scorer.score_per_label(golds, predictions_per_rule[rule], label)
        if prec>=threshold:
          kept[label].append(rule2)
    return kept

def get_scores(gold, curr, rfilter=None, prev=None):

    predictions = list()
    if prev != None:
        prev = open(prev).readlines()

    for i, item in enumerate(open(curr)):
        item = item.strip()

        p = Counter()
        if prev != None:
            item2 = prev[i].strip().split('\t')[0]
            p.update({a.split('\t')[0]:1.1 for a in item2.split('|')})
            
        if "no_relation" not in item:
            for a in item.split('|'):
                if rfilter == None:
                    p.update({a.split('\t')[0]:1})
                elif a.split('\t')[-1] in rfilter[a.split('\t')[0]]:
                    p.update({a.split('\t')[0]:1})
                else:
                     p.update({"no_relation":1})
        else:
             p.update({"no_relation":1})
        if p.most_common(1)[0][0] == 'no_relation' and len(p)!=1:
            p = p.most_common(2)[1][0]
        else:
            p = p.most_common(1)[0][0]
        predictions.append(p)

    prec_micro, recall_micro, f1_micro, f1s = scorer.score(gold, predictions, verbose=False)

    return prec_micro, recall_micro, f1_micro

golds, predictions_per_rule = load_data('src/main/resources/data/dev_1pc.json', sys.argv[1])

final = None
best = 0
bt = -1
for i in range(5, 10):
    th = float(i/10)
    kept = eval_w_threshold(th, predictions_per_rule, golds)
    _,_, f1 = get_scores(golds, sys.argv[1], kept)
    if f1>best:
        final = kept
        best = f1
        bt = th
print (bt, best)
with open(sys.argv[2], 'w') as f:
    f.write(json.dumps(final))


