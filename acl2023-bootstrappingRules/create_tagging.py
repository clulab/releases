import json
from collections import Counter, defaultdict
from termcolor import colored
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model_output', type=str)
parser.add_argument('--rule_output', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--rule', action='store_true')
parser.add_argument('--kept', type=str)

args = parser.parse_args()

triggers = json.load(open('triggers.json'))
j = json.load(open('src/main/resources/data/train.json'))

def filter(trigger, ss, se, os, oe, tokens, trigger_words):
    res = [i for i in trigger if i not in range(ss, se+1) and i not in range(os, oe+1) and tokens[i] in trigger_words]
    return res

def from_model(model_output, output):
    with open(output, 'w') as f:
        for i, item in enumerate(model_output):
            if item["from_prev"]:
                f.write (item['prev_label']+"\t"+str(item["gold_tags"]))
                f.write('\n')
            elif item["predicted_label"]!="no_relation":
                f.write (item['predicted_label']+"\t"+str(item["predicted_tags"]))
                f.write('\n')
            else:
                f.write ("no_relation\t[]")
                f.write('\n')


def from_rule(model_output, rule_output, output, kept):
    kept = json.load(open(kept))
    with open(output, 'w') as f:
        for i, item in enumerate(model_output):
            
            ss = j[i]['subj_start']
            se = j[i]['subj_end']
            os = j[i]['obj_start']
            oe = j[i]['obj_end']
            p = Counter()
            t = dict()
            for a in rule_output[i].strip().split('|'):
                rl = a.split('\t')[0]
                rl = rl if rl in kept and a.split('\t')[-1] in kept[rl] else 'no_relation'
                masked = eval(a.split('\t')[1])
                if rl != 'no_relation':
                    trigger_words = triggers[re.sub(r'_\d+_', '_0_', a.split('\t')[2])].split('"')
                    temp = [k for k in range(masked[0], masked[1])]
                    temp = filter(temp, ss, se, os, oe, j[i]['token'], trigger_words)
                    if rl not in t:
                        t[rl] = temp
                    elif len(t[rl]) > len(temp):
                        t[rl] = temp
                p.update({rl:1})

            if p.most_common(1)[0][0] == 'no_relation' and len(p)!=1:
                predicted_label = p.most_common(2)[1][0]
            else:
                predicted_label = p.most_common(1)[0][0]

            predicted_tags = t.get(predicted_label, [])

            if item["from_prev"]:
                f.write (item['prev_label']+"\t"+str(item["gold_tags"]))
                f.write('\n')
            elif predicted_label!="no_relation":
                f.write (predicted_label+"\t"+str(predicted_tags))
                f.write('\n')
            else:
                f.write ("no_relation\t[]")
                f.write('\n')


model_output = json.load(open(args.model_output))
rule_output = open(args.rule_output).readlines()
if args.rule:
    from_rule(model_output, rule_output, args.output, args.kept)
else:
    from_model(model_output, args.output)
