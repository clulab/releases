"""
Run evaluation with saved models.
"""
import time
import random
import argparse
from tqdm import tqdm
import torch

from dataloader import DataLoader
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from pytorch_pretrained_bert.tokenization import BertTokenizer

import json

from termcolor import colored

import statistics

def check(tags, ids):
    for i in ids:
        if i<len(tags) and tags[i] == 1:
            return True
    return False

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--device', type=int, default=0, help='Word embedding dimension.')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')

# load opt
it = args.model_dir.split("/")[-1]
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, tokenizer, True)

origin = json.load(open(data_file))
tagging = []
with open(opt['data_dir'] + '/tagging_{}_0.txt'.format(args.dataset)) as f:
    # tagging = f.readlines()
    for i, line in enumerate(f):
        tagging.append(line)

# helper.print_config(opt)
if "tacred" in opt["data_dir"]:
    label2id = constant.LABEL_TO_ID_tacred
else:
    label2id = constant.LABEL_TO_ID_conll
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []

x = 0
exact_match = 0
other = 0
tags = []
durations = list()
for c, b in tqdm(enumerate(batch)):
    start_time = time.time()
    preds,t,_,_ = trainer.predict(b, id2label, tokenizer)
    duration = time.time() - start_time
    durations.append(duration)
    predictions += preds
    tags += t
    batch_size = len(preds)
output = list()
tagging_scores = []
output = list()

for i, p in enumerate(predictions):
    rp, tagged = tagging[i].split('\t')
    tagged = eval(tagged)
    predictions[i] = id2label[p] if id2label[p] == 'no_relation' or batch.entity_validations[i] in constant.TACRED_VALID_CONDITIONS[id2label[p]] else "no_relation"
    output.append({'gold_label':batch.gold()[i], 'predicted_label':predictions[i], 'predicted_tags':[], 'gold_tags':[], 'from_prev':True if rp!='no_relation' else False})

    # if p!=0:
    output[-1]["predicted_tags"] = [j for j, t in enumerate(batch.words[i]) if check(tags[i], t[1])]
    if len(tagged)>0:
        output[-1]['gold_tags'] = tagged
        correct = 0
        pred = 0
        for j, t in enumerate(batch.words[i]):
            if check(tags[i], t[1]):
                pred += 1
                if j in tagged:
                    correct += 1
        if pred > 0:
            r = correct / pred
        else:
            r = 0
        if len(tagged) > 0:
            p = correct / len(tagged)
        else:
            p = 0
        try:
            f1 = 2.0 * p * r / (p + r)
        except ZeroDivisionError:
            f1 = 0
        tagging_scores.append((r, p, f1))

print ("Average: {:.3f} sec/batch".format(statistics.mean(durations)))

with open("output_%s.json"%it, 'w') as f:
    f.write(json.dumps(output))
p, r, f1, ba = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p*100,r*100,f1*100))
tr, tp, tf = zip(*tagging_scores)
print("{} set tagging  result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,statistics.mean(tr)*100,statistics.mean(tp)*100,statistics.mean(tf)*100))
print("Evaluation ended.")
