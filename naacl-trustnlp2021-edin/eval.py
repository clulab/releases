"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/' +'vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, opt['data_dir'] + '/mappings_{}.txt'.format(args.dataset), opt['data_dir'] + '/interval_{}.txt'.format(args.dataset), evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
references = []
candidates = []
all_probs = []
batch_iter = tqdm(batch)

import json
from collections import defaultdict
with open('dataset/tacred/mappings_train.txt') as f:
    mappings = f.readlines()

with open('dataset/tacred/rules.json') as f:
    rules = json.load(f)
rule_dict = defaultdict(int)
for m in mappings:
    if 't_' in m or 's_' in m:
        for l, r in eval(m):
            r = ''.join(helper.word_tokenize(rules[r]))
            rule_dict[r] += 1

whole = set(rule_dict.keys())

x = 0
exact_match = 0
other = 0

d_wrong = list()
r_wrong = list()
s_wrong = list()

d_correct = list()
r_correct = list()
s_correct = list()

d_wrong_no = list()
r_wrong_no = list()
s_wrong_no = list()

d_correct_no = list()
r_correct_no = list()
s_correct_no = list()

gold_no_rule = list()
pred_no_rule = list()

gold_w_rule = list()
pred_w_rule = list()

x_no = 0
x_w  = 0
for c, b in enumerate(batch_iter):
    preds, words, decoded, loss = trainer.predict(b)
    predictions += preds

    batch_size = len(preds)
    for i in range(batch_size):
        output = decoded[i]
        candidate = []
        for r in output[1:]:
            if int(r) == 3:
                break
            else:
                candidate.append(vocab.id2rule[int(r)])
        if len(batch.refs[x][0])!=0:
            gold_w_rule.append(batch.gold()[x])
            pred_w_rule.append(id2label[preds[i]])
            x_no += 1
            # if id2label[preds[i]]!=batch.gold()[x]:
            #     s_wrong.append(' '.join([vocab.id2word[w] for w in words[i] if w!=0]))
            #     d_wrong.append((''.join(candidate), id2label[preds[i]]))
            #     r_wrong.append((''.join(batch.refs[x][0]), batch.gold()[x]))
            # else:
            #     s_correct.append(' '.join([vocab.id2word[w] for w in words[i] if w!=0]))
            #     d_correct.append((''.join(candidate), id2label[preds[i]]))
            #     r_correct.append((''.join(batch.refs[x][0]), batch.gold()[x]))
        else:#if id2label[preds[i]] != 'no_relation':
            gold_no_rule.append(batch.gold()[x])
            pred_no_rule.append(id2label[preds[i]])
            x_w += 1
            # if id2label[preds[i]]!=batch.gold()[x]:
            #     s_wrong_no.append(' '.join([vocab.id2word[w] for w in words[i] if w!=0]))
            #     d_wrong_no.append((''.join(candidate), id2label[preds[i]]))
            #     r_wrong_no.append((''.join(batch.refs[x][0]), batch.gold()[x]))
            # else:
            #     s_correct_no.append(' '.join([vocab.id2word[w] for w in words[i] if w!=0]))
            #     d_correct_no.append((''.join(candidate), id2label[preds[i]]))
            #     r_correct_no.append((''.join(batch.refs[x][0]), batch.gold()[x]))
        x += 1
# print ('wrong pred', len(d_wrong))
# l = random.sample(range(len(d_wrong)), min(50, len(d_wrong)))
# for i in l:
#     print (s_wrong[i])
#     print (d_wrong[i])
#     print (r_wrong[i])
#     print ()
# print ()
# print ()
# print ()
# print ('correct pred', len(d_correct))
# l = random.sample(range(len(d_correct)), min(50, len(d_correct)))
# for i in l:
#     print (s_correct[i])
#     print (d_correct[i])
#     print (r_correct[i])
#     print ()
# print ()
# print ()
# print ()
# print ('wrong pred no ref',len(d_wrong_no))
# l = random.sample(range(len(d_wrong_no)), min(50, len(d_wrong_no)))
# for i in l:
#     print (s_wrong_no[i])
#     print (d_wrong_no[i])
#     print (r_wrong_no[i])
#     print ()
# print ()
# print ()
# print ()
# print ('correct pred no ref', len(d_correct_no))
# l = random.sample(range(len(d_correct_no)), min(50, len(d_correct_no)))
# for i in l:
#     print (s_correct_no[i])
#     print (d_correct_no[i])
#     print (r_correct_no[i])
#     print ()

predictions = [id2label[p] for p in predictions]
# for pred in predictions:
#     print (pred)
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
p2, r2, f12 = scorer.score(gold_no_rule, pred_no_rule, verbose=True)
p3, r3, f13 = scorer.score(gold_w_rule, pred_w_rule, verbose=True)
bleu = 0#corpus_bleu(references, candidates)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}\t{:.4f}".format(args.dataset,p,r,f1,bleu))
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}\t{:.4f}".format("no rule",p2,r2,f12,x_no))
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}\t{:.4f}".format("rule",p3,r3,f13,x_w))

print("Evaluation ended.")

