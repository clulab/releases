#!/usr/bin/env python

import os
import sys
import argparse
import random
import pickle
from copy import deepcopy
import torch
import numpy as np
from utils import chunker, parse_uas, get_slanted_triangular_lr
from conll import read_conll, eval_conll, parse_conll
from vocabulary import *
from model import Pat
import time
from bert_features import Bert

# TODO things to modify:
#  1) batch size to a power of n -- DONE - not tested yet.
#  2) weight decay -- DONE - seems to improve a bit. Obtained 92 whne using weight decay 0.00005. Values this low seem to improve. Higher don't help
#  3) add a predict postag head during training (more gradient flowing) -- done, but this makes necessary to remove the partofspeech tags from the embeddings before the BiLSTM layer, which impacts negatively. Overcoming this might mean a more complicated model.
#  4) dropout -- DONE
#  5) upos or xpos switch -- DONE
#  6) modify when predicting relation -- either append the predicted head (not the highest, the one that was the result of the cycle removal), or the a weighted sum using the probabilities.

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('train')
parser.add_argument('dev')
parser.add_argument('--output', default='output')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--glove-emb', type=str, default=None)
parser.add_argument('--elmo-opts', type=str, default=None)
parser.add_argument('--elmo-weights', type=str, default=None)
parser.add_argument('--random-seed', type=int, default=1, help='random seed')
parser.add_argument('--disable-early-stopping', action='store_true', help='disable early stopping')
parser.add_argument('--early-stopping-on', type=str, default="uas", help="'las' or 'uas'. When it doesn't increase for a number of epochs equal to --max-epochs-without-improvement, stops")
parser.add_argument('--max-epochs-without-improvement', type=int, default=3, help='max number of epochs with no improvement (only used with early stopping)')
parser.add_argument('--word-emb-size', type=int, default=100, help='word embedding size')
parser.add_argument('--tag-emb-size', type=int, default=40, help='part-of-speech embedding size')
parser.add_argument('--bilstm-num-layers', type=int, default=2, help='number of bilstm layers')
parser.add_argument('--bilstm-hidden-size', type=int, default=400, help='bilstm hidden size')
parser.add_argument('--bilstm-dropout', type=float, default=0.1, help='dropount on bilstm')
parser.add_argument('--batch-size', type=int, default=64, help='mini batch size')

parser.add_argument('--mlp-hidden-size', type=int, default=500, help='hidden size of the MLP')
parser.add_argument('--mlp-output-size', type=int, default=100, help='output size of the MLP')

parser.add_argument('--pos-count-threshold', type=int, default=0, help='pos label count threshold')
parser.add_argument('--pos-hidden-size', type=int, default=100, help='pos hidden layer size')

parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')

parser.add_argument('--char-emb', action='store_true', help='use character embeddings')
parser.add_argument('--char-emb-hidden-size', type=int, default=25, help='output size of the embeddings model')
parser.add_argument('--char-emb-size', type=int, default=50, help='size of embedddings used for each char within the model')

parser.add_argument('--position-emb', action='store_true', help='use position embeddings')
parser.add_argument('--position-emb-max-pos', type=int, default=150, help='max position that can be encoded to embeddings')
parser.add_argument('--position-emb-size', type=int, default=20, help='position embeddings size')

parser.add_argument('--bert', action='store_true', help='use bert features')
parser.add_argument('--bert-batch-size', type=int, default=1, help='bert batch size')
parser.add_argument('--bert-layers', type=str, default='-1,-2,-3,-4', help='layers extracted from bert')
parser.add_argument('--bert-store-features', action='store_true', help='store bert features')
parser.add_argument('--bert-load-features', action='store_true', help='load bert features from data.piclke file')
parser.add_argument('--bert-hidden-size', type=int, default=768, help='used in order to tell Pat model the hidden size of bert')
parser.add_argument('--bert-max-seq-length', type=int, default=512, help='set the max length for bert')
parser.add_argument('--bert-multilingual-cased', action='store_true', help='use bert-base-multilingual-cased; the default is (en only) bert-base-uncased')


parser.add_argument('--polyglot', type=str, required=False, default=None, help="path to polyglot to be used as embedding")

parser.add_argument('--loss-weight-factor', type=float, required=False, default=1.0, help="Weight factor to consider when adding the losses")

#CNN Char Embeddings
parser.add_argument('--cnn-ce', action='store_true', help='use cnn char embeddings')
parser.add_argument('--cnn-embeddings-size', type=int, default=50, help='size of embeddings used for each char')
parser.add_argument('--cnn-ce-kernel-size', type=int, default=3, help='size of the filter to be used in (must be odd)')
parser.add_argument('--cnn-ce-out-channels', type=int, default=50, help='output size')

#Label classifier using head
parser.add_argument('--use-head', action='store_true', help="use the gold head (train mode) and predicted head (evaluation mode) for predicting the label")

#Adam Optimizer
parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 in Adam")
parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 in Adam")
parser.add_argument('--weight-decay', type=float, default=0.0, help="Weight decay")
parser.add_argument('--slanted-triangle-lr', action='store_true', help="Use or not the slanted triangle learning rate technique")

#Dropout
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout value")

#Part of speech tag
parser.add_argument('--part-of-speech', type=str, default="upos", help='Which part of speech tag to use. One of {"upos", or ""xpos"')

parser.add_argument('--which-cuda', type=int, default=0, help='which cuda to use')

args = parser.parse_args()
print(args)

# ensure both elmo opts and weights were given or none of them
if ((args.elmo_opts and args.elmo_weights) or (not args.elmo_opts and not args.elmo_weights)) == False:
    print('error: both --elmo-weights and --elmo-opts must be set')
    exit()

# ensure output dir exists
if not os.path.exists(args.output):
    os.makedirs(args.output)
model_name = os.path.join(args.output, 'pat')

print(f'\nsetting random seed to {args.random_seed}')
# Set randoms
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.bert_load_features:
    print('\n Loading data from data.pickle and ignoring train and dev parameters')
    with open(os.path.join(args.output, 'bert_features.pickle'), 'rb') as f:
        data = pickle.load(f)
    train = data['train']
    train_copy = deepcopy(train)
    dev = data['dev']

else:
    print(f'\nloading training data from {args.train}')
    train = read_conll(args.train, lower_case=not args.bert_multilingual_cased)
    # remove <root> token
    #train = [s[1:] for s in train]
    # ignore sentences that had only <root>
    #train = [s for s in train if len(s) > 0]

    #train = train[:50]
    # keep a copy of the train dataset   that will be used for evaluation
    train_copy = deepcopy(train)

    print(f'\nloading development data from {args.dev}')
    dev = read_conll(args.dev, lower_case=not args.bert_multilingual_cased)
    #dev = [s[1:] for s in dev]
    #dev = [s for s in dev if len(s) > 0]

# if bert is used, generate bert_features
if args.bert and args.bert_load_features != True:
    print('\nloading BERT...')
    # 168 here because it's the longest sentence in our training dataset
    # FIXME: add max_seq_length as parameter
    bert = Bert(args.bert_layers, args.bert_max_seq_length, args.bert_batch_size, args.bert_multilingual_cased, args.which_cuda)
    # set bert hidden size
    args.bert_hidden_size = bert.model.config.hidden_size

    print('extracting features for dev...')
    bert.extract_bert_features(dev)
    print('extracting features for training...')
    bert.extract_bert_features(train)
    if args.bert_store_features:
        data = {'train':train, 'dev':dev}
        print('saving bert features to disk')
        with open(os.path.join(args.output, 'bert_features.pickle'), 'wb') as f:
            pickle.dump(data, f)
    print('done with BERT.')

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('\nmaking vocabularies')
word_vocab = make_word_vocabulary(train)
print(f'{len(word_vocab):,} distinct words')
tag_vocab = make_tag_vocabulary(train, args.part_of_speech)
print(f'{len(tag_vocab):,} distinct POS tags')
char_vocab = make_char_vocabulary(train)
print(f'{len(char_vocab):,} distinct characters')
pos_vocab = make_pos_vocabulary(train, prune=args.pos_count_threshold)
print(f'{len(pos_vocab):,} distinct positions with count > {args.pos_count_threshold}')
deprel = make_deprel_vocabulary(train)
print(f'{len(deprel):,} distinct dependencies')

# early stopping
best_dev_uas = -np.inf
best_dev_las = -np.inf
best_msg = None
best_epoch = None
num_epochs_since_best = 0
what = args.early_stopping_on

print('\ntraining')
device = torch.device(f'cuda:{args.which_cuda}' if torch.cuda.is_available() else 'cpu')
pat = Pat(args, word_vocab, tag_vocab, pos_vocab, deprel, char_vocab).to(device)
pat.mode = 'training'


optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, pat.parameters()),
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay
)
for epoch in range(args.epochs):
    pat = pat.train()
    pat.mode = 'training'
    # Adaptive learning rate - slanted triangle learning rate technique
    if args.slanted_triangle_lr:
        for param_group in optimizer.param_groups:
            new_lr = get_slanted_triangular_lr(epoch, args.epochs)
            param_group['lr'] = new_lr
    start = time.time()
    print('starting epoch', epoch)
    random.shuffle(train)
    for batch in chunker(train, args.batch_size):
        # clear gradients
        optimizer.zero_grad()
        # calculate loss
        loss = pat.train_conll(batch)
        # back propagate
        loss.backward()
        # optimize parameters
        optimizer.step()
        # print loss only if not using gpu due to slow downs caused by transfering data from gpu to cp
        if device.type == 'cpu':
            print('loss =', loss.item())
    print('-' * 50)
    print('epoch', epoch)

    print('  elapsed time:', int((time.time()-start)/60), 'minutes and', int((time.time()-start)%60), 'seconds')
    # Evaluate
    pat = pat.eval()
    pat.mode = 'evaluation'
    pat.no_cycles = False
    pat.print_nr_of_cycles = False
    with torch.no_grad():
        parse_conll(pat, dev, args.batch_size, clear=True)
    dev_uas, dev_las = eval_conll(dev, args.dev, verbose=False)
    print('  dev uas:', dev_uas)
    print('  dev las:', dev_las)
    if not args.disable_early_stopping:
        new_dev_uas = parse_uas(dev_uas)
        new_dev_las = parse_uas(dev_las)
        improved = False
        if what == 'uas':
            if best_dev_uas < new_dev_uas:
                best_dev_uas = new_dev_uas
                improved=True
        elif what == 'las':
            if best_dev_las < new_dev_las:
                best_dev_las = new_dev_las
                improved=True
        else:
            exit(f"Unknown --early-stopping-improve-on , {what}, . Should be either 'las' or 'uas'")


        if improved:
            best_epoch = epoch
            best_msg = '' + dev_uas + ' ' + dev_las
            num_epochs_since_best = 0
            print('current best')
            print('saving', model_name)
            pat.save(model_name)
        else:
            num_epochs_since_best += 1
            print('no improvement for', num_epochs_since_best, 'epochs')
            if num_epochs_since_best >= args.max_epochs_without_improvement:
                print('quitting')
                print('-' * 50)
                print('best model found in epoch', best_epoch)
                print(str(args.cnn_ce), " ", str(args.cnn_ce_kernel_size), " ", str(args.cnn_ce_out_channels), " ", str(args.cnn_embeddings_size))
                print('  dev uas and las:', best_msg, ' ', what)
                print('\n\n\n\n')
                sys.exit()


    else:
        print('saving', model_name)
        pat.save(model_name)
    print('-' * 50)
