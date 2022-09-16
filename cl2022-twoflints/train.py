import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dataloader import DataLoader
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from pytorch_pretrained_bert.tokenization import BertTokenizer

import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.set_defaults(lower=False)

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--device', type=int, default=0, help='gpu device to use.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--warmup_prop', type=float, default=0.3, help='Proportion of training to perform linear learning rate warmup for.')
parser.add_argument('--burnin', type=int, default=5)

parser.add_argument("--eval_per_epoch", default=10, type=int, help="How many times it evaluates on dev set per epoch")
parser.add_argument("--top", default=1, type=int)

args = parser.parse_args()

opt = vars(args)
if "tacred" in opt["data_dir"]:
    label2id = constant.LABEL_TO_ID_tacred
else:
    label2id = constant.LABEL_TO_ID_conll
    
opt['num_class'] = len(label2id)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
    args.device = None
elif args.cuda:
    with torch.cuda.device(args.device):
        torch.cuda.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')

train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, tokenizer, False, opt['data_dir'] + '/tagging_train.txt')
train_num_example = train_batch.num_examples
train_batch = list(train_batch)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, tokenizer)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

global_step = 0
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

opt['train_batch'] = len(train_batch)

# model
if not opt['load']:
    trainer = BERTtrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file'] 
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = BERTtrainer(model_opt)
    trainer.load(model_file)  

id2label = dict([(v,k) for k,v in label2id.items()])

current_lr = opt['lr']

eval_step = max(1, len(train_batch) // args.eval_per_epoch)
dev_score_history = []
# start training
durations = list()
for epoch in range(1, opt['num_epoch']+1):
    
    train_loss = 0
    random.shuffle(train_batch)
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss, current_lr = trainer.update(batch, epoch)
        if args.cuda:
            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))
        if (i + 1) % eval_step == 0:
            # eval on dev
            print("Evaluating on dev set...")
            predictions = []
            dev_loss = 0
            for _, batch in enumerate(dev_batch):
                preds, tags, dloss = trainer.predict(batch, id2label, tokenizer)
                predictions += preds
                dev_loss += dloss
            predictions = [id2label[p] for p in predictions]
            train_loss = train_loss / train_num_example * opt['batch_size'] # avg loss per batch
            dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

            dev_p, dev_r, dev_f1, bi_acc = scorer.score(dev_batch.gold(), predictions)
            print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}, binary_accuracy = {:.4f}".format(epoch,\
                train_loss, dev_loss, dev_f1, bi_acc))
            dev_score = dev_f1
            file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))

            # save
            if dev_score_history == [] or dev_score > max(dev_score_history):
                model_file = model_save_dir + '/best_model.pt'
                trainer.save(model_file)
                print("new best model saved.")
                file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
                    .format(epoch, dev_p*100, dev_r*100, dev_score*100))

            dev_score_history += [dev_score]
            print("")
print("Training ended with {} epochs.".format(epoch))
