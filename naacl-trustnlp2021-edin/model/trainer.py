"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from model.decoder import Decoder
from utils import constant, torch_utils

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'classifier': self.classifier.state_dict(),
                'decoder': self.decoder.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    rules = None
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        inputs += [Variable(batch[13].cuda())]
        labels = Variable(batch[10].cuda())
        rules  = Variable(batch[12]).cuda()
    else:
        inputs = [Variable(b) for b in batch[:10]]
        inputs += [Variable(batch[13])]
        labels = Variable(batch[10])
        rules  = Variable(batch[12])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.classifier = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.decoder = Decoder(opt)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_d = nn.NLLLoss(ignore_index=constant.PAD_ID)
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.decoder.parameters() if p.requires_grad]
        if opt['cuda']:
            self.classifier.cuda()
            self.decoder.cuda()
            self.criterion.cuda()
            self.criterion_d.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.classifier.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        loss = 0
        # classifier
        logits, pooling_output, encoder_outputs, hidden = self.classifier(inputs)
        if self.opt['classifier']:
            loss = self.criterion(logits, labels)
            # l2 decay on all conv layers
            if self.opt.get('conv_l2', 0) > 0:
                loss += self.classifier.conv_l2() * self.opt['conv_l2']
            # l2 penalty on output representations
            if self.opt.get('pooling_l2', 0) > 0:
                loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        if self.opt['decoder']:
            # decoder
            batch_size = labels.size(0)
            rules = rules.view(batch_size, -1)
            masks = inputs[1]
            max_len = rules.size(1)
            rules = rules.transpose(1,0)
            output = Variable(torch.LongTensor([constant.SOS_ID] * batch_size)) # sos
            output = output.cuda() if self.opt['cuda'] else output
            loss_d = 0
            h0 = hidden.view(self.opt['num_layers'], batch_size, -1)
            c0 = hidden.view(self.opt['num_layers'], batch_size, -1)
            decoder_hidden = (h0, c0)
            for t in range(1, max_len):
                output, decoder_hidden, attn_weights = self.decoder(
                        output, masks, decoder_hidden, encoder_outputs)
                loss_d += self.criterion_d(output, rules[t])
                output = rules.data[t]
                if self.opt['cuda']:
                    output = output.cuda()
            loss += loss_d/max_len if (self.opt['classifier'] and max_len!=0) else loss_d
        if loss != 0:
            loss_val = loss.item()
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.opt['max_grad_norm'])
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.opt['max_grad_norm'])
            self.optimizer.step()
        else:
            loss_val = 0
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.classifier.eval()
        self.decoder.eval()
        logits, hidden, encoder_outputs, hidden = self.classifier(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        # decoder
        batch_size = labels.size(0)
        decoded = []
        masks = inputs[1]
        output = Variable(torch.LongTensor([constant.SOS_ID] * batch_size)) # sos
        output = output.cuda() if self.opt['cuda'] else output
        decoded = torch.zeros(constant.MAX_RULE_LEN, batch_size)
        decoded[0] = output
        if self.opt['cuda']:
                decoded = decoded.cuda()
        h0 = hidden.view(self.opt['num_layers'], batch_size, -1)
        c0 = hidden.view(self.opt['num_layers'], batch_size, -1)
        decoder_hidden = (h0, c0)
        for t in range(1, constant.MAX_RULE_LEN):
            output, decoder_hidden, attn_weights = self.decoder(
                    output, masks, decoder_hidden, encoder_outputs)
            topv, topi = output.data.topk(1)
            output = topi.view(-1)
            decoded[t] = output
        decoded = decoded.transpose(0, 1).tolist()
        if unsort:
            _, decoded, words = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    decoded, inputs[0])))]
        return predictions, words, decoded, loss.item()