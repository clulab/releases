"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bert import BERTencoder, BERTclassifier, Tagger
from utils import constant, torch_utils

from pytorch_pretrained_bert.optimization import BertAdam

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.tagger.load_state_dict(checkpoint['tagger'])
        device = self.opt['device']
        self.opt = checkpoint['config']
        self.opt['device'] = device

    def save(self, filename):
        params = {
                'classifier': self.classifier.state_dict(),
                'encoder': self.encoder.state_dict(),
                'tagger': self.tagger.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda, device):
    if cuda:
        with torch.cuda.device(device):
            inputs = [batch[i].to('cuda') for i in range(4)]
            labels = Variable(batch[-1].cuda())
    else:
        inputs = [Variable(batch[i]) for i in range(4)]
        labels = Variable(batch[-1])
    return inputs, labels, batch[4]

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.tagger = Tagger()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion2 = nn.BCELoss()

        param_optimizer = list(self.classifier.named_parameters())+list(self.encoder.named_parameters())+list(self.tagger.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if opt['cuda']:
            with torch.cuda.device(self.opt['device']):
                self.encoder.cuda()
                self.classifier.cuda()
                self.tagger.cuda()
                self.criterion.cuda()
                self.criterion2.cuda()

        self.optimizer = BertAdam(optimizer_grouped_parameters,
             lr=opt['lr'],
             warmup=opt['warmup_prop'],
             t_total= opt['train_batch'] * self.opt['num_epoch'])

    def update(self, batch, epoch):
        inputs, labels, has_tag = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.classifier.train()
        self.tagger.train()

        h, b_out = self.encoder(inputs)
        tagging_output = self.tagger(h)
        
        loss = self.criterion2(b_out, (~(labels.eq(0))).to(torch.float32).unsqueeze(1))
        for i, f in enumerate(has_tag):
            if f:
                loss += self.criterion2(tagging_output[i], inputs[3][i].unsqueeze(1).to(torch.float32))
                logits = self.classifier(h[i], inputs[0][i].unsqueeze(0), inputs[3][i].unsqueeze(0))
                loss += self.criterion(logits, labels.unsqueeze(1)[i])
            elif labels[i] != 0 and epoch > self.opt['burnin']:
                tag_cands, n = self.tagger.generate_cand_tags(tagging_output[i], self.opt['device'])
                if n != -1 and len(tag_cands)!=0:
                    logits = self.classifier(h[i], torch.cat(n*[inputs[0][i].unsqueeze(0)], dim=0), tag_cands)
                    best = np.argmax(logits.data.cpu().numpy(), axis=0).tolist()[labels[i]]
                    loss += self.criterion(logits[best].unsqueeze(0), labels.unsqueeze(1)[i])
                else:
                    print (n, tag_cands)

        loss_val = loss.item()

        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        h = b_out = logits = inputs = labels = None
        return loss_val, self.optimizer.get_lr()[0]

    def predict(self, batch, id2label, tokenizer):
        inputs, labels, has_tag = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        # forward
        self.encoder.eval()
        self.classifier.eval()
        self.tagger.eval()
        with torch.no_grad():
            h, b_out = self.encoder(inputs)
            tagging_output = self.tagger(h)
            words = inputs[0]
            ent_mask = torch.logical_and(words.unsqueeze(2).ge(0), words.unsqueeze(2).lt(9))
            tagging_mask = torch.round(tagging_output).squeeze(2)
            tagging_max = np.argmax(tagging_output.masked_fill(ent_mask, -constant.INFINITY_NUMBER).squeeze(2).data.cpu().numpy(), axis=1)
            tagging = torch.round(tagging_output).squeeze(2)
            logits = self.classifier(h, inputs[0], tagging_mask)
            probs = F.softmax(logits, 1) * torch.round(b_out)
        loss = self.criterion2(b_out, (~(labels.eq(0))).to(torch.float32).unsqueeze(1)) + self.criterion(logits, labels).item()
        for i, f in enumerate(has_tag):
            if f:
                loss += self.criterion2(tagging_output[i], inputs[3][i].unsqueeze(1).to(torch.float32))
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        tags = []
        for i, p in enumerate(predictions):
            if p != 0:
                t = tagging[i].data.cpu().numpy().tolist()
                if sum(t) == 0:
                    t[tagging_max[i]] = 1
                tags += [t]
            else:
                tags += [[]]
        return predictions, tags, loss
        

