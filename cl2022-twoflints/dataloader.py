"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import string

from utils import constant, helper
from collections import defaultdict
from statistics import mean

from termcolor import colored

# hide_relations = ["per:employee_of", "per:age", "org:city_of_headquarters", "org:country_of_headquarters", "org:stateorprovince_of_headquarters", "per:origin"]

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, tokenizer, do_eval = True, tagging = None):
        self.batch_size = batch_size
        self.opt = opt
        self.label2id = constant.LABEL_TO_ID
        self.tokenizer = tokenizer
        self.do_eval = do_eval

        if not do_eval:
            assert tagging is not None
            with open(tagging) as f:
                self.tagging = f.readlines()

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, opt)

        if not do_eval:
            data = sorted(data, key=lambda f: len(f[0]))
        
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-2]] for d in data]
        self.words = [d[-1] for d in data]
        self.num_examples = len(data)
        
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(self.data), filename))

    def preprocess(self, data, opt):

        missed = 0
        """ Preprocess the data and convert to ids. """
        processed = []
        for c, d in enumerate(data):
            tokens = list()
            words  = list()
            origin = list()
            if not self.do_eval:
                _, tagged = self.tagging[c].split('\t')
                tagged = eval(tagged)
            else:
                tagged = []
            tagging_mask = list()

            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            subj = []
            obj = []
            for i, t in enumerate(d['token']):
                if i == ss:
                    words.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+d['subj_type']+']']+1))
                    tagging_mask.append(0)
                if i == os:
                    words.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+d['obj_type']+']']+1))
                    tagging_mask.append(0)
                if i>=ss and i<=se:
                    # for sub_token in self.tokenizer.tokenize(t):
                    #     subj.append(sub_token)
                    origin.append((colored(t, "blue"), [len(words)]))
                elif i>=os and i<=oe:
                    # for sub_token in self.tokenizer.tokenize(t):
                    #     obj.append(sub_token)
                    origin.append((colored(t, "yellow"), [len(words)]))
                else:
                    t = convert_token(t)
                    origin.append((t, range(len(words)+1, len(words)+1+len(self.tokenizer.tokenize(t)))))
                    for j, sub_token in enumerate(self.tokenizer.tokenize(t)):
                        words.append(sub_token)
                        if i in tagged and j == len(self.tokenizer.tokenize(t))-1:
                            tagging_mask.append(1)
                        else:
                            tagging_mask.append(0)
            
            words = ['[CLS]'] + words + ['[SEP]']
            relation = self.label2id[d['relation']]
            tagging_mask = [0]+tagging_mask+[0]
            tokens = self.tokenizer.convert_tokens_to_ids(words)
            if len(tokens) > self.opt['max_length']:
                tokens = tokens[:self.opt['max_length']]
                tagging_mask = tagging_mask[:self.opt['max_length']]
            mask = [1] * len(tokens)
            segment_ids = [0] * len(tokens)
            if self.do_eval:
                processed += [(tokens, mask, segment_ids, tagging_mask, sum(tagging_mask)!=0, relation, origin)]
            elif len([aa for aa in tokens if aa>0 and aa<20]) == 2:# or relation == 0) and d['relation'] not in hide_relations:
                processed += [(tokens, mask, segment_ids, tagging_mask, sum(tagging_mask)!=0, relation, origin)]
                
            # if sum(tagging_mask)!=0:
            #     print (d['token'])
            #     print (words)
            #     print ([w for i,w in enumerate(d['token']) if i in tagged])
            #     print ([w for i, w in enumerate(words) if tagging_mask[i]==1])
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        
        # word dropout
        words = batch[0]
        mask = batch[1]
        segment_ids = batch[2]
        tagging_mask = batch[3]
        # convert to tensors
        words = get_long_tensor(words, batch_size)
        mask = get_long_tensor(mask, batch_size)
        segment_ids = get_long_tensor(segment_ids, batch_size)
        tagging_mask = get_long_tensor(tagging_mask, batch_size)

        rels = torch.LongTensor(batch[-2])#

        return (words, mask, segment_ids, tagging_mask, batch[4], rels)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max([len(x) for x in tokens_list])
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i,:len(s)] = torch.LongTensor(s)
    return tokens

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token
