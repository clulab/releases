from typing import List
from utils import extract_sentences_from_odinson_doc

from queryparser import QueryParser
from torch.utils.data import Dataset

import random
import json

class OdinsynthDataset(Dataset):

    def __init__(self, names, num_steps_per_name, spec_length, specs_dir, steps_dir):
        self.names = names
        self.num_steps_per_name = num_steps_per_name # list[int] - number of steps for each name
        self.spec_length = spec_length
        self.specs_dir = specs_dir
        self.steps_dir = steps_dir

        self.index_to_name_line = [] # list[(name, lineno)]
        for name, num_steps, _ in zip(self.names, self.num_steps_per_name, self.spec_length):
            for i in range(num_steps):
                self.index_to_name_line.append((name, i))


    def __len__(self):
        return len(self.index_to_name_line)

    def __getitem__(self, index):
        # read data
        name, lineno = self.index_to_name_line[index]
        selections, doc = self.read_spec(name)
        # NOTE we're trying to return a single step
        # but we're reading all of them and selecting the one we want
        # and wrapping it in a list so that we don't modify the code below
        steps = [self.read_steps(name)[lineno]][0]

        sels = selections['specs']
        # Defensively check the order of specs
        if sorted(selections['specs'], key=lambda x: x['sentId']) != selections['specs']:
            sels = sorted(selections['specs'], key=lambda x: x['sentId'])
            # raise ValueError('Specs are not in order')
        
        sentences = [' '.join(x) for x in extract_sentences_from_odinson_doc(doc)]

        # return item
        data = []
        current = []
        # The correct one
        for sel, sen in zip(sels, sentences):
            current.append({'text' : [sen, sel['start'], sel['end'], steps[1], steps[0], 1]})
        data.append(current)
        current = []
        for sel, sen in zip(sels, sentences):
            current.append({'text' : [sen, sel['start'], sel['end'], steps[1], steps[1], 0]})
        data.append(current)
        shuffled = steps[2:]
        random.shuffle(shuffled)
        for pattern in shuffled[:16]:
            current = []
            for sel, sen in zip(sels, sentences):
                current.append({'text': [sen, sel['start'], sel['end'], steps[1], pattern, 0]})
            data.append(current)

        return data

    def read_spec(self, name):
        filename = self.specs_dir/f'{name}.spec.json'
        with open(filename) as f:
            [spec, doc] = list(f)
            spec = json.loads(spec)
            doc = json.loads(doc)
        return (spec, doc)

    def read_steps(self, name):
        return list(self.iter_steps(name))

    def iter_steps(self, name):
        filename = self.steps_dir/f'{name}.steps.json'
        result = []
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                result.append([
                    # the first element should have a higher score than the rest
                    d['next_correct_rule'],
                    # the second element is correct, but should score lower than the first
                    d['current_rule'],
                    # the rest of the elements are incorrect
                    *d['next_incorrect_rules']
                ])
        return result



from sklearn.model_selection import train_test_split

def make_databundle(
    specs_dir,
    steps_dir,
    **kwargs,
):
    # with open('/data/nlp/corpora/odinsynth/data/rules100k_unrolled/train_names') as fin:
        # lines = fin.readlines()
        # lines = [l.split('\t')[1].split('/')[-1] for l in lines]
    all_names = [p.name[:-10] for p in specs_dir.glob('*.spec.json')]
    # all_names = [p[:-11] for p in all_names]
    train_names, valid_names = train_test_split(all_names, test_size=0.25)
    # valid_names, _ = train_test_split(valid_names, test_size=0.5, **kwargs)
    num_steps_train = [num_steps(steps_dir, name) for name in train_names]
    num_steps_valid = [num_steps(steps_dir, name) for name in valid_names]
    spec_lengths_train = [spec_length(specs_dir, name) for name in train_names]
    spec_lengths_valid = [spec_length(specs_dir, name) for name in valid_names]
    train_ds = OdinsynthDataset(train_names, num_steps_train, spec_lengths_train, specs_dir, steps_dir)
    valid_ds = OdinsynthDataset(valid_names, num_steps_valid, spec_lengths_valid, specs_dir, steps_dir)
    return {'train': train_ds, 'test': valid_ds}
    
def make_databundle_from_names(
    specs_dir,
    steps_dir,
    path = '/data/nlp/corpora/odinsynth/data/rules100k_unrolled/train_names',
):
    with open(path) as fin:
        lines = fin.readlines()
        lines = [l.split('\t')[1].split('/')[-1] for l in lines]
        
    all_names = [p[:-11] for p in lines]
    train_names, valid_names = train_test_split(all_names, test_size=0.25)
    # valid_names, _ = train_test_split(valid_names, test_size=0.5, **kwargs)
    num_steps_train = [num_steps(steps_dir, name) for name in train_names]
    num_steps_valid = [num_steps(steps_dir, name) for name in valid_names]
    spec_lengths_train = [spec_length(specs_dir, name) for name in train_names]
    spec_lengths_valid = [spec_length(specs_dir, name) for name in valid_names]
    train_ds = OdinsynthDataset(train_names, num_steps_train, spec_lengths_train, specs_dir, steps_dir)
    valid_ds = OdinsynthDataset(valid_names, num_steps_valid, spec_lengths_valid, specs_dir, steps_dir)
    return {'train': train_ds, 'test': valid_ds}

def num_steps(steps_dir, name):
    # file is json-lines, so each line is a step
    with open(steps_dir/f'{name}.steps.json') as f:
        return sum(1 for line in f)

def spec_length(specs_dir, name):
    with open(specs_dir/f'{name}.spec.json') as f:
        lines = f.readlines()
        return max([x['numTokens'] for x in json.loads(lines[1])['sentences']])
