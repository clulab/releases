from typing import Dict, List
import torch
import random
import numpy as np  
import json
from torch import tensor
import tqdm
from pathlib import Path

def init_random(seed):
    """
    Init torch, torch.cuda and numpy with same seed. For reproducibility.
    :param seed: Random number generator seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_arrow_dataset(path):
    import datasets

    # Full data
    dataset = datasets.load_from_disk('/home/rvacareanu/temp/huggingface_datasets_100k_complete')
    def easy_split(x):
        sentence_length = len(x['text'][0].split(' '))
        highlight_length = int(x['text'][2]) - int(x['text'][1])
        return (sentence_length < 20 and highlight_length < 5)

    def medium_split(x):
        sentence_length = len(x['text'][0].split(' '))
        highlight_length = int(x['text'][2]) - int(x['text'][1])

        return (sentence_length < 30 and highlight_length < 7)

    dataset_1 = dataset.filter(function = lambda x: easy_split(x))
    dataset_2 = dataset.filter(function = lambda x: medium_split(x))
    # dataset_3 = dataset.filter(function = lambda x: not easy_split(x))
    dataset_1.save_to_disk('/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1')
    dataset_2.save_to_disk('/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2')

    return

"""
    :param words_of_interests: offsets for each word of interest (in the original input space)
    Returns a list of ints. This can be used to identify the original words.

    Example:    
    >> s = ["Nothingness", "is", "nothingness", "obviously"]
    >> starts = [1]
    >> ends   = [3]
    >> ast = False
    >> sentence_tokenized = tokenizer(s, truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=ast, return_offsets_mapping=True)
    >> words_to_highlight = []
    >> if ast:
    >>     starts = [s+1 for s in starts]
    >>     ends   = [s+1 for s in ends]
    >> for s, e in zip(starts, ends):
    >>     words_to_highlight += range(s, e)
    >> index = highlighted_indices_tokenization_space(words_to_highlight, sentence_tokenized['offset_mapping'])
    >> sentence_tokenized['token_type_ids'][0][index] = 1
    >> text = np.array([tokenizer.decode(x) for x in sentence_tokenized['input_ids'][0]])
    >> print(text)
    >> print(text[sentence_tokenized['token_type_ids'][0] == 1])

    highlighted_indices_tokenization_space: Highlight the complete word. A subword tokenization might split the highlighted word into multiple word pieces. This highlights all pieces
    highligh_word_start_tokenization_space: Highlight the first piece. A subword tokenization might split the highlighted word into multiple word pieces. This highlights only the first one
"""
def highlighted_indices_tokenization_space(words_of_interest: List[int], offset_mapping: tensor) -> List[int]:
    sentence_idx = torch.arange(offset_mapping.shape[1]).unsqueeze(dim=1)  # prepare an idx to make operation easier
    sentence_offsets_with_idx = torch.cat([sentence_idx, offset_mapping[0]], dim=1) # append idx
    word_start = sentence_offsets_with_idx[sentence_offsets_with_idx[:,1] == 0] # keep only the ones which represents the start of a new word
    last_word_start = word_start.shape[0]
    indices = []

    for woi in words_of_interest:
        if woi < last_word_start - 1:
            indices += range(word_start[woi][0], word_start[woi+1][0])
        elif woi == last_word_start - 1:
            indices += range(word_start[woi][0], sentence_offsets_with_idx.shape[0])
        else:
            raise ValueError("Outside sentence boundaries")

    return indices

# only start of the word
def highligh_word_start_tokenization_space(words_of_interest: List[int], offset_mapping: tensor) -> List[int]:
    sentence_idx = torch.arange(offset_mapping.shape[1]).unsqueeze(dim=1)  # prepare an idx to make operation easier
    sentence_offsets_with_idx = torch.cat([sentence_idx, offset_mapping[0]], dim=1) # append idx
    word_start = sentence_offsets_with_idx[sentence_offsets_with_idx[:,1] == 0] # keep only the ones which represents the start of a new word
    last_word_start = word_start.shape[0]
    indices = []

    for woi in words_of_interest:
        if woi < last_word_start:
            indices.append(word_start[woi][0].item())
        else:
            raise ValueError("Outside sentence boundaries")

    return indices

# start of the word in a list, the continuation of each word in another list; can also be obtained by doing a diff 
# between highlighted_indices_tokenization_space and highligh_word_start_tokenization_space
def highlighted_start_continuation_indices_tokenization_space(words_of_interest: List[int], offset_mapping: tensor) -> Dict[str, List[int]]:
    sentence_idx = torch.arange(offset_mapping.shape[1]).unsqueeze(dim=1)  # prepare an idx to make operation easier
    sentence_offsets_with_idx = torch.cat([sentence_idx, offset_mapping[0]], dim=1) # append idx
    word_start = sentence_offsets_with_idx[sentence_offsets_with_idx[:,1] == 0] # keep only the ones which represents the start of a new word
    last_word_start = word_start.shape[0]
    indices = []

    start_indices        = []
    continuation_indices = []

    for woi in words_of_interest:
        start_indices.append(word_start[woi][0].item())
        if woi < last_word_start - 1:
            continuation_indices += range(word_start[woi][0]+1, word_start[woi+1][0])
        elif woi == last_word_start - 1:
            continuation_indices += range(word_start[woi][0]+1, sentence_offsets_with_idx.shape[0])
        else:
            raise ValueError("Outside sentence boundaries")

    return {"word_start": start_indices, "continuation": continuation_indices}

def no_highlight(words_of_interest: List[int], offset_mapping: tensor) -> List[int]:
    return []


def read_spec(name):
    with open(name) as f:
        [spec, doc] = list(f)
        spec = json.loads(spec)
        doc = json.loads(doc)
        return (spec, doc)

def read_steps(name):
    return list(iter_steps(name))

def iter_steps(name):
    with open(name) as f:
        for line in f:
            d = json.loads(line)
            yield [
                # the first element should have a higher score than the rest
                d['next_correct_rule'],
                # the second element is correct, but should score lower than the first
                d['current_rule'],
                # the rest of the elements are incorrect
                *d['next_incorrect_rules']
            ]

"""
    data_dir    -> a directory containing a folder named specs and a folder named steps
                   The specs folder should contain the specification (Odinson Document and user highlights)
    out_path    -> where to save the new file
    output_dual -> if True, the output will be of the form:
                    '\t'.join([sentence, highlight_start, highlight_end, current_rule, next_correct_rule, next_incorrect_rule])
                   if False, the output will be of the form:
                    '\t'.join([sentence, highlight_start, highlight_end, current_rule, next_potential_rule, is_it_correct])
                   In other words, if output_dual is True, the output file can be used for pairwise comparison. If
                   it is False, the output is suitable only for pointwise comparison (but potentially using the current_rule)
"""
def unroll(data_dir_str, out_path, output_dual=False):
    data_dir = Path(data_dir_str)
    all_names = [p.name[:-10] for p in (data_dir/'specs/').glob('*.spec.json')]
    fout = open(out_path, 'w+')
    for name in tqdm.tqdm(all_names):
        # Read selections and sentences
        selections, doc = read_spec(data_dir/f'specs/{name}.spec.json')
        selections = sorted(selections['specs'], key=lambda x: x['sentId'])
        sentences = []
        for s in doc['sentences']:
            for field in s['fields']:
                if field['name'] == 'word':
                    sentences.append(field['tokens'])
        # Read the steps
        steps = read_steps(data_dir/f'steps/{name}.steps.json')

        # unrolled_steps
        for step in steps:
            cr   = step[1]
            ncr  = step[0]
            nirs = step[2:]
            if output_dual:
                for nir in nirs:
                    for sel, sen in zip(selections, sentences):
                        start = sel['start']
                        end   = sel['end']
                        jsen  = ' '.join(sen)
                        fout.write(f'{jsen}\t{start}\t{end}\t{cr}\t{ncr}\t{nir}\n')
            else:
                for sel, sen in zip(selections, sentences):
                    start = sel['start']
                    end   = sel['end']
                    jsen  = ' '.join(sen)
                    fout.write(f'{jsen}\t{start}\t{end}\t{cr}\t{ncr}\t1\n')
                    for nir in nirs:
                        fout.write(f'{jsen}\t{start}\t{end}\t{cr}\t{nir}\t0\n')
    fout.close()

"""
    Extract the tokens corresponding to the given field_name
    :param doc - odinson document as a json
    :param field_name - str, corresponding to the field name from where
                        to extract the tokens (e.g. 'word', 'lemma') 
"""
def extract_sentences_from_odinson_doc(doc, field_name = 'word'):
    sentences = []
    for s in doc['sentences']:
        for field in s['fields']:
            if field['name'] == field_name:
                sentences.append(field['tokens'])
    
    return sentences


"""
Split a file where each line represents a standalone example into train and test partition.
"""
def train_test_split_lbl(path, size=0.25, random_seed=1):
    init_random(random_seed)
    fin = open(path)
    name = '.'.join(path.split('.')[:-1])
    ext  = path.split('.')[-1]
    fout1 = open(f'{name}_train.{ext}', 'w+')
    fout2 = open(f'{name}_test.{ext}', 'w+')
    for line in tqdm.tqdm(fin):
        if random.random() < size:
            fout2.write(line)
        else:
            fout1.write(line)

    fin.close()
    fout1.close()
    fout2.close()


def calc_reciprocal_ranks(scores_per_step):
    """assumes first score is correct score"""
    reciprocal_ranks = []
    for scores in scores_per_step:
        correct_score = scores[0]
        incorrect_scores = scores[1:]
        # repeat correct score for the pairwise comparison
        correct_scores = correct_score.expand(len(incorrect_scores))
        # reciprocal rank - how many incorrect scores are higher than the correct score
        rr = 1 / ((incorrect_scores >= correct_scores).sum() + 1)
        reciprocal_ranks.append(rr.unsqueeze(dim=0))
    return torch.cat(reciprocal_ranks)


class CumulativeMovingAverage:

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.average = 0.0

    def __float__(self):
        return float(self.average)

    def __int__(self):
        return int(self.average)

    def __format__(self, format_spec):
        return format(self.average, format_spec)

    def __iadd__(self, x):
        self.n += 1
        self.total += float(x)
        self.average = self.total / self.n
        return self


"""
Have a constraint on the number of holes
The constraint here:
    - no more than 2 hole queries (e.g. □ □ □ is not allowed)
    - no more than 1 hole constraint (e.g. [□] [□] is not allowed)
    - no more than 2 hole matcher (e.g. [□=□] [□=□] is not allowed)
"""
def filter_query_holes(query) -> bool:
    number_of_holes = query.num_holes_by_type()
    return number_of_holes[0] < 3 and number_of_holes[1] < 2 and number_of_holes[2] < 3


