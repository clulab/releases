#!/usr/bin/env python
from __future__ import division

import sys
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt



def read_gold_mentions(filename):
    counts, entities, labels = [], [], []
    with open(filename) as f:
        for line in f:
            [count, entity, label] = line.strip().split('\t')
            counts.append(float(count))
            entities.append(entity)
            labels.append(label)
    return np.array(counts), np.array(entities), np.array(labels)



def majority_gold(counts, entities, labels):
    unique_entities = np.unique(entities)
    unique_labels = []
    for entity in unique_entities:
        entity_counts = counts[entities==entity]
        entity_labels = labels[entities==entity]
        max_label = entity_labels[np.argmax(entity_counts)]
        unique_labels.append(max_label)
    return unique_entities, np.array(unique_labels)



def parse_log_file(filename):
    epoch_data = []
    data = None
    pattern = re.compile('^[Ee]poch (\d+)')
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line: # skip empty lines
                continue
            match = pattern.match(line)
            if match: # new epoch
                if data:
                    epoch_data.append(data)
                data = {}
                continue
            chunks = line.split('\t')
            label = chunks[0]
            entities = chunks[1:]
            data[label] = entities
    if data:
        epoch_data.append(data)
    return epoch_data




def eval_entity(entity, label, entities, labels):
    correct_labels = labels[entities==entity]
    return 1.0 if label in correct_labels else 0.0



# expects a structure similar to this:
# epoch_data = [
#     {"PER": ["Clinton", "Bush"], "LOC": ...},
#     {"PER": ["John Doe", "someone else"], "LOC": ...},
# ]
def evaluate_epochs(epoch_data, entities, labels):
    results = defaultdict(list)
    for epoch in epoch_data:
        for label in epoch:
            scores = []
            for e in epoch[label]:
                scores.append(eval_entity(e, label, entities, labels))
            results[label].append(scores)
    return results



# expects the result of evaluate_epochs
def accumulate(results):
    new_results = defaultdict(list)
    for label in results:
        prev = []
        for epoch in results[label]:
            prev += epoch
            new_results[label].append(prev[:]) # append prev copy
    return new_results



def count(results):
    new_results = defaultdict(list)
    for label in results:
        for epoch in results[label]:
            new_results[label].append(len(epoch))
    return new_results



def precision(results):
    prec = defaultdict(list)
    for label in results:
        for epoch in results[label]:
            p = sum(epoch) / len(epoch)
            prec[label].append(p)
    return prec



def plot_lines(filename, title, prec_per_label, count_per_label):
    for k in ['PER', 'LOC', 'ORG', 'MISC']:
        plt.plot(count_per_label[k], prec_per_label[k], label=k)
    plt.title(title)
    plt.xlabel('throughput')
    plt.ylabel('precision')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.savefig(filename)
    plt.close()

def score_results(logFile):
    gold_counts, gold_entities, gold_labels = read_gold_mentions('conll_entity_label_counts.txt')
    majority_entities, majority_labels = majority_gold(gold_counts, gold_entities, gold_labels)
    results = parse_log_file(logFile)
    scores = evaluate_epochs(results, majority_entities, majority_labels)
    cum_scores = accumulate(scores)
    prec_scores = precision(cum_scores)
    res = ( (prec_scores['PER'][-1]*count(cum_scores)['PER'][-1]) + (prec_scores['ORG'][-1]*count(cum_scores)['ORG'][-1]) + (prec_scores['LOC'][-1]*count(cum_scores)['LOC'][-1]) + (prec_scores['MISC'][-1]*count(cum_scores)['MISC'][-1]) ) / 4
    return res

if __name__ == '__main__':
    gold_counts, gold_entities, gold_labels = read_gold_mentions('../data/conll_entity_label_counts.txt')
    majority_entities, majority_labels = majority_gold(gold_counts, gold_entities, gold_labels)

    results = parse_log_file(sys.argv[1] )# e.g. 'pools_output.txt')
    scores = evaluate_epochs(results, majority_entities, majority_labels)
    cum_scores = accumulate(scores)
    prec_scores = precision(cum_scores)

    print ("-----------------")
    print (prec_scores)
    print ("-----------------")
    print (count(cum_scores))
    print ("-----------------")
    
    print (prec_scores['PER'][-1])
    print (count(cum_scores)['PER'][-1])
    print ("PER res = " + str(prec_scores['PER'][-1]*count(cum_scores)['PER'][-1]))
    
    print (prec_scores['ORG'][-1])
    print (count(cum_scores)['ORG'][-1])
    print ("ORG res = " + str(prec_scores['ORG'][-1]*count(cum_scores)['ORG'][-1]))
    
    print (prec_scores['LOC'][-1])
    print (count(cum_scores)['LOC'][-1])
    print ("LOC res = " + str(prec_scores['LOC'][-1]*count(cum_scores)['LOC'][-1]))
    
    print (prec_scores['MISC'][-1])
    print (count(cum_scores)['MISC'][-1])
    print ("MISC res = " + str(prec_scores['MISC'][-1]*count(cum_scores)['MISC'][-1]))
    
    res = ( (prec_scores['PER'][-1]*count(cum_scores)['PER'][-1]) + (prec_scores['ORG'][-1]*count(cum_scores)['ORG'][-1]) + (prec_scores['LOC'][-1]*count(cum_scores)['LOC'][-1]) + (prec_scores['MISC'][-1]*count(cum_scores)['MISC'][-1]) ) / 4
    print (res)
    
   # plot_lines('ajay_results.pdf', "gupta et. al. 2015 - Our implementation -- 20 epochs, scienceIE dataset", prec_scores, count(cum_scores))
    plot_lines(sys.argv[2], "embeddings results", prec_scores, count(cum_scores))

