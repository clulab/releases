#!/usr/bin/env python
from __future__ import division

import sys
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from operator import add

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
        entity_counts = counts[entities == entity]
        entity_labels = labels[entities == entity]
        max_label = entity_labels[np.argmax(entity_counts)]
        unique_labels.append(max_label)
    return unique_entities, np.array(unique_labels)


## NOTE: Assumes that epoch 0 consist of the seeds.
def parse_log_file(filename):
    epoch_data = None
    data = None
    trial_data = []
    pattern = re.compile('^[Ee]poch (\d+)')
    pattern2 = re.compile('^[Tt]rial (\d+)')
    eid = 0
    tid = 0

    multitrialflag = True
    with open(filename) as f:
        peek = f.readline()
        match = pattern.match(peek)
        if match:
            print("Single trial file")
            multitrialflag = False
        else:
            print("Multi-trial file")
    f.close()

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            match = pattern.match(line)
            match2 = pattern2.match(line)
            if match2:  # new trial
                # print("Starting new trial" + str(tid))
                eid = 0
                tid += 1
                if epoch_data:
                    if data:
                        epoch_data.append(data)
                    trial_data.append(epoch_data)
                data = {}
                epoch_data = []
                continue

            if match:  # new epoch
                # print("Starting new epoch" + str(eid))
                eid += 1
                if multitrialflag == False:
                    epoch_data = []
                    multitrialflag = True  ## NOTE: Reset the flag so that the computation proceeds as multi-trial file
                if data:
                    epoch_data.append(data)
                data = {}
                continue
            chunks = line.split('\t')
            label = chunks[0]
            entities = chunks[1:]
            data[label] = entities

    ## Add the last data / epoch_data to the list
    if data:
        epoch_data.append(data)
    if epoch_data:
        trial_data.append(epoch_data)

    # print("Done parsing file " + filename)
    # print(len(trial_data))
    return trial_data


def eval_entity(entity, label, entities, labels):
    correct_labels = labels[entities == entity]
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
            new_results[label].append(prev[:])  # append prev copy
    return new_results


def count(results):
    new_results = defaultdict(list)
    for label in results:
        for epoch in results[label]:
            new_results[label].append(len(epoch))
    # print (" COUNT CUM SCORES (in function) : " + str(new_results))
    return new_results


def precision(results):
    prec = defaultdict(list)
    for label in results:
        for epoch in results[label]:
            p = sum(epoch) / len(epoch)
            prec[label].append(p)
    return prec

def plot_lines(filename, title, prec_per_label, count_per_label, err_arr=None):
    plot_style = ["o", "v", "D", "s", "p", "8", "v", "^", "1", "2"]  # "P","*"]

    print("Label: " + title)
    print("------------------")
    print("system\tthroughput\tprecision")
    print("-----")

    for idx, k in enumerate(prec_per_label):
        if err_arr == None:
            plt.plot(count_per_label[k], prec_per_label[k], label=k, marker=plot_style[idx], markersize=4)
        else:
            plt.errorbar(count_per_label[k], prec_per_label[k], err_arr[k], label=k, marker=plot_style[idx],
                         markersize=8)
        # print(k + "\t" + str(count_per_label[k]) + "\t" + str(prec_per_label[k]))
        print(k + "\t" + str(count_per_label[k][-1]) + "\t" + str(prec_per_label[k][-1]))
    plt.title(title)
    plt.xlabel('throughput')
    plt.ylabel('precision')
    plt.ylim([0, 1.1])
    plt.legend(prop={'size': 12})
    plt.savefig(filename)
    plt.close()
    print("------------------")


def printParse(results):
    for tid, trial in enumerate(results):
        sys.stdout.write("Trial " + str(tid) + "\n")
        for eid, epoch in enumerate(trial):
            sys.stdout.write("Epoch " + str(eid) + "\n")
            categories = ['PER', 'LOC', 'ORG', 'MISC']
            for lbl in categories:
                sys.stdout.write(lbl + "\t")
                for e in epoch[lbl]:
                    sys.stdout.write(e + "\t")
                sys.stdout.write("\n")

if __name__ == '__main__':

    gold_label_counts_file = sys.argv[1] # e.g. 'data/entity_label_counts_emboot.filtered.txt'
    num_systems = sys.argv[2]

    gold_counts, gold_entities, gold_labels = read_gold_mentions(gold_label_counts_file)
    majority_entities, majority_labels = majority_gold(gold_counts, gold_entities, gold_labels)

    files = []
    for i in range(1, int(num_systems) + 1):
        files.append(sys.argv[i + 2])

    results_arr_multitrials = defaultdict(list)
    for i in files:
        results_arr_multitrials[i] = parse_log_file(i)

    labels = list(set(gold_labels))
    print(labels)

    ## TODO: Add some checks and asserts here .. if [no. of trials] or [n. of epochs in each trial] in each of the output files do not Match.

    prec_scores_arr = defaultdict(list)
    count_cum_scores_arr = defaultdict(list)
    prec_scores_arr_err = defaultdict(list)
    for i in files:
        num_trials = len(results_arr_multitrials[i])
        results = results_arr_multitrials[i]
        scores = [evaluate_epochs(results[j], majority_entities, majority_labels) for j in range(num_trials)]
        cum_scores = [accumulate(scores[j]) for j in range(num_trials)]
        prec_scores = [precision(cum_scores[j]) for j in range(num_trials)]
        count_cum_scores = [count(cum_scores[j]) for j in range(num_trials)]

        ### Note:  Computing the avg precision across many trials within an experiment file (if num_trials == 1, then avg is the original array)
        tmp = np.asarray([[[e for e in prec_scores] for lbl, prec_scores in sorted(prec_scores_trial.items())] for  ## NOTE: This is to be sorted .. so that the labels align up
                          prec_scores_trial in prec_scores])
        prec_scores_arr[i] = tmp.mean(axis=0) # precision scores averaged across all the trials
        if num_trials > 1:
            prec_scores_arr_err[i] = tmp.std(axis=0)  # precision errors stdev computed across all the trials
        else:
            prec_scores_arr_err[i] = np.zeros(prec_scores_arr[i].shape)

        ### Note:  Computing the avg throughput across many trials within an experiment file (if num_trials == 1, then avg is the original array)
        tmp = np.asarray(
            [[[e for e in count_cum_scores] for lbl, count_cum_scores in sorted(count_cum_scores_trial.items())] for ## NOTE: This is to be sorted .. so that the labels align up
             count_cum_scores_trial in count_cum_scores])
        count_cum_scores_arr[i] = tmp.mean(axis=0)  # throughput averaged across all the trials

    ## NOTE: plotting the values for each of the categories (outer loop) for every output file that is to be plotted (inner loop)
    for idx, l in enumerate(sorted(labels)):
        prec_scores = defaultdict(list)
        count_cum_scores = defaultdict(list)
        prec_scores_err = defaultdict(list)

        for i in files:
            prec_scores[i] = prec_scores_arr[i][idx]
            count_cum_scores[i] = count_cum_scores_arr[i][idx]
            prec_scores_err[i] = prec_scores_arr_err[i][idx]

        plot_lines("plot-" + l + ".pdf", l, prec_scores, count_cum_scores, prec_scores_err)

    ## NOTE: Plotting the overall precision vs throughput (avg over every category label in the dataset)
    prec_scores = defaultdict(list)
    count_cum_scores = defaultdict(list)
    for i, file in enumerate(files):
        list1_prec_scores = prec_scores_arr[file]
        list2_count_cum_scores = count_cum_scores_arr[file]

        int_res = [(element[0], element[1][0] * element[1][1]) ## 2nd element is the product of precision (averaged over multiple trials) and thoughput (averaged over multiple trials) to obtain the cumulative correct counts for every category for every epoch
                   for lbl in range(len(list1_prec_scores)) ## for every category
                   for element in enumerate(zip(list1_prec_scores[lbl], list2_count_cum_scores[lbl]))] # for every epoch
        int_values = set(map(lambda x: x[0], int_res))

        prec_scores[file] = [reduce(add, [y[1] for y in int_res if y[0] == x]) for x in int_values] # add the correct counts for every category (obtained by multiplying Precision and thoughput: prev step) to obtain the cumulative total correct count (summed over all the categories) for every epoch

        count_cum_scores[file] = np.zeros((list2_count_cum_scores[0].shape))
        for lst in list2_count_cum_scores:
            count_cum_scores[file] += lst
        prec_scores[file] = prec_scores[file] / count_cum_scores[file] # normalize by cumulative counts in every epoch to obtain overall precision

    plot_lines("plot-overall.pdf", "OVERALL", prec_scores, count_cum_scores)
