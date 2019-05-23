import re
import numpy as np
from collections import defaultdict


def read_file(filename):
    with open(filename) as f:
        for line in f:
            [entity, freq, scores] = line.strip().split('\t')
            freq = int(float(freq))
            scores = {kv.split(':')[0]:float(kv.split(':')[1]) for kv in scores.split(' ')}
            yield (entity, freq, scores)


if __name__ == '__main__':
    results = dict()
    # noisy or
    for (entity, freq, scores) in read_file('sgd_interpretable_scores.txt'):
        if entity not in results:
            results[entity] = dict(PER=1.0, ORG=1.0, LOC=1.0, MISC=1.0)
        for i in range(freq):
            for label in scores:
                results[entity][label] *= 1.0 - scores[label]
    for entity in results:
        for label in results[entity]:
            results[entity][label] = 1.0 - results[entity][label]
    # select labels
    labels = []
    for entity in results:
        max_score = 0
        max_label = ''
        for label in results[entity]:
            s = results[entity][label]
            if s > max_score:
                max_score = s
                max_label = label
        labels.append((entity, max_label, max_score))
    for (entity, label, score) in sorted(labels, key=lambda x: x[2], reverse=True):
        print '%s\t%s\t%s' % (score, label, entity)
