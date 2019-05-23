#!/usr/bin/env python



import io
import re
import sys
from collections import defaultdict
import numpy as np
from vocabulary import Vocabulary
from scipy.stats.distributions import entropy




def get_clusters(file):
    clusters = {'</s>': 'NONE'}
    all_clusters = defaultdict(list)
    with io.open(file, encoding='utf8') as f:
        for line in f:
            [entity, cluster] = line.strip().split('\t')
            all_clusters[entity].append(cluster)
    for entity in all_clusters:
        clusters[entity] = get_by_max_count(all_clusters[entity])
    return clusters



def get_by_max_count(ss):
    symbols, counts = np.unique(ss, return_counts=True)
    return symbols[np.argmax(counts)]



def read_embeddings(file):
    symbols = []
    embeddings = []
    with io.open(file, encoding='utf8') as f:
        for line in f:
            [symbol, embedding] = line.split('\t')
            embedding = [float(elem) for elem in embedding.strip().split(' ')]
            symbols.append(symbol)
            embeddings.append(embedding)
    return np.array(symbols), np.array(embeddings)



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

def merge_epochs(epochs):
    pools = defaultdict(list)
    for epoch in epochs:
        for label in epoch:
            pools[label].extend(epoch[label])
    return pools

def norm(embeddings):
    if embeddings.ndim == 1:
        norm = np.sqrt(np.sum(np.square(embeddings)))
    elif embeddings.ndim == 2:
        norm = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
    else:
        raise ValueError('wrong number of dimensions')
    return embeddings / norm

def softmax(x):
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    return e_x / np.expand_dims(np.sum(e_x, axis=1), axis=1)



if __name__ == '__main__':

    top_n = int(sys.argv[1])
    labels = ['PER', 'LOC', 'ORG', 'MISC']
    entities, ent_embeddings = read_embeddings('sgd/sgd_final_word_embeddings_128_1.0_100_15.txt')
    patterns, pat_embeddings = read_embeddings('sgd/sgd_final_context_embeddings_128_1.0_100_15.txt')
    ent_embeddings = norm(ent_embeddings)
    pat_embeddings = norm(pat_embeddings)
    cluster_dict = get_clusters('conll_labels.txt')
    clusters = np.array([cluster_dict[ent] for ent in entities])

#     for label in labels:
#         mean_ent_emb = np.mean(ent_embeddings[clusters==label], axis=0)
#         pat_scores = np.dot(pat_embeddings, mean_ent_emb)
#         indices = np.argsort(pat_scores)[::-1] # sort decreasingly
#         top_patterns = patterns[indices[:top_n]]

#         print('%s:' % label)
#         for p in top_patterns:
#             print('  %s' % p)
#         print('---')


    entity_vocab = Vocabulary.from_file('conll_entity_vocabulary_pruned.txt')
    pattern_vocab = Vocabulary.from_file('conll_pattern_vocabulary_pruned.txt')

    pools = merge_epochs(parse_log_file('sgd/sgd_final_pools_log_128_1.0_100_15.txt'))

    def get_centroid(embeddings, pool, vocab):
        ids = [vocab.get_id(e) for e in pool if vocab.contains(e)]
        return np.mean(embeddings[ids], axis=0)

    def get_centroids(embeddings, pools, vocab):
        return np.array([norm(get_centroid(embeddings, pools[l], vocab)) for l in labels])

        

    def print_close_patterns(top_n):
        centroid_embs = get_centroids(ent_embeddings, pools, entity_vocab)
        freq = np.array(pattern_vocab.counts)
        log_freq = np.log10(freq + 1)
        scores = softmax(np.dot(pat_embeddings, centroid_embs.T)) #* log_freq
        entro = entropy(scores.T)
        predictions = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)        
        score_diff = np.max(-scores + np.expand_dims(max_scores, axis=1), axis=1)
        for i in np.argsort(-entro * log_freq):# * score_diff):
            pred = predictions[i]
            cat = labels[pred]
            print(cat, '\t', freq[i], '\t', scores[i], '\t', pattern_vocab.get_word(i))


    print(labels)
    print_close_patterns(top_n)
