#!/usr/bin/env python



import io
import sys
from collections import defaultdict
import numpy as np
from vocabulary import Vocabulary



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



if __name__ == '__main__':

    top_n = int(sys.argv[1])
    labels = ['PER', 'LOC', 'MISC', 'ORG']
    entities, ent_embeddings = read_embeddings('word_vectors.txt')
    patterns, pat_embeddings = read_embeddings('context_vectors.txt')
    cluster_dict = get_clusters('conll_labels.txt')
    clusters = np.array([cluster_dict[ent] for ent in entities])

    for label in labels:
        mean_ent_emb = np.mean(ent_embeddings[clusters==label], axis=0)
        pat_scores = np.dot(pat_embeddings, mean_ent_emb)
        indices = np.argsort(pat_scores)[::-1] # sort decreasingly
        top_patterns = patterns[indices[:top_n]]

        print('%s:' % label)
        for p in top_patterns:
            print('  %s' % p)
        print('---')


    entity_vocab = Vocabulary.from_file('conll_entity_vocabulary_pruned.txt')

    pools = {
        'PER': ['Clinton', 'Dole', 'Arafat', 'Yeltsin', 'Lebed', 'Dutroux', 'Wasim Akram', 'Mushtaq Ahmed', 'Waqar Younis', 'Mother Teresa'],
        'LOC': ['U.S.', 'Germany', 'Britain', 'Australia', 'France', 'England', 'Spain', 'Italy', 'China', 'Russia'],
        'ORG': ['Reuters', 'U.N.', 'PUK', 'OSCE', 'NATO', 'EU', 'Honda', 'European Union', 'Ajax', 'KDP'],
        'MISC': ['Russian', 'German', 'British', 'French', 'Dutch', 'Israeli', 'GMT', 'Iraqi', 'European', 'English'],
    }

    def print_close_entities(top_n):
        for label in pools:
            indices = [entity_vocab.get_id(e) for e in pools[label] if entity_vocab.contains(e)]
            embeddings = ent_embeddings[indices]
            mean_ent_emb = np.mean(embeddings, axis=0)
            scores = np.dot(ent_embeddings, mean_ent_emb)
            indices = np.argsort(scores)[::-1] # sort decreasingly

            print('%s:' % label)
            for i in indices[:top_n]:
                if entities[i] not in pools[label]:
                    print('  %s  --  %s' % (scores[i], entities[i]))
            print('---')

    def print_close_patterns(top_n):
        for label in pools:
            indices = [entity_vocab.get_id(e) for e in pools[label] if entity_vocab.contains(e)]
            embeddings = ent_embeddings[indices]
            mean_ent_emb = np.mean(embeddings, axis=0)
            scores = np.dot(pat_embeddings, mean_ent_emb)
            indices = np.argsort(scores)[::-1] # sort decreasingly

            print('%s:' % label)
            for i in indices[:top_n]:
                print('  %s  --  %s' % (scores[i], patterns[i]))
            print('---')


    print("=" * 50)
    print_close_entities(top_n)
    print("=" * 50)
    print_close_patterns(top_n)
