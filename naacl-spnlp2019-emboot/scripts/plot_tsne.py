#!/urs/bin/env python

import io
import re
import sys
from glob import glob
from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



colors = ['r', 'g', 'b', 'm', 'r']
marker = dict(blue='o', green='s', red='v', magenta='^')



def plot_tsne(embeddings, filename, plot_only, colors, init):
    tsne = TSNE(perplexity=30, n_components=2, init=init, n_iter=5000)
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    # labels = labels[:plot_only]
    colors = colors[:plot_only]
    plt.figure(figsize=(18, 18)) # in inches
    for i in range(1, plot_only):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=colors[i], marker=marker[colors[i]], s=320)
    # paths = []
    # for i, cat in enumerate(categories):
    #     print(cat, colors[i])
    #     embs = low_dim_embs[labels==cat]
    #     p = plt.scatter(embs[:,0], embs[:,1], marker=shape[0], color=colors[i], s=80)
    #     paths.append(p)
    # plt.legend(paths, categories, scatterpoints=1)
    plt.savefig(filename)
    plt.close()
    return low_dim_embs



def read_embeddings(filename):
    words = []
    embeddings = []
    with io.open(filename, encoding='utf8') as f:
        for line in f:
            [word, data] = line.strip().split('\t')
            emb = [float(x) for x in data.split()]
            words.append(word)
            embeddings.append(emb)
    return np.array(words), np.array(embeddings) # discard </s>



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



def get_by_max_count(ss):
    symbols, counts = np.unique(ss, return_counts=True)
    return symbols[np.argmax(counts)]



def get_clusters(file):
    clusters = {'</s>': 'NONE'}
    all_clusters = defaultdict(list)
    with io.open(file, encoding='utf8') as f:
        for line in f:
            [entity, cluster] = line.split('\t')
            all_clusters[entity].append(cluster)
    for entity in all_clusters:
        clusters[entity] = get_by_max_count(all_clusters[entity])
    return clusters



def get_colors(entities, clusters):
    colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan', 'yellow']
    cluster_color = {}
    entity_colors = []
    for e in entities:
        cluster = clusters[e]
        if cluster not in cluster_color:
            cluster_color[cluster] = len(cluster_color.keys())
        color = colors[cluster_color[clusters[e]]]
        entity_colors.append(color)
    print(cluster_color)
    return entity_colors



def sort_labels(words, entities, labels):
    sorted_labels = []
    for w in words:
        label = labels[entities==w][0] if len(labels[entities==w]) > 0 else 'NONE'
        sorted_labels.append(label)
    return np.array(labels)



if __name__ == '__main__':
    gold_counts, gold_entities, gold_labels = read_gold_mentions('conll_entity_label_counts.txt')
    majority_entities, majority_labels = majority_gold(gold_counts, gold_entities, gold_labels)
    categories = np.unique(majority_labels)

    init = 'pca'

    for filename in sorted(glob('sgd/sgd_final_word_embeddings_*-epoch-*')):
        print(filename)
        words, embeddings = read_embeddings(filename)
        colors = get_colors(words, get_clusters('conll_labels.txt'))
        # labels = sort_labels(words, majority_entities, majority_labels)
        init = plot_tsne(embeddings, filename + '.pdf', plot_only=501, colors=colors, init=init)

