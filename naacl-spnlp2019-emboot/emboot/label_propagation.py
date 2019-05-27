#!/usr/bin/env python

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats.distributions import entropy
from sklearn.semi_supervised import label_propagation
from vocabulary import Vocabulary
from datautils import Datautils
import math
import json
from plot_results import read_gold_mentions, majority_gold

def make_dataset(words, contexts, word_vocab, context_vocab):
    n_entities = word_vocab.size()
    n_patterns = context_vocab.size()
    mat = np.zeros((n_entities, n_patterns))
    pmiMat = np.zeros((n_entities, n_patterns))
    
    for i in range(len(words)):
        row = words[i]
        for ctx in contexts[i]:
            col = ctx
            mat[row, col] += 1
    
    rowSums = np.zeros(n_entities)
    for i in range(len(words)):
        row = words[i]
        rowSum = 0
        for ctx in contexts[i]:
            col = ctx 
            rowSum += mat[row, col]      
        rowSums[row] = rowSum
    
    colSums = np.zeros(n_patterns)
    for i in range(len(words)):
        row = words[i]
        for ctx in contexts[i]:
            col = ctx
            colSums[col] += mat[row,col]
        
    totalPatternSum = 0
    for i in range(n_patterns):
        totalPatternSum += colSums[i]
        
    
    for i in range(len(words)):
        row = words[i]
        for ctx in contexts[i]:
            col = ctx
            pmiMat[row,col] = math.log( (mat[row, col] * totalPatternSum) / (rowSums[row] * colSums[col] )  )
           
    # print (mat)
    # print (pmiMat)
    return mat

def majority_labels(filename, vocab):
    counts, entities, labels = read_gold_mentions(filename)
    entities, labels = majority_gold(counts, entities, labels)
    labels = np.insert(labels, 0, 'NONE')
    indices = [0] + [vocab.get_id(e) for e in entities]
    return labels[indices]

def make_annotations(seeds, vocab, labels):
    n = vocab.size()
    y = -np.ones(n)
    for label in seeds:
        i = labels.index(label)
        for entity in seeds[label]:
            y[vocab.get_id(entity)] = i
    return y

# def label_propagation_wrapper(gamma, max_iter):
#     top_n = 10
#
#     word_vocab_fn = 'conll_entity_vocabulary_pruned.txt'
#     context_vocab_fn = 'conll_pattern_vocabulary_pruned.txt'
#     data_fn = 'conll_training_data_pruned.txt'
#     counts_fn = 'conll_entity_label_counts.txt'
#
#     labels = ['PER', 'LOC', 'ORG', 'MISC']
#
#     seeds = {
#         ##### Seeds for ONTONOTES
#
#
#         ##### Seeds for TRAIN
#         'PER': ['Clinton', 'Dole', 'Arafat', 'Yeltsin', 'Lebed', 'Dutroux',
#             'Wasim Akram', 'Mushtaq Ahmed', 'Waqar Younis', 'Mother Teresa'],
#         'LOC': ['U.S.', 'Germany', 'Britain', 'Australia', 'France', 'England',
#             'Spain', 'Italy', 'China', 'Russia'],
#         'ORG': ['Reuters', 'U.N.', 'PUK', 'OSCE', 'NATO', 'EU', 'Honda',
#             'European Union', 'Ajax', 'KDP'],
#         'MISC': ['Russian', 'German', 'British', 'French', 'Dutch', 'Israeli',
#             'GMT', 'Iraqi', 'European', 'English'],
#
#         ##### Seeds for DEV
#         # 'PER' : ["Wang",    "Lebed",    "Clinton",    "Yeltsin",    "Edberg",
#         #          "Suu Kyi",    "Dole",    "Bernardin",    "Jordan",    "Jansher"],
#         # 'MISC': ["Iraqi",    "Israeli",    "European",    "Russian",    "German",
#         #          "World Cup",    "Dutch",    "Mexican",    "English",    "WORLD CUP"],
#         # 'ORG':  ["Reuters",    "U.N.",    "Surrey",    "KDP",    "OSCE",    "Derbyshire",
#         #          "Ruch",    "Interfax",    "Pirelli",    "Somerset"],
#         # 'LOC':  ["U.S.",    "Germany",    "Russia",    "France",    "England",    "Israel",
#         #          "Iraq",    "Australia",    "Chechnya",    "China"],
#     }
#
#     #print('reading data ...')
#     word_vocab = Vocabulary.from_file(word_vocab_fn)
#     context_vocab = Vocabulary.from_file(context_vocab_fn)
#     words, contexts = read_data(data_fn, word_vocab, context_vocab)
#     #print('formatting data ...')
#     # X and y are the gold data
#     X = make_dataset(words, contexts, word_vocab, context_vocab)
#     y = majority_labels(counts_fn, word_vocab)
#     # y_train is what we would consider the pools
#     y_train = make_annotations(seeds, word_vocab, labels)
#     indices = np.arange(len(y))
#     gamma = int(gamma)
#     max_iter = int(max_iter)
#     logFile = "lp_pools_log.txt"
#
#     unique, counts = np.unique(y_train, return_counts=True)
#     #print(dict(zip(unique, counts)))
#
#     # write seeds to log file
#     pools_log = open(logFile, 'w')
#     pools_log.write('Epoch 0\n')
#     for label in range(len(labels)):
#         chunks = [labels[label]]
#         for i in indices[y_train==label]:
#             chunks.append(word_vocab.get_word(i))
#         pools_log.write('\t'.join(chunks) + '\n')
#
#     for epoch in range(1, 21):
#         #print('Epoch %s' % epoch)
#         pools_log.write('Epoch %s\n' % epoch)
#         model = label_propagation.LabelSpreading(gamma=gamma, max_iter=max_iter) # PARAMS to tune ,., see scikit dovcumentyatyion
# #         model = label_propagation.LabelSpreading(kernel = 'knn', n_neighbors=gamma, max_iter=max_iter)
#         model.fit(X, y_train)
#
#         # predicted labels
#         predictions = model.transduction_ ## Print this to see if it changes from epoch to epoch
#         # print (predictions)
#         # low entropy means we are pretty sure
#         confidences = entropy(model.label_distributions_.T)
#
#         for label in range(len(labels)):
#             # entities of the right label that are currently unannotated
#             mask = np.logical_and(predictions == label, y_train == -1)
#             ii = indices[mask]
#             cc = confidences[mask]
#             # promote top n entities
#             promoted = ii[np.argsort(cc)][:top_n]
#             y_train[promoted] = label
#
#             chunks = [labels[label]]
#             chunks.extend(word_vocab.get_word(i) for i in promoted)
#             #print(chunks)
#             pools_log.write('\t'.join(chunks) + '\n')
#
#         unique, counts = np.unique(predictions, return_counts=True)
#         #print(dict(zip(unique, counts)))
#
#
#     res = score_results(logFile)
#     print (str(gamma) + "\t" + str(max_iter) + "\t" + str(res))


if __name__ == '__main__':

    top_n = 100

    ######### CONLL CONFIG #########################
    # word_vocab_fn = 'data/entity_vocabulary.emboot.filtered.txt'
    # context_vocab_fn = 'data/pattern_vocabulary_emboot.filtered.txt'
    # data_fn = 'data/training_data_with_labels_emboot.filtered.txt'
    # counts_fn = 'data/entity_label_counts_emboot.filtered.txt'
    # seeds_file = '../data/seed.ladder.json'
    # logFile = "lp_expts/lp_pools_conll.txt"
    # scoreFile = "lp_expts/lp_pools_conll.score.txt"
    ####################################################

    ######### ONTONOTES CONFIG #########################
    word_vocab_fn = 'data-ontonotes/entity_vocabulary.emboot.filtered.txt'
    context_vocab_fn = 'data-ontonotes/pattern_vocabulary_emboot.filtered.txt'
    data_fn = 'data-ontonotes/training_data_with_labels_emboot.filtered.txt'
    counts_fn = 'data-ontonotes/entity_label_counts_emboot.filtered.txt'
    seeds_file = '../data/ladder.ontonotes.json'
    logFile = "lp_expts/lp_pools_onto.txt"
    scoreFile = "lp_expts/lp_pools_onto.score.txt"
    ####################################################

    with open(seeds_file) as seeds_file_h:
        seeds = json.load(seeds_file_h)

    with open(seeds_file) as seeds_file_h:
        labels = list(json.load(seeds_file_h).keys())
    labels.sort()
    print(labels)

    sfh = open(scoreFile, "w")
    #print('reading data ...')
    word_vocab = Vocabulary.from_file(word_vocab_fn)
    context_vocab = Vocabulary.from_file(context_vocab_fn)
    words, contexts, _ = Datautils.read_data(data_fn, word_vocab, context_vocab)
    #print('formatting data ...')
    # X and y are the gold data
    X = make_dataset(words, contexts, word_vocab, context_vocab)
    y = majority_labels(counts_fn, word_vocab)
    # y_train is what we would consider the pools
    y_train = make_annotations(seeds, word_vocab, labels)
    indices = np.arange(len(y))
    #gamma = float(sys.argv[1]) ## when 'int' it is num_neigbours 
    #max_iter = int(sys.argv[2])

    total_promoted = 0

    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    # write seeds to log file
    pools_log = open(logFile, 'w')
    pools_log.write('Epoch 0\n')
    for label in range(len(labels)):
        chunks = [labels[label]]
        for i in indices[y_train==label]:
            chunks.append(word_vocab.get_word(i))
            sfh.write(str(0) + "\t" + str(word_vocab.get_word(i)) + "\t" + str(label) + "\n")
            total_promoted += 1
        pools_log.write('\t'.join(chunks) + '\n')

    print ("Epoch 0 --> Size of chunks : " + str(sum([ len(c) for c in labels] )))
    print ("Epoch 0 --> chunks : " + str([ c for c in labels] ))
    print ("Epoch 0 --> Total promoted : " + str(total_promoted))
    # for epoch in range(1, 3):
    epoch = 1
    while total_promoted < 19984: #ontonotes;;  # 5522: #conll;;
        print('Epoch %s' % epoch)
        pools_log.write('Epoch %s\n' % epoch)
        #model = label_propagation.LabelSpreading(gamma=gamma, max_iter=max_iter) # PARAMS to tune ,., see scikit dovcumentyatyion
#         model = label_propagation.LabelSpreading(kernel = 'knn', n_neighbors=gamma, max_iter=max_iter) 
        model = label_propagation.LabelPropagation() # PARAMS to tune ,., see scikit dovcumentyatyion
        #model = label_propagation.LabelSpreading()
        model.fit(X, y_train)

        # predicted labels
        predictions = model.transduction_ ## Print this to see if it changes from epoch to epoch 
        # print (predictions)
        # low entropy means we are pretty sure
        confidences = entropy(model.label_distributions_.T)

        for label in range(len(labels)):
            # entities of the right label that are currently unannotated
            mask = np.logical_and(predictions == label, y_train == -1)
            ii = indices[mask]
            cc = confidences[mask]
            # print ("----CC-----")
            # print (cc.shape)
            # print ("---------")
            # promote top n entities
            cc_sorted = np.sort(cc)[:top_n]
            # print ("----CC_sorted-----")
            # print (cc_sorted)
            # print ("---------")
            promoted = ii[np.argsort(cc)][:top_n]
            y_train[promoted] = label

            promoted_entitystr = [word_vocab.get_word(i) for i in promoted]
            for idx, e in enumerate(promoted_entitystr):
                sfh.write(str(epoch) + "\t" + str(e) + "\t" + str(label) + "\n")

            chunks = [labels[label]]
            chunks.extend(word_vocab.get_word(i) for i in promoted)
            total_promoted += len(promoted)
            # print(chunks)
            pools_log.write('\t'.join(chunks) + '\n')

        print ("Epoch " + str(epoch) + " --> Size of chunks : " + str(sum([ len(c) for c in labels] )))
        print ("Epoch " + str(epoch) + " --> chunks : " + str([ c for c in labels] ))
        print ("Epoch " + str(epoch) + " Total promoted --> " + str(total_promoted))
        unique, counts = np.unique(predictions, return_counts=True)
        print(dict(zip(unique, counts)))
        epoch += 1

    sfh.close()
    # res = score_results(logFile)
    # print (res)
#     print str(gamma) + "\t" + str(max_iter) + "\t" + str(res)
