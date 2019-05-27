#!/usr/bin/env python

import numpy as np
from collections import defaultdict

class Datautils:

    ## read the data from the file with the entity_ids provided by entity_vocab and context_ids provided by context_vocab
    ## data format:
    #################
    ## [label]\t[Entity Mention]\t[[context mention1]\t[context mention2],\t...[]]
    ## NOTE: The label to be removed later from the dataset and the routine suitably adjusted. Inserted here for debugging

    @classmethod
    def read_data(cls, filename, entity_vocab, context_vocab, skip_label=False):
        labels = []
        entities = []
        contexts = []
        sdx = 0 if skip_label else 1

        with open(filename) as f:
            word_counts = dict()
            for line in f:
                vals = line.strip().split('\t')
                if not skip_label:
                    labels.append(vals[0])
                if vals[1] not in word_counts:
                    word_counts[vals[sdx]] = 1
                else:
                    word_counts[vals[sdx]] += 1
                word_id = entity_vocab.get_id(vals[sdx])
                if word_id is not None:
                    entities.append(word_id)
                contexts.append([context_vocab.get_id(c) for c in vals[(sdx+1):] if context_vocab.get_id(c) is not None])

            num_count_words = 0
            for word in word_counts:
                num_count_words+=1
            print('num count words:',num_count_words)

        return np.array(entities), np.array([np.array(c) for c in contexts]), np.array(labels)

    ## Takes as input an array of entity mentions(ids) along with their contexts(ids) and converts them to individual pairs of entity and context
    ## Entity_Mention_1  -- context_mention_1, context_mention_2, ...
    ## ==>
    ## Entity_Mention_1 context_mention_1 0 ## Note the last number is the mention id, needed later to associate entity mention with all its contexts
    ## Entity_Mention_1 context_mention_2 1
    ## ....

    @classmethod
    def prepare_for_skipgram(cls, entities, contexts):

        entity_ids = []
        context_ids = []
        mention_ids = []
        for i in range(len(entities)):
            word = entities[i]
            context = contexts[i]
            for c in context:
                entity_ids.append(word)
                context_ids.append(c)
                mention_ids.append(i)
        return np.array(entity_ids), np.array(context_ids), np.array(mention_ids)

    def context2id(self, context_vocab):
        context2ID_dict = {}
        for pattern in context_vocab:
            context2ID_dict[pattern] = context_vocab.indexOf(pattern)
        # context_ids = []
        # for i in range(len(entities)):
        #     word = entities[i]
        #     context = contexts[i]
        #     for c in context:
        #         entity_ids.append(word)
        #         context_ids.append(c)
        #         mention_ids.append(i)


    ## NOTE: To understand the current negative sampling and replace it with a more simpler version. Keeping it as it is now.
    @classmethod
    def collect_negatives(cls, entities_for_sg, contexts_for_sg, entity_vocab, context_vocab):

        n_entities = entity_vocab.size()
        n_contexts = context_vocab.size()
        negatives = np.empty((n_entities, n_contexts))

        for i in range(n_entities):
            negatives[i,:] = np.arange(n_contexts)
        negatives[entities_for_sg, contexts_for_sg] = 0

        resample_neg = False
        if resample_neg:
            for sg_entity,sg_context in zip(entities_for_sg,contexts_for_sg):
                negatives[sg_entity,sg_context] = negatives[sg_entity,np.random.randint(n_contexts)]
                while negatives[sg_entity,sg_context] == 0:
                    negatives[sg_entity,sg_context] = negatives[sg_entity,np.random.randint(n_contexts)]
            for edx in range(n_entities):
                negatives[edx,0] = negatives[edx,np.random.randint(n_contexts)]
                while negatives[edx,0] == 0:
                    negatives[edx,0] = negatives[edx,np.random.randint(n_contexts)]
        return negatives

    @classmethod
    def construct_indices(cls, mentions, contexts):

        entityToPatternsIdx = defaultdict(set)
        for men, ctxs in zip(list(mentions), list([list(c) for c in contexts])):
            for ctx in ctxs:
                tmp = entityToPatternsIdx[men]
                tmp.add(ctx)
                entityToPatternsIdx[men] = tmp

        patternToEntitiesIdx = defaultdict(set)
        for men, ctxs in zip(list(mentions), list([list(c) for c in contexts])):
            for ctx in ctxs:
                tmp = patternToEntitiesIdx[ctx]
                tmp.add(men)
                patternToEntitiesIdx[ctx] = tmp

        return entityToPatternsIdx, patternToEntitiesIdx
