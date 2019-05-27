#!/usr/bin/env python

use_lmnn = True
use_mcnpl = False

import math
import time
import argparse
import json
import sys
import numpy as np
from vocabulary import Vocabulary
from datautils import Datautils

if use_lmnn:
    from w2vboot_lmnn import Word2vec
elif use_mcnpl:
    from w2vboot_mcnpl import Word2vec
else:
    from w2vboot_classifier import Word2vec
    #from w2vboot_classifier_interactive import Word2vec

from w2v import Gigaword
from tsne import plot_tsne,do_tsne
import editdistance
from sklearn import decomposition
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import euclidean_distances,linear_kernel
from sklearn import linear_model
import os
from collections import defaultdict
import io
from scipy.sparse import find

from vis.bh_tsne.bhtsne import bh_tsne
from vis.tsne_e import EasyTSNE

from feats import PMIFeats,EmbeddingFeats,GigawordFeats
from sampling import ClassBalancedMostConfident,ClassBalancedLeastConfident,ClassBalancedRandom,ClassBalancedMostLeastConfident
from sampling import ClusterSampler,ClusterDensitySampler,ClassBalancedClusterSampler,RepresentativeUncertainSampler
import pickle

from scipy.stats.distributions import entropy

#np.random.seed(1)

##################################################################
##### FEATURE NAMES AS PARAMS FOR ENTITY ENTITY CLASSIFIER
##################################################################
pmi_feat_name = 'pmi'
emboot_score_feat_name = 'emboot-score'
emboot_embedding_feat_name = 'emboot-embedding'
w2v_score_feat_name = 'w2v-score'
w2v_embedding_feat_name = 'w2v-embedding'

semantic_drift_w2v_feat_name = 'drift-w2v'
semantic_drift_emboot_feat_name = 'drift-emboot'

##################################################################
### NOTE: Hard coding the semantic drift features .. TODO: change this to a parameter
num_ents_close_to_seeds = 10
num_ents_close_to_most_recent = 10
##################################################################
class InteractiveEmboot:

    def __init__(self):
        print('initializing emboot...')
        ## TODO: create a timestamped output directory for the output files

        ### INITIALIZATION BLOCK
        #########################################################################
        #########################################################################

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ### Ontonotes data
        #########################################################################
        # parser.add_argument('--data', dest='data', default='./data/ontonotes_training_data_pruned.txt', help='data file')
        # parser.add_argument('--entity_vocab', dest='entity_vocab', default='./data/ontonotes_entity_vocabulary_pruned.txt', help='entity vocabulary file')
        # parser.add_argument('--context_vocab', dest='context_vocab', default='./data/ontonotes_pattern_vocabulary_pruned.txt', help='context vocabulary file')
        # parser.add_argument('--counts', dest='counts', default='./data/ontonotes_labels.txt', help='ontonotes labels')
        # parser.add_argument('--seeds-file', dest='seeds_file', default='./data/RandomSeedSet.ontonotes0.json', help='Seeds file formattted as Json')
        #########################################################################

        ### Filtered Ontonotes data
        #########################################################################
        parser.add_argument('--data', dest='data', default='./data/filtered_ontonotes_training.txt', help='data file')
        parser.add_argument('--entity_vocab', dest='entity_vocab', default='./data/filtered_ontonotes_entities.txt', help='entity vocabulary file')
        parser.add_argument('--context_vocab', dest='context_vocab', default='./data/filtered_ontonotes_patterns.txt', help='context vocabulary file')
        parser.add_argument('--seeds-file', dest='seeds_file', default='./data/SeedSet.Ontonotes.json', help='Seeds file formattted as Json')
        #########################################################################

        ### Conll data
        #########################################################################
        # parser.add_argument('--data', dest='data', default='./data/training_data_with_labels_emboot.filtered.txt', help='data file')
        # parser.add_argument('--entity_vocab', dest='entity_vocab', default='./data/entity_vocabulary.emboot.filtered.txt', help='entity vocabulary file')
        # parser.add_argument('--context_vocab', dest='context_vocab', default='./data/pattern_vocabulary_emboot.filtered.txt', help='context vocabulary file')
        # parser.add_argument('--counts', dest='counts', default='./data/entity_labels_emboot.filtered.txt', help='conll labels')
        # parser.add_argument('--seeds-file', dest='seeds_file', default='./data/SeedSet.conll.emnlp2017.json', help='Seeds file formattted as Json')
        #########################################################################

        ### Conll data
        #########################################################################
        # parser.add_argument('--data', dest='data', default='./data/merged_training.txt', help='data file')
        # parser.add_argument('--entity_vocab', dest='entity_vocab', default='./data/merged_entities.txt', help='entity vocabulary file')
        # parser.add_argument('--context_vocab', dest='context_vocab', default='./data/merged_patterns.txt', help='context vocabulary file')
        # parser.add_argument('--seeds-file', dest='seeds_file', default='./data/SeedSet.conll.emnlp2017.json', help='Seeds file formattted as Json')
        #########################################################################

        ### Other parameters to the algorithm
        #########################################################################
        parser.add_argument('--embedding-size', dest='embedding_size', type=int, default=100, help='embedding size')
        parser.add_argument('--neg-samples', dest='neg_samples', type=int, default=400, help='number of negative samples')
        parser.add_argument('--minibatch', dest='minibatch_size', type=int, default=512, help='size of minibatch')
        parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--burnin-epochs', dest='burnin_epochs', type=int, default=200, help='number of burnin skip-gram epochs')
        parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=1.0, help='learning rate')
        parser.add_argument('--promote-global', dest='promote_global', default=-1.0, help='Promote Entities globally; percentage specified as a number between 0 & 1; -1: using category-wise promotion')
        parser.add_argument('--usePatPool', dest='usePatPool', default=True, help='Whether to use a pool of patterns for bootstrapping')
        ### NOTE : Commenting pushpull-samples parameters .. as we currently are not sampling but taking all pairs
        # parser.add_argument('--pushpull-samples', dest='pushpull_samples', type=int, default=0, help='sample size for push/pull energies')
        # parser.add_argument('--pushpull-pat-samples', dest='pushpull_pat_samples', default=0, help='sample size for push/pull pattern energies')
        #parser.add_argument('--features', dest='features_list', default="pmi,emboot-score,emboot-embedding,w2v-score,w2v-embedding", help='features to the entity classifier')
        #parser.add_argument('--features', dest='features_list', default="pmi,emboot-score,emboot-embedding", help='features to the entity classifier')
        #parser.add_argument('--features', dest='features_list', default="pmi,w2v-score,w2v-embedding", help='features to the entity classifier')
        #parser.add_argument('--features', dest='features_list', default="pmi", help='features to the entity classifier')
        #parser.add_argument('--features', dest='features_list', default="w2v-score,w2v-embedding", help='features to the entity classifier')
        parser.add_argument('--features', dest='features_list', default="emboot-embedding", help='features to the entity classifier')
        parser.add_argument("--w2v", dest='w2vfile', default='./data/deps_filtered.words',help='dependency w2v embeddings pre-trained on ?')
        parser.add_argument("--initGiga", dest='initGigaEmbed', default=False, help='initialize emboot embeddings with gigaword embeddings [true|false]')

        parser.add_argument("--initEntityEmbeddings", dest='initEntityEmbeddings', default='', help='initialize entity embeddings')
        parser.add_argument("--initPatternEmbeddings", dest='initPatternEmbeddings', default='', help='initialize entity embeddings')

        ### Output files
        #########################################################################
        parser.add_argument('--logfile', dest='logfile', default='pools_output.txt', help='entities promoted per epoch')
        # parser.add_argument('--wordembs', dest='wordembs', default='word_embeddings.txt', help='word embeddings')
        # parser.add_argument('--ctxembs', dest='ctxembs', default='context_embeddings.txt', help='context embeddings')
        parser.add_argument('--gen-tsne',dest='genTsne',default=False,help='Generate t-SNR plot when true')
        parser.add_argument('--word_plot', dest='word_plot', default='%s_words_epoch_%s.pdf', help='words plot')
        parser.add_argument('--context_plot', dest='context_plot', default='%s_context_epoch_%s.pdf', help='context plot')
        parser.add_argument('--use-gpu',dest='useGpu',default="-1", help='which GPU to use (on clara)')

        self.args,unknown = parser.parse_known_args()

        #########################################################################
        #########################################################################

        ### Print the parameters
        #########################################################################
        print('Emboot - Promoting entities using a classifier')
        print('----------------------------------')
        print('data file: ', self.args.data)
        print('entity vocabulary file: ', self.args.entity_vocab)
        print('context vocabulary file: ', self.args.context_vocab)
        print('embedding size: ', self.args.embedding_size)
        print('negative samples sz: ', self.args.neg_samples)
        print('minibatch:', self.args.minibatch_size)
        print('epochs:', self.args.epochs)
        print('burnin epochs: ', self.args.burnin_epochs)
        print('learning rate: ', self.args.learning_rate)
        print('output file: ', self.args.logfile)
        print('seeds file: ', self.args.seeds_file)
        if self.args.promote_global == -1.0:
            print('Promoting entities per category')
        else:
            print('Promoting ', self.args.promote_global, " fraction of the entities globally")

        print('Using ALL the pairs of entities in the pool for push-pull objective')

        print('Use Pattern Pool:', self.args.usePatPool)
        print('Using ALL the pairs of entities and patterns in the pool for push-pull objective')
        print('Features to the classifier: ', self.args.features_list)
        print('Pre-initializing Emboot embeddings with gigaword embeddings: ', self.args.initGigaEmbed)
        print('Generate t-SNE plots? ', self.args.genTsne)

        #########################################################################

        # read vocabularies
        self.entity_vocab = Vocabulary.from_file(self.args.entity_vocab)
        #self.entity_vocab.prepare(6)
        self.context_vocab = Vocabulary.from_file(self.args.context_vocab)
        #self.context_vocab.prepare(6)

        # read training data
        self.mentions, self.contexts, self.labels = Datautils.read_data(self.args.data, self.entity_vocab, self.context_vocab, skip_label=False)

        self.entity_ids, self.context_ids, self.mention_ids = Datautils.prepare_for_skipgram(self.mentions, self.contexts)

        self.all_negatives = Datautils.collect_negatives(self.entity_ids, self.context_ids, self.entity_vocab, self.context_vocab)
        self.rand_ints = np.random.randint(0,self.all_negatives.shape[1],size=self.all_negatives.shape[0]*self.all_negatives.shape[1])

        self.entityToPatternsIdx, self.patternToEntitiesIdx = Datautils.construct_indices(self.mentions, self.contexts)

        # Compute the entity-pattern counts
        entity_context_cooccurrrence = np.array(list(zip(self.entity_ids, self.context_ids)))
        entity_patterns, counts = np.unique(entity_context_cooccurrrence, axis=0, return_counts=True)
        self.entity_context_cooccurrrence_counts = {(int(ep[0]),int(ep[1])):count for ep,count in zip(entity_patterns,counts)}

        self.totalEntityPatternCount = sum(self.entity_context_cooccurrrence_counts.values())
        self.totalEntityCount = sum(self.entity_vocab.counts)
        self.totalPatternCount = sum(self.context_vocab.counts)

        self.model = None

        # seed categories
        ## NOTE: Sort the categories to avoid random iterator
        with open(self.args.seeds_file) as sfh:
            self.categories = list(json.load(sfh).keys())
        self.categories.sort()
        # hack for consistency with CONNL!
        self.categories = ['GPE','AFF','ORG','PER']

        self.human_labels = []

        self.top_n = 10

        self.neg_k = self.args.neg_samples
        self.emb_size = self.args.embedding_size

        # log files
        self.pools_log = None

        # epoch info
        self.bs_epoch = 0
        self.num_epochs = self.args.epochs
        self.minibatch_size = self.args.minibatch_size

        self.burnin_epochs = self.args.burnin_epochs
        self.usePatPool = self.args.usePatPool

        self.feature_list = self.parse_feature_list(self.args.features_list)
        self.initGigaEmbed = self.args.initGigaEmbed

        self.gigaW2vEmbed = None
        self.lookupGiga = None
        ## Load gigaword embeddings only when necessary
        if w2v_embedding_feat_name in self.feature_list or w2v_score_feat_name in self.feature_list or semantic_drift_w2v_feat_name in self.feature_list or self.initGigaEmbed:
            self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_dep_embeddings(self.args.w2vfile)

        self.genTsne = self.args.genTsne

        # feats
        if pmi_feat_name in self.feature_list:
            self.pmi_feats = PMIFeats(self.entity_context_cooccurrrence_counts,self.entity_vocab,self.context_vocab)
        if emboot_score_feat_name in self.feature_list or emboot_embedding_feat_name in self.feature_list:
            self.emboot_feats = EmbeddingFeats()
        if w2v_score_feat_name in self.feature_list or w2v_embedding_feat_name in self.feature_list:
            self.w2v_feats = GigawordFeats(self.gigaW2vEmbed,self.lookupGiga,self.entity_vocab)
        self.sparse_npmi = None

        if self.args.useGpu == "-1":
            print("Using CPU ...")
        else:
            self.useGpu = self.args.useGpu
            print("Using GPU : " + self.useGpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = self.useGpu

        self.promotion_epochs = dict()

        # interaction-specific variables
        self.cached_projections = dict()
        self.cached_min_sigmas = dict()
        self.cached_max_sigmas = dict()

        self.use_neg_unigram = False
        self.smoothed_pow = 1.0
        self.entity_weights = np.ones(self.entity_vocab.size())
        #########################################################################
        #########################################################################

    def initialize_negative_sampling(self):
        self.context_counts = np.array([self.context_vocab.get_count(context_id) for context_id in np.arange(self.context_vocab.size())])
        self.smoothed_context_counts = np.power(self.context_counts,self.smoothed_pow)
        self.context_distribution = self.smoothed_context_counts / np.sum(self.smoothed_context_counts)
        self.context_range = np.arange(self.context_distribution.shape[0])

    def parse_feature_list(self, feature_string):
        feature_list = feature_string.split(",")
        return feature_list

    def softmax(self, x):
        e_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
        return e_x / np.expand_dims(np.sum(e_x, axis=1), axis=1)

    def initialize_model(self):
        print('initializing Word2Vec model ...')
        prior_entity_filename = None if self.args.initEntityEmbeddings is '' else np.load(self.args.initEntityEmbeddings)
        prior_pattern_filename = None if self.args.initPatternEmbeddings is '' else np.load(self.args.initPatternEmbeddings)
        self.bs_epoch = 0
        self.model = Word2vec(self.entity_vocab, self.context_vocab,
                              self.emb_size, self.args.neg_samples, self.args.learning_rate,
                              self.categories,
                              self.usePatPool,
                              self.initGigaEmbed, self.gigaW2vEmbed, self.lookupGiga,
                              prior_entity_embedding=prior_entity_filename, prior_pattern_embedding=prior_pattern_filename)

    def initialize_margin_model(self, margin, use_cos_sim):
        print('initializing margin-based Word2Vec model ...')
        prior_entity_filename = None if self.args.initEntityEmbeddings is '' else np.load(self.args.initEntityEmbeddings)
        prior_pattern_filename = None if self.args.initPatternEmbeddings is '' else np.load(self.args.initPatternEmbeddings)
        self.bs_epoch = 0
        self.model = Word2vec(self.entity_vocab, self.context_vocab,
                              self.emb_size, self.args.neg_samples, self.args.learning_rate,
                              self.categories,
                              self.usePatPool,
                              self.initGigaEmbed, self.gigaW2vEmbed, self.lookupGiga,
                              prior_entity_embedding=prior_entity_filename, prior_pattern_embedding=prior_pattern_filename,
                              margin=margin,cossim_margin=use_cos_sim,entity_weights=self.entity_weights)

    def initialize_mcnpl_model(self):
        print('initializing MCNPL Word2Vec model ...')
        prior_entity_filename = None if self.args.initEntityEmbeddings is '' else np.load(self.args.initEntityEmbeddings)
        prior_pattern_filename = None if self.args.initPatternEmbeddings is '' else np.load(self.args.initPatternEmbeddings)
        self.bs_epoch = 0
        self.model = Word2vec(self.entity_vocab, self.context_vocab,
                              self.emb_size, self.args.neg_samples, self.args.learning_rate,
                              self.categories,
                              self.usePatPool,
                              self.initGigaEmbed, self.gigaW2vEmbed, self.lookupGiga,
                              prior_entity_embedding=prior_entity_filename, prior_pattern_embedding=prior_pattern_filename)

    def initialize_seeds(self):
        self.entity_vocab
        print('Initializing seeds ... from file : ', self.args.seeds_file)
        with open(self.args.seeds_file) as seeds_file:
            seed_data = json.load(seeds_file)
            categories = list(seed_data.keys())
            categories.sort()
            for label in categories:
                self.model.add_seeds(label, seed_data[label])
                for word in seed_data[label]:
                    self.promotion_epochs[word] = 0

    def do_negative_context_sampling(self):
        start = time.time()
        #np.random.shuffle(self.rand_ints)
        #self.negative_context_sampling = self.rand_ints.reshape(self.all_negatives.shape)
        self.negative_context_sampling = np.random.randint(0, self.all_negatives.shape[1], size=self.all_negatives.shape)
        print('time to negative sample:',(time.time()-start),self.all_negatives.shape)

    def minibatches(self, size):
        indices = np.arange(self.entity_ids.size, dtype='int32')
        np.random.shuffle(indices)
        while indices.size > 0:
            mb_indices = indices[:size]
            indices = indices[size:]
            mb_entities = self.entity_ids[mb_indices]
            mb_contexts = self.context_ids[mb_indices]
            yield mb_entities, mb_contexts

###################################################################################################
###### PATTERN PROMOTION #########
###################################################################################################
    def compute_pattern_pmi_logfreq(self, pattern, category):
        pmi = self.compute_pattern_pmi(pattern, category)
        patternCounts = self.context_vocab.get_count(pattern)
        return pmi * math.log(patternCounts)

    def compute_pattern_pmi(self, pattern, category):
        entities = self.patternToEntitiesIdx[pattern]
        entities_in_pool = set(self.model.pools_entities[category])
        entities_in_common = entities_in_pool.intersection(entities)

        ## number of times pattern `p` matches an entity with category `cat`
        positiveEntityCounts = 0
        for e in entities_in_common:
            positiveEntityCounts += self.entity_context_cooccurrrence_counts[(e,pattern)]

        countEntityMentions = 0
        for e in self.model.pools_entities[category]:
            countEntityMentions += self.entity_vocab.get_count(e)

        patternCounts = self.context_vocab.get_count(pattern)

        pmi = math.log(float(positiveEntityCounts) / ( countEntityMentions * patternCounts ) )

        return pmi

    '''
    def promote_knn(self,k=7):
        normalized_embs = Gigaword.norm(self.model.get_entity_embeddings())
        cat_to_idx = dict([(cat,cdx) for cdx,cat in enumerate(self.categories)])
        idx_to_cat = [cat for cat in self.categories]
        pool_ids,pool_cats = [],[]
        for cat in self.categories:
            pool_ids.extend(self.model.pools_entities[cat])
            pool_cats.extend([cat_to_idx[cat]]*len(self.model.pools_entities[cat]))
        candidate_ids = list(set(self.mentions) - set(pool_ids))
        pool_cats = np.array(pool_cats,dtype=np.int32)
        candidate_ids,pool_ids = np.array(candidate_ids,dtype=np.int32), np.array(pool_ids,dtype=np.int32)

        sim_scores = linear_kernel(normalized_embs[candidate_ids,:],normalized_embs[pool_ids,:])
        sorted_score_inds = np.argsort(sim_scores,axis=-1)[:,-k:]
        cat_votes = np.array([np.bincount(pool_cats[sorted_score_inds[idx,:]]) for idx in range(sim_scores.shape[0])],dtype=np.int32)
        estimated_cats = np.argmax(cat_votes,axis=1)
        cat_scores = np.array([np.sum(sim_scores[idx,sorted_score_inds[idx,:]]) for idx in range(sim_scores.shape[0])])
    '''

    def promote_oldschool_entities(self):
        centroid_embs = np.array([Gigaword.norm(self.model.get_mean_entPool_embedding(c)) for c in self.categories])
        normalized_embs = Gigaword.norm(self.model.get_entity_embeddings())
        log_freq = np.log10(np.array(self.entity_vocab.counts) + 1)
        scores = self.softmax(np.dot(normalized_embs, centroid_embs.T)) #* log_freq
        entro = entropy(scores.T)
        predictions = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        score_diff = np.max(-scores + np.expand_dims(max_scores, axis=1), axis=1)
        to_promote = {cat:[] for cat in self.categories}
        #goodness = entro * log_freq * score_diff
        goodness = entro * log_freq
        latest_promotions = []
        for i in np.argsort(-goodness):
            pred = predictions[i]
            cat = self.categories[pred]
            score = scores[i]
            if all(i not in self.model.pools_entities[c] for c in self.categories) and len(to_promote[cat]) < self.top_n:
                self.model.pools_entities[cat].append(i)
                to_promote[cat].append(i)
                latest_promotions.append([cat,self.entity_vocab.get_word(i)])
            if all(len(to_promote[c]) >= self.top_n for c in self.categories):
                break
        #
        return latest_promotions

    def promote_patterns_by_pmi(self, promote_global, epochId):

        time_start_patpromotion = time.clock()

        self.pools_log_patterns.write('Epoch %s\n' % epochId)

        for cat in self.categories:
            ## find the patterns corresponding to all the entities in the pool, using the index
            pool_entities = set([e for e in self.model.pools_entities[cat]])
            candidate_patterns = set([p for e in pool_entities for p in list(self.entityToPatternsIdx[e])])

            ## pre-select patterns only if they co-occur with more than one entity in the pool
            preselected_patterns = list()
            for pat in candidate_patterns:
                candidates = self.patternToEntitiesIdx[pat]
                matches = candidates.intersection(pool_entities)
                if len(matches) > 1:
                    preselected_patterns.append(pat)

            ## Compute PMI based scores for each of the pre-selected patterns

            ## DONE: drop patterns which are overlapping with other candidate patterns or patterns in the current pool
            ## porting of `takeNonOverlapping()` from EPB
            preselected_patterns_non_overlapping = list()
            for candPat in preselected_patterns:
                toAdd = all([self.notContainsOrContained(candPat, pat)
                             for pat in self.model.pools_contexts[cat]]) and\
                        all([self.notContainsOrContained(candPat, pat)
                             for pat in preselected_patterns_non_overlapping])
                if toAdd:
                    preselected_patterns_non_overlapping.append(candPat)

            pmi_scores = list()
            for pat in preselected_patterns_non_overlapping:
                pmi = self.compute_pattern_pmi(pat, cat)
                # pmi = self.compute_pattern_pmi_logfreq(pat, cat) # replacing PMI with PMI*logfreq
                pmi_scores.append((pat,pmi))

            ## sort in decreasing order of scores and select the top patterns
            sorted_pmi_scores = sorted(pmi_scores, key=lambda tup: tup[1], reverse=True)

            ## Add the newly selected patterns to the pool
            to_promote = list()
            for pat, pmi in sorted_pmi_scores:
                ## DONE: drop patterns that are already present in the pool)
                if all(pat not in self.model.pools_contexts[c] for c in self.categories) and len(to_promote) < self.top_n:
                    to_promote.append(pat)

            self.model.pools_contexts[cat].extend(to_promote)
            contexts = [self.context_vocab.get_word(p) for p in to_promote]
            self.pools_log_patterns.write('\t'.join([cat] + contexts) + '\n')
            self.pools_log_patterns.flush()

        print("[pattern promotion] Time taken : " + str(time.clock() - time_start_patpromotion))

    def notContainsOrContained(self, pat1, pat2):
        pat1String = self.context_vocab.get_word(pat1)
        pat2String = self.context_vocab.get_word(pat2)

        if pat1String not in pat2String and pat2String not in pat1String:
            return True

        return False

    def get_training_ents(self, promoted_only=False, round_num=None):
        labels = []
        ents = []
        human_label_set = set([entity_id for entity_id,_ in self.human_labels])
        for cat in self.categories:
            for ent in self.model.pools_entities[cat]:
                # only doing promoted?
                if promoted_only and ent in human_label_set:
                    continue
                ent_word = self.entity_vocab.get_word(ent)
                # only at the specified round
                if round_num is not None and self.promotion_epochs[ent_word] != round_num:
                    continue
                labels.append(cat)
                ents.append(ent)
        return labels,ents

    def get_feats(self, entities, is_pool_entities, append_embeddings=False):
        all_feats = []
        all_feat_names = []
        if pmi_feat_name in self.feature_list:
            all_feats.append(self.pmi_feats.compute_pmi_feats(self.model,entities,self.categories))
            all_feat_names.append(pmi_feat_name)
        if emboot_score_feat_name in self.feature_list:
            all_feats.append(self.emboot_feats.compute_embedding_feats(self.model,entities,self.categories,is_pool_entities=is_pool_entities))
            all_feat_names.append(emboot_score_feat_name)
        if w2v_score_feat_name in self.feature_list:
            all_feats.append(self.w2v_feats.compute_embedding_feats(self.model,entities,self.categories,is_pool_entities=is_pool_entities))
            all_feat_names.append(w2v_score_feat_name)
            #all_feats.append(self.w2v_feats.w2v_entity_embeddings[entities,:])
            #all_feat_names.append(emboot_embedding_feat_name)
        if append_embeddings:
            if emboot_embedding_feat_name in self.feature_list:
                emboot_embeddings = Gigaword.norm(self.model.get_entity_embeddings())
                all_feats.append(emboot_embeddings[entities,:])
                all_feat_names.append(emboot_embedding_feat_name)
            #TODO: w2v embeddings? but what if one does not exist?
        return all_feats,all_feat_names
        #return np.concatenate(all_feats,axis=1),d_class_feats,(d_total_feats-d_class_feats)

###################################################################################################
###################################################################################################

    def print_impt_features(self, vectorizer, model, class_labels):
        """Prints features with the highest coefficient values, per class"""
        feature_names = vectorizer.get_feature_names()
        print("---------------------------------------------------------")
        print("Top 10 features : ")
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[-10:][::-1]
            print("%s:\n----\n%s" % (class_label,"\n".join(str(feature_names[j]+":"+str(model.coef_[i][j])) for j in top10)))
            print("---------------------------------------------------------")
        print("---------------------------------------------------------")

        print("Bottom 10 features : ")
        for i, class_label in enumerate(class_labels):
            bottom10 = np.argsort(model.coef_[i])[:10]
            print("%s:\n----\n%s" % (class_label,"\n".join(str(feature_names[j]+":"+str(model.coef_[i][j])) for j in bottom10)))
            print("---------------------------------------------------------")
        print("---------------------------------------------------------")

###################################################################################################
#### ENTITIY PROMOTION
###################################################################################################
    def promote_entities_by_classifier(self, promote_global, epochId, actually_promote=True):

        training_labels,training_ents = self.get_training_ents()
        training_dataset,_ = self.get_feats(training_ents,True,True)
        training_dataset = np.concatenate(training_dataset,axis=1)

        ## train a logistic regression classifier on this dataset
        classifier = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs', class_weight='balanced')
        classifier.fit(training_dataset, training_labels)

        ## run the trained classifier on candidate entities
        all_entities = set(self.mentions)
        entities_in_pool = set([e for sublist in self.model.pools_entities.values() for e in sublist])
        candidate_entities = list(all_entities - entities_in_pool)
        test_dataset,_ = self.get_feats(candidate_entities,False,True)
        test_dataset = np.concatenate(test_dataset,axis=1)

        ## promote the top candidate entities
        classes = classifier.classes_
        ## class of entities that have the highest prob scores on the test dataset
        classifier_results = classifier.predict_proba(test_dataset)
        test_predictions = classes[np.argmax(classifier_results, axis=1)]
        ## highest scores (log probs) of the predictions from the classifier
        test_predictions_maxscores = np.max(classifier_results, axis=1)


        ## [[entity_ids...],[predictions from classifier]]
        candidate_entities_test_predictions = np.array([np.array(candidate_entities),test_predictions])
        ## use the log prob scores sorted in descending order as the indices to sort the above matrix
        sorting_index = np.argsort(test_predictions_maxscores)[::-1]
        ## final sorted entityids along with their class labels
        candidate_entities_test_predictions_sorted = candidate_entities_test_predictions[:,sorting_index]

        to_promote = {cat:[] for cat in self.categories}
        to_promote_stats = {cat:[] for cat in self.categories}

        latest_promoted_entities = []
        if promote_global == -1.0:
            for idx,entid_pred in enumerate(candidate_entities_test_predictions_sorted.T):
                entid = int(entid_pred[0])
                cat = entid_pred[1]
                log_prob_score = test_predictions_maxscores[sorting_index[idx]]

                if all(entid not in self.model.pools_entities[c] for c in self.categories) and len(to_promote[cat]) < self.top_n:
                    #print('promoting',self.entity_vocab.get_word(entid),'to category',cat,log_prob_score)
                    latest_promoted_entities.append([cat,self.entity_vocab.get_word(entid)])
                    to_promote[cat].append(entid)
                    to_promote_stats[cat].append((str(entid), self.entity_vocab.get_word(entid), str(log_prob_score), str(self.entity_vocab.get_count(entid))))
                if all(len(to_promote[c]) >= self.top_n for c in self.categories):
                    break
            #
        #

        if actually_promote:
            self.pools_log.write('Epoch %s\n' % epochId)
            for cat in self.categories:
                ee = to_promote[cat]
                self.model.pools_entities[cat].extend(ee)
                for entity in ee:
                    self.promotion_epochs[self.entity_vocab.get_word(entity)] = epochId

                entities = [self.entity_vocab.get_word(e) for e in ee]
                self.pools_log.write('\t'.join([cat] + entities) + '\n')
                self.pools_log.flush()
            #
        #

        '''
        pools_stats_log_1 = open(self.args.logfile+"_stats.1.txt", 'a')
        pools_stats_log_1.write("Epoch: " + str(epochId)+"\n")
        for cat in self.categories:
            for entid, ent, logprob, freq in to_promote_stats[cat]:
                pools_stats_log_1.write(str(entid) + "\t" + ent+"\t"+cat+"\t"+logprob+"\t"+freq+ "\n")
        pools_stats_log_1.write("------------\n")
        pools_stats_log_1.close()
        '''

        return latest_promoted_entities

    def compute_sparse_npmi(self):
        print('pmi feats...')
        pmi_feats = PMIFeats(self.entity_context_cooccurrrence_counts,self.entity_vocab,self.context_vocab)
        n_entities,n_patterns = self.entity_vocab.size(),self.context_vocab.size()
        all_entities,all_patterns = np.arange(0,n_entities),np.arange(0,n_patterns)
        pmi,probs = pmi_feats.cat_pmi(all_entities,all_patterns,return_probs=True)
        print('npmi...')
        npmi = pmi / -np.log(np.maximum(1e-16,probs))
        print('find...')
        entity_inds,pattern_inds,npmi_vals = find(npmi)
        npmi_vals = (npmi_vals+1)/2
        print('min npmi:',np.min(npmi_vals),'max npmi:',np.max(npmi_vals))
        self.sparse_npmi = []
        print('sparsify...')
        for entity in all_entities:
            if entity == 0:
                self.sparse_npmi.append([])
                continue
            sparse_inds = np.argwhere(entity_inds==entity)[:,0]
            self.sparse_npmi.append((pattern_inds[sparse_inds],npmi_vals[sparse_inds]))
        print('done...')

###################################################################################################

    def get_informative_entity_list(self,n_informative=500,max_entropy=False):
        # figure out ids for training and testing
        pool_cats,training_ents = self.get_training_ents()
        candidate_entities = np.array(list(set(self.mentions) - set(training_ents)),dtype=np.int32)
        pool_entities = np.array(training_ents,dtype=np.int32)
        pool_words = [self.entity_vocab.get_word(pool_entity) for pool_entity in pool_entities]

        # extract features
        tic = time.time()
        pool_feats,pool_feat_names = self.get_feats(pool_entities,True,True)
        feat_name_idx = dict([(feat_name,idx) for idx,feat_name in enumerate(pool_feat_names)])
        pool_dataset = np.concatenate(pool_feats,axis=1)
        candidate_feats,candidate_feat_names = self.get_feats(candidate_entities,False,True)
        candidate_dataset = np.concatenate(candidate_feats,axis=1)
        toc = time.time()

        # determine most (un)informative candidate entities
        classifier = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs', class_weight='balanced')
        classifier.fit(pool_dataset, pool_cats)
        candidate_predictions = classifier.predict_proba(candidate_dataset)
        candidate_entropies = -np.sum((candidate_predictions*np.log2(candidate_predictions+1e-16)), axis=1)
        sorted_entropies = np.argsort(candidate_entropies)
        entropy_inds = sorted_entropies[-n_informative:] if max_entropy else sorted_entropies[:n_informative]
        informative_entities = candidate_entities[entropy_inds]
        informative_words = [self.entity_vocab.get_word(informative_ind) for informative_ind in informative_entities]

        return informative_entities,informative_words

    def get_informative_entities(self,n_informative=500,use_emboot_embedding=False,sampling_method='cluster',excluded_entities=None):
        n_entities,n_patterns = self.entity_vocab.size(),self.context_vocab.size()
        # figure out ids for training and testing
        all_pool_cats,training_ents = self.get_training_ents()
        if excluded_entities is None:
            candidate_entities = np.array(list(set(self.mentions) - set(training_ents)),dtype=np.int32)
        else:
            candidate_entities = np.array(list(set(self.mentions) - set(training_ents) - set(excluded_entities)),dtype=np.int32)
        all_pool_entities = np.array(training_ents,dtype=np.int32)
        all_pool_words = [self.entity_vocab.get_word(pool_entity) for pool_entity in all_pool_entities]
        all_words = [self.entity_vocab.get_word(idx) for idx in range(n_entities)]

        # extract features
        tic = time.time()
        all_pool_feats,all_pool_feat_names = self.get_feats(all_pool_entities,True,True)
        feat_name_idx = dict([(feat_name,idx) for idx,feat_name in enumerate(all_pool_feat_names)])
        all_pool_dataset = np.concatenate(all_pool_feats,axis=1)
        candidate_feats,candidate_feat_names = self.get_feats(candidate_entities,False,True)
        candidate_dataset = np.concatenate(candidate_feats,axis=1)
        toc = time.time()

        #pool_dataset = np.concatenate(pool_feats[:-1],axis=1) if emboot_embedding_feat_name in self.feature_list else np.concatenate(pool_feats,axis=1)
        #candidate_dataset = np.concatenate(candidate_feats[:-1],axis=1) if emboot_embedding_feat_name in self.feature_list else np.concatenate(candidate_feats,axis=1)

        # build classifier
        classifier = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs', class_weight='balanced')
        classifier.fit(all_pool_dataset, all_pool_cats)

        # balance most informative labeled entities
        max_labeled_ents_per_cat = 10
        pool_predictions = classifier.predict_proba(all_pool_dataset)
        pool_entropies = -np.sum((pool_predictions*np.log2(pool_predictions+1e-16)), axis=1)
        sorted_entropies = np.argsort(pool_entropies)
        informative_pool_entities = []
        informative_pool_cats = []
        pool_ents_per_cat = dict([(cat,0) for cat in self.categories])
        for ent_idx in sorted_entropies:
            pool_ent = all_pool_entities[ent_idx]
            pool_ent_cat = all_pool_cats[ent_idx]
            if pool_ents_per_cat[pool_ent_cat] == max_labeled_ents_per_cat:
                continue
            informative_pool_entities.append(pool_ent)
            informative_pool_cats.append(pool_ent_cat)
            pool_ents_per_cat[pool_ent_cat]+=1
        #
        informative_pool_feats,informative_pool_feat_names = self.get_feats(informative_pool_entities,True,True)
        informative_dataset = np.concatenate(informative_pool_feats,axis=1)
        informative_pool_words = [self.entity_vocab.get_word(ent) for ent in informative_pool_entities]

        # determine most (un)informative candidate entities
        candidate_predictions = classifier.predict_proba(candidate_dataset)
        candidate_entropies = -np.sum((candidate_predictions*np.log2(candidate_predictions+1e-16)), axis=1)
        sorted_entropies = np.argsort(candidate_entropies)

        if sampling_method == 'most_confident':
            sampler = ClassBalancedMostConfident(candidate_predictions)
        elif sampling_method == 'least_confident':
            sampler = ClassBalancedLeastConfident(candidate_predictions)
        elif sampling_method == 'random':
            sampler = ClassBalancedRandom(candidate_predictions)
        elif sampling_method == 'most_least_confident':
            sampler = ClassBalancedMostLeastConfident(candidate_predictions)
        elif sampling_method == 'cluster':
            candidate_words = [self.entity_vocab.get_word(ent) for ent in candidate_entities]
            sampler = ClusterSampler(candidate_predictions,candidate_dataset,candidate_words)
        elif sampling_method == 'density_cluster':
            candidate_words = [self.entity_vocab.get_word(ent) for ent in candidate_entities]
            sampler = ClusterDensitySampler(candidate_predictions,candidate_dataset,candidate_words)
        elif sampling_method == 'balanced_cluster':
            candidate_words = [self.entity_vocab.get_word(ent) for ent in candidate_entities]
            sampler = ClassBalancedClusterSampler(candidate_predictions,candidate_dataset,informative_dataset,informative_pool_cats,candidate_words)
        elif sampling_method == 'representative_uncertain':
            if self.sparse_npmi is None:
                self.compute_sparse_npmi()
            candidate_words = [self.entity_vocab.get_word(ent) for ent in candidate_entities]
            sampler = RepresentativeUncertainSampler(candidate_predictions,self.sparse_npmi,candidate_entities,informative_pool_entities,candidate_dataset,all_words)
        sample_inds = sampler.sample(n_informative)

        sample_inds = np.array(sample_inds,dtype=np.int32)
        informative_entities = candidate_entities[sample_inds]
        informative_words = [self.entity_vocab.get_word(informative_ind) for informative_ind in informative_entities]

        # concat individual features, combine into one
        X_feats,X_weights = [],[]
        for feat_name in all_pool_feat_names:
            pool_feat = informative_pool_feats[feat_name_idx[feat_name]]
            candidate_feat = candidate_feats[feat_name_idx[feat_name]]
            informative_feat = candidate_feat[sample_inds,:]
            concat_feat = np.concatenate((informative_feat,pool_feat),axis=0)
            X_feats.append(concat_feat)
            X_weights.append(1.0)

        # compute projection
        init_embedding = []
        for pdx,ent_ind in enumerate(informative_entities):
            if ent_ind not in self.cached_projections:
                init_embedding.append(None)
            else:
                init_embedding.append(self.cached_projections[ent_ind])
        for pdx,pool_ind in enumerate(informative_pool_entities):
            if pool_ind not in self.cached_projections:
                init_embedding.append(None)
            else:
                init_embedding.append(self.cached_projections[pool_ind])
        init_embedding = np.array(init_embedding)

        if len(self.cached_min_sigmas) == 0:
            easy_tsne = EasyTSNE(X_feats,Y_reg=init_embedding,perplexity=60,X_weights=np.array(X_weights))
        else:
            min_sigmas,max_sigmas = [0]*len(X_feats),[0]*len(X_feats)
            for feat_name in all_pool_feat_names:
                fdx = feat_name_idx[feat_name]
                min_sigmas[fdx] = 1e-1*self.cached_min_sigmas[feat_name]
                max_sigmas[fdx] = 1e1*self.cached_max_sigmas[feat_name]
            easy_tsne = EasyTSNE(X_feats,Y_reg=init_embedding,perplexity=60,X_weights=np.array(X_weights),sigma_mins=min_sigmas,sigma_maxs=max_sigmas,perplexity_iters=15)
        #easy_tsne = EasyTSNE(X_feats,Y_reg=init_embedding,perplexity=60,X_weights=np.array(X_weights))
        easy_tsne.sgd_with_momentum(n_iters=500)
        tsne_projection = easy_tsne.Y

        n_real_informative = sample_inds.shape[0]
        unlabeled_projection = tsne_projection[:n_real_informative,:]
        pool_projection = tsne_projection[n_real_informative:,:]

        for feat_name in all_pool_feat_names:
            min_feat_sigma = np.min(easy_tsne.found_sigmas[feat_name_idx[feat_name],:])
            max_feat_sigma = np.max(easy_tsne.found_sigmas[feat_name_idx[feat_name],:])
            self.cached_min_sigmas[feat_name] = min_feat_sigma
            self.cached_max_sigmas[feat_name] = max_feat_sigma

        for pdx,ent_ind in enumerate(informative_entities):
            self.cached_projections[ent_ind] = unlabeled_projection[pdx,:]
        for pdx,pool_ind in enumerate(informative_pool_entities):
            self.cached_projections[pool_ind] = pool_projection[pdx,:]

        return informative_entities,unlabeled_projection,informative_words, informative_pool_entities,pool_projection,informative_pool_words,informative_pool_cats

    def get_frequent_entities(self, n_frequent=500, use_emboot_embedding=False):
        # figure out ids for training and testing
        pool_cats,training_ents = self.get_training_ents()
        candidate_entities = np.array(list(set(self.mentions) - set(training_ents)),dtype=np.int32)
        candidate_counts = np.array([self.entity_vocab.get_count(ent) for ent in candidate_entities],dtype=np.int32)
        sorted_counts = np.argsort(candidate_counts)
        frequent_entities = candidate_entities[sorted_counts[-n_frequent:]]
        frequent_words = [self.entity_vocab.get_word(frequent_ind) for frequent_ind in frequent_entities]
        pool_entities = np.array(training_ents,dtype=np.int32)
        pool_words = [self.entity_vocab.get_word(pool_entity) for pool_entity in pool_entities]

        # extract features
        tic = time.time()
        pool_entity_pmi_feats = self.pmi_feats.compute_pmi_feats(self.model,training_ents,self.categories)
        pool_entity_embedding_score_feats = self.emboot_feats.compute_embedding_feats(self.model,training_ents,self.categories,is_pool_entities=True)

        frequent_entity_pmi_feats = self.pmi_feats.compute_pmi_feats(self.model,frequent_entities,self.categories)
        frequent_entity_embedding_score_feats = self.emboot_feats.compute_embedding_feats(self.model,frequent_entities,self.categories,is_pool_entities=False)
        toc = time.time()

        emboot_embeddings = self.model.get_entity_embeddings()
        frequent_emboot_embedding,pool_emboot_embedding = emboot_embeddings[frequent_entities,:],emboot_embeddings[pool_entities,:]

        # concat individual features, combine into one
        pmi_feats = np.concatenate((frequent_entity_pmi_feats,pool_entity_pmi_feats),axis=0)
        emboot_score_feats = np.concatenate((frequent_entity_embedding_score_feats,pool_entity_embedding_score_feats),axis=0)
        emboot_feats = np.concatenate((frequent_emboot_embedding,pool_emboot_embedding),axis=0)
        X_feats = [pmi_feats,emboot_score_feats,emboot_feats]

        # compute projection
        #init_embedding = [None]*(len(frequent_entities)+len(pool_entities))
        init_embedding = []
        for pdx,ent_ind in enumerate(frequent_entities):
            if ent_ind not in self.cached_projections:
                init_embedding.append(None)
            else:
                init_embedding.append(self.cached_projections[ent_ind])
        for pdx,pool_ind in enumerate(pool_entities):
            if pool_ind not in self.cached_projections:
                init_embedding.append(None)
            else:
                init_embedding.append(self.cached_projections[pool_ind])
        init_embedding = np.array(init_embedding)

        easy_tsne = EasyTSNE(X_feats,Y_reg=init_embedding,perplexity=20,X_weights=np.array([0.4,0.4,0.2]))
        easy_tsne.sgd_with_momentum(n_iters=500)
        tsne_projection = easy_tsne.Y

        unlabeled_projection = tsne_projection[:n_frequent,:]
        pool_projection = tsne_projection[n_frequent:,:]

        for pdx,ent_ind in enumerate(frequent_entities):
            self.cached_projections[ent_ind] = unlabeled_projection[pdx,:]
        for pdx,pool_ind in enumerate(pool_entities):
            self.cached_projections[pool_ind] = pool_projection[pdx,:]

        return frequent_entities,unlabeled_projection,frequent_words, pool_entities,pool_projection,pool_words,pool_cats

    def evaluate_latest_entities(self, top_e=-1):
        all_preds = dict()
        human_entities = [human_supervision[0] for human_supervision in self.human_labels]
        for cat in self.model.pools_entities:
            cat_list = self.model.pools_entities[cat] if top_e == -1 else self.model.pools[cat][-top_e:]
            all_preds[cat] = []
            for e_id in cat_list:
                if e_id not in human_entities:
                    all_preds[cat].append(self.entity_vocab.get_word(e_id))
                    #print('predicted',cat,'for',self.entity_vocab.get_word(e_id))
        return all_preds

    def add_to_pool(self, cat, entity_id, epochId):
        self.human_labels.append((entity_id,cat))
        self.model.pools_entities[cat].append(entity_id)
        self.promotion_epochs[self.entity_vocab.get_word(entity_id)] = epochId

    def get_category_statistics(self):
        human_label_set = set([entity_id for entity_id,_ in self.human_labels])
        max_num_epochs = max(self.promotion_epochs.values())+1
        category_stats = dict()
        for cat in self.model.pools_entities:
            cat_freqs = []
            for _ in range(max_num_epochs):
                cat_freqs.append([0,0])
            for entity_id in self.model.pools_entities[cat]:
                entity_word = self.entity_vocab.get_word(entity_id)
                epoch_promoted = self.promotion_epochs[entity_word]
                #print('entity',entity_word,'cat',cat,'epoch promoted',epoch_promoted)
                if entity_id in human_label_set:
                    cat_freqs[epoch_promoted][1]+=1
                else:
                    cat_freqs[epoch_promoted][0]+=1
            #
            category_stats[cat] = cat_freqs
        #
        return category_stats

    def write_embeddings(self, entity_embedding_filename, pattern_embedding_filename):
        self.model.write_embeddings(entity_embedding_filename, pattern_embedding_filename)

    def train_some_epochs(self,num_epochs=4):
        if not use_lmnn and not use_mcnpl:
            self.model.gather_eq_pools(self.usePatPool)
            self.model.gather_ne_pools(self.usePatPool)
        elif use_mcnpl:
            self.model.prepare_pairwise_pool()

        tic = time.time()
        for epoch in range(1, num_epochs + 1):
            print('training epoch', epoch)

            total_loss = 0.0
            mb_count = 0

            if use_lmnn:
                self.model.preprocess_triplet(self.usePatPool,head_k=400)

            for mb_entities, mb_contexts in self.minibatches(self.minibatch_size):
                if use_lmnn:
                    #self.model.sample_triplet_losses(self.usePatPool)
                    self.model.hard_negative_triplet_sampling(self.usePatPool)
                elif use_mcnpl:
                    self.model.sample_npair_loss(self.minibatch_size)

                if self.neg_k == 0:
                    mb_negatives = self.all_negatives[mb_entities]
                else:
                    if self.use_neg_unigram:
                        mb_negatives = np.random.choice(self.context_range, size=self.neg_k*mb_entities.shape[0], p=self.context_distribution)
                        mb_negatives = mb_negatives.reshape((mb_entities.shape[0],self.neg_k))
                    else:
                        mb_negatives = np.array([self.all_negatives[wdx,np.random.randint(0, self.all_negatives.shape[1], size=self.neg_k)] for wdx in mb_entities])
                loss = self.model.train(mb_entities, mb_contexts, mb_negatives, self.usePatPool)
                total_loss += loss[0]
                mb_count += 1
                avg_loss = total_loss / mb_count
                sys.stdout.write('average loss: %.6f current loss: %.6f  \r'%(avg_loss,loss[0]))
            #
        #
        toc = time.time()
        print('train some epochs time:',(toc-tic))
    #

    def training_epoch(self):

        self.bs_epoch+=1
        total_loss = 0.0
        mb_count = 0
        print('bootstrapping epoch', self.bs_epoch)

        if not use_lmnn and not use_mcnpl:
            self.model.gather_eq_pools(self.usePatPool)
            self.model.gather_ne_pools(self.usePatPool)
        elif use_mcnpl:
            self.model.prepare_pairwise_pool()

        for i in range(1, self.num_epochs + 1):
            print('epoch', i)
            if use_lmnn:
                self.model.preprocess_triplet(self.usePatPool,head_k=400)

            for mb_entities, mb_contexts in self.minibatches(self.minibatch_size):
                if use_lmnn:
                    #self.model.sample_triplet_losses(self.usePatPool)
                    self.model.hard_negative_triplet_sampling(self.usePatPool)
                elif use_mcnpl:
                    #self.model.sample_npair_loss(self.minibatch_size)
                    self.model.sample_npair_loss(self.minibatch_size//len(self.categories))

                if self.neg_k == 0:
                    mb_negatives = self.all_negatives[mb_entities]
                else:
                    if self.use_neg_unigram:
                        mb_negatives = np.random.choice(self.context_range, size=self.neg_k*mb_entities.shape[0], p=self.context_distribution)
                        mb_negatives = mb_negatives.reshape((mb_entities.shape[0],self.neg_k))
                    else:
                        mb_negatives = np.array([self.all_negatives[wdx,np.random.randint(0, self.all_negatives.shape[1], size=self.neg_k)] for wdx in mb_entities])

                # loss = self.model.train_skip_gram(mb_entities, mb_contexts, mb_negatives)
                loss = self.model.train(mb_entities, mb_contexts, mb_negatives, self.usePatPool)
                total_loss += loss[0]
                mb_count += 1
                avg_loss = total_loss / mb_count
                if use_lmnn:
                    margin_unsatisfied_perc = np.count_nonzero(loss[1]) / float(loss[1].shape[0]) if loss[1].shape[0]>0 else 0
                    sys.stdout.write('average loss: %.6f current loss: %.6f margin unsatisfied: %.6f \r'%(avg_loss,loss[0],margin_unsatisfied_perc))
                elif use_mcnpl:
                    sys.stdout.write('average loss: %.6f current loss: %.6f unsupervised objective: %.6f supervised objective: %.6f\r'%(avg_loss,loss[0],loss[1],loss[2]))
                else:
                    sys.stdout.write('average loss: %.6f current loss: %.6f \r'%(avg_loss,loss[0]))
                '''
                min_sig,max_sig = np.min(loss[4]),np.max(loss[4])
                sys.stdout.write('average loss: %.6f current loss: %.6f sg: %.6f eq: %.6f ne: %.6f sigs: %.6f %.6f\r'%(avg_loss,loss[0],loss[1],loss[2],loss[3],min_sig,max_sig))
                '''
        #
        '''
        normalized_embs = Gigaword.norm(self.model.get_entity_embeddings())
        for word in self.promotion_epochs:
            word_idx = self.entity_vocab.get_id(word)
            scores = normalized_embs.dot(normalized_embs[word_idx,:]).squeeze()
            sorted_inds = np.argsort(scores)
            print('reference entity:',word)
            for idx in range(1,10):
                print('closest entity[',idx,']:',self.entity_vocab.get_word(sorted_inds[-idx]),':',scores[sorted_inds[-idx]])
        '''
    #

    ## Run only the skip-gram style learning without promoting entities. This is to stabilize the enbeddings before relying on them to make sense predictions
    def burnin_iterations(self):
        print("Running the burnin iterations: ")
        total_loss = 0.0
        mb_count = 0

        if use_lmnn:
            self.model.sample_triplet_losses(self.usePatPool)
        elif use_mcnpl:
            self.model.prepare_pairwise_pool()
            self.model.sample_npair_loss(1)
        else:
            ### Pushpull energies from the entities in the pool
            #########################################################################
            self.model.gather_eq_pools(self.usePatPool)
            self.model.gather_ne_pools(self.usePatPool)
            #########################################################################

        for i in range(1, self.burnin_epochs + 1):
            print("\nBurnin epoch " + str(i))

            for mb_entities, mb_contexts in self.minibatches(self.minibatch_size):
                if self.neg_k == 0:
                    mb_negatives = self.all_negatives[mb_entities]
                else:
                    if self.use_neg_unigram:
                        mb_negatives = np.random.choice(self.context_range, size=self.neg_k*mb_entities.shape[0], p=self.context_distribution)
                        mb_negatives = mb_negatives.reshape((mb_entities.shape[0],self.neg_k))
                    else:
                        mb_negatives = np.array([self.all_negatives[wdx,np.random.randint(0, self.all_negatives.shape[1], size=self.neg_k)] for wdx in mb_entities])
                '''
                if np.any(mb_negatives[0,:] == 0):
                    print('zero?',mb_entities[0],mb_negatives[0,:])
                '''

                loss = self.model.burnin_train(mb_entities, mb_contexts, mb_negatives, self.usePatPool)
                total_loss += loss[0]
                mb_count += 1
                avg_loss = total_loss / mb_count
                sys.stdout.write('average loss: %.6f current loss: %.6f  \r'%(avg_loss,loss[0]))
                emboot_embeddings = self.model.get_entity_embeddings()
            #
        #
        '''
        normalized_embs = Gigaword.norm(self.model.get_entity_embeddings())
        for word in self.promotion_epochs:
            word_idx = self.entity_vocab.get_id(word)
            scores = normalized_embs.dot(normalized_embs[word_idx,:]).squeeze()
            sorted_inds = np.argsort(scores)
            print('reference entity:',word)
            for idx in range(1,10):
                print('closest entity[',idx,']:',self.entity_vocab.get_word(sorted_inds[-idx]),':',scores[sorted_inds[-idx]])
        '''
    #

if __name__ == '__main__':
    n_trials = 10
    n_rounds = 10
    use_cos_sim = True
    #all_experiments = [ (40,15,0.6,'uniform'), (400,15,0.6,'uniform'), (40,100,0.6,'uniform'), (400,100,0.6,'uniform'), (10,100,0.6,'unigram'), (10,100,0.6,'smoothed_unigram'), (40,100,0.6,'unigram'), (40,100,0.6,'smoothed_unigram'), (10,50,0.9,'unigram'), (10,100,0.9,'unigram'), (40,50,0.9,'unigram'), (40,100,0.9,'unigram')]
    all_experiments = [(10,100,0.4,'unigram')]
    #all_experiments = [ (50,100,0.5,'smoothed_unigram'), (400,100,0.5,'uniform') ]
    #all_experiments = [ (400,100,0.6,'uniform'), (400,100,0.4,'uniform'), (10,100,0.6,'unigram'), (10,100,0.4,'unigram') ]
    for experiment in all_experiments:
        k = experiment[0]
        d = experiment[1]
        margin = experiment[2]
        sampling_type = experiment[3]
        #trials_filename =  'pretrained.pkl'
        #trials_filename =  'filtered_ontonotes.pkl'
        #trials_filename =  'hard_norm_emboot_pretrained-'+str(k)+'-'+str(d)+'-'+str(margin)+'-'+sampling_type+'.pkl'
        trials_filename =  'ontonotes-'+str(k)+'-'+str(d)+'-'+str(margin)+'-'+sampling_type+'.pkl'
        all_trials = []

        for trial in range(n_trials):
            print('experiment',experiment,'trial',trial)
            emboot = InteractiveEmboot()
            emboot.usePatPool = False

            emboot.neg_k = k
            emboot.emb_size = d
            if sampling_type == 'uniform':
                emboot.use_neg_unigram = False
            elif sampling_type == 'unigram':
                emboot.use_neg_unigram = True
                emboot.smoothed_pow = 1.0
            elif sampling_type == 'smoothed_unigram':
                emboot.use_neg_unigram = True
                emboot.smoothed_pow = 0.75
            emboot.initialize_negative_sampling()

            emboot.pools_log = open(emboot.args.logfile, 'w')
            if emboot.usePatPool:
                emboot.pools_log_patterns = open(emboot.args.logfile+"_patterns.txt", 'w')

            if use_lmnn:
                emboot.initialize_margin_model(margin,True)
            elif use_mcnpl:
                emboot.initialize_mcnpl_model()
            else:
                emboot.initialize_model()

            emboot.initialize_seeds()

            emboot.burnin_iterations()

            # Bootstrapping epoch:
            #########################################################################
            trial = []
            # round 0: just the seeds
            round_preds = dict()
            for cat in emboot.model.pools_entities:
                round_preds[cat] = [emboot.entity_vocab.get_word(word_id) for word_id in emboot.model.pools_entities[cat]]
            trial.append(round_preds)
            for rdx in range(n_rounds):
                if emboot.usePatPool:
                    emboot.promote_patterns_by_pmi(emboot.args.promote_global, (rdx+1))

                emboot.training_epoch()

                ### Promote entities using a multi-class classifier
                predictions = emboot.promote_entities_by_classifier(emboot.args.promote_global, (rdx+1))
                #predictions = emboot.promote_oldschool_entities()
                for prediction in predictions:
                    print('promoting',prediction[1],'to',prediction[0],'count:',emboot.entity_vocab.get_count(emboot.entity_vocab.get_id(prediction[1])))

                round_preds = dict([(cat,[]) for cat in emboot.categories])
                for cat,entity in predictions:
                    round_preds[cat].append(entity)
                trial.append(round_preds)
            #########################################################################
            all_trials.append(trial)
            pickle.dump(all_trials, open(trials_filename, 'wb'))
        #
    #
#
