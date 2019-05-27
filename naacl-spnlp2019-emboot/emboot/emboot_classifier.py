#!/usr/bin/env python

import math
import time
import argparse
import json
import sys
import numpy as np
from vocabulary import Vocabulary
from datautils import Datautils
from w2vboot_classifier import Word2vec
from w2v import Gigaword
from tsne import plot_tsne
import editdistance
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import os
from collections import defaultdict
import io


np.random.seed(1)

######################################################################################################
#### t-SNE stuff ... todo: needs to be cleaned
######################################################################################################
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
    # print('----------------- Number of colors: ------------------------------ ' + str(len(cluster_color.keys())))
    return entity_colors

word_tsne = 'pca'
context_tsne = 'pca'
def plot_embeddings(model, epoch, args):
    global word_tsne
    global context_tsne
    word_embeddings = model.get_entity_embeddings()
    word_labels = model.get_entity_vocabulary().words
    colors = get_colors(word_labels, get_clusters(args.counts))
    word_tsne = plot_tsne(word_embeddings, word_labels, args.word_plot % (args.logfile, epoch), colors=colors, init=word_tsne)
    context_embeddings = model.get_context_embeddings()
    context_labels = model.get_context_vocabulary().words
    context_tsne = plot_tsne(context_embeddings, context_labels, args.context_plot % (args.logfile, epoch), init=context_tsne)

######################################################################################################

##################################################################
##### FEATURE NAMES AS PARAMS FOR ENTITY ENTITY CLASSIFIER
##################################################################
ed_feat_name = 'ed'
ed_global_feat_name = 'ed-global'
pmi_feat_name = 'pmi'
pmi_global_feat_name = 'pmi-global'
embed_emboot_feat_name = 'embed-emboot'
embed_emboot_global_feat_name = "embed-emboot-global"
embed_w2v_feat_name = 'embed-w2v'
embed_w2v_global_feat_name = 'embed-w2v-global'
semantic_drift_w2v_feat_name = 'drift-w2v'
semantic_drift_emboot_feat_name = 'drift-emboot'

##################################################################
### NOTE: Hard coding the semantic drift features .. TODO: change this to a parameter
num_ents_close_to_seeds = 10
num_ents_close_to_most_recent = 10
##################################################################
class Emboot:

    def __init__(self):
        ## TODO: create a timestamped output directory for the output files

        ### INITIALIZATION BLOCK
        #########################################################################
        #########################################################################

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ### Ontonotes data
        #########################################################################
        # parser.add_argument('--data', dest='data', default='./data/Ontonotes/training_data_with_labels_emboot_pruned.txt', help='data file')
        # parser.add_argument('--entity_vocab', dest='entity_vocab', default='./data/Ontonotes/entity_vocabulary.emboot_pruned.txt', help='entity vocabulary file')
        # parser.add_argument('--context_vocab', dest='context_vocab', default='./data/Ontonotes/pattern_vocabulary_emboot_pruned.txt', help='context vocabulary file')
        # parser.add_argument('--counts', dest='counts', default='./data/Ontonotes/entity_labels_emboot.txt', help='ontonotes labels')
        # parser.add_argument('--seeds-file', dest='seeds_file', default='./data/Ontonotes/SeedSet.ontonotes.json', help='Seeds file formattted as Json')
        #########################################################################

        ### Conll data
        #########################################################################
        parser.add_argument('--data', dest='data', default='./data/training_data_with_labels_emboot.filtered.txt', help='data file')
        parser.add_argument('--entity_vocab', dest='entity_vocab', default='./data/entity_vocabulary.emboot.filtered.txt', help='entity vocabulary file')
        #parser.add_argument('--context_vocab', dest='context_vocab', default='./data/pattern_vocabulary_emboot.filtered.txt', help='context vocabulary file')
        parser.add_argument('--context_vocab', dest='context_vocab', default='./data/pattern_vocabulary_emboot.filtered.txt', help='context vocabulary file')
        parser.add_argument('--counts', dest='counts', default='./data/entity_labels_emboot.filtered.txt', help='conll labels')
        parser.add_argument('--seeds-file', dest='seeds_file', default='./data/SeedSet.conll.emnlp2017.json', help='Seeds file formattted as Json')
        #########################################################################

        ### Other parameters to the algorithm
        #########################################################################
        parser.add_argument('--embedding-size', dest='embedding_size', type=int, default=15, help='embedding size')
        parser.add_argument('--neg-samples', dest='neg_samples', type=int, default=40, help='number of negative samples')
        parser.add_argument('--minibatch', dest='minibatch_size', type=int, default=512, help='size of minibatch')
        parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('--burnin-epochs', dest='burnin_epochs', type=int, default=200, help='number of burnin skip-gram epochs')
        parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=1.0, help='learning rate')
        parser.add_argument('--promote-global', dest='promote_global', default=-1.0, help='Promote Entities globally; percentage specified as a number between 0 & 1; -1: using category-wise promotion')
        parser.add_argument('--usePatPool', dest='usePatPool', default=True, help='Whether to use a pool of patterns for bootstrapping')
        ### NOTE : Commenting pushpull-samples parameters .. as we currently are not sampling but taking all pairs
        # parser.add_argument('--pushpull-samples', dest='pushpull_samples', type=int, default=0, help='sample size for push/pull energies')
        # parser.add_argument('--pushpull-pat-samples', dest='pushpull_pat_samples', default=0, help='sample size for push/pull pattern energies')
        parser.add_argument('--features', dest='features_list', default="pmi,ed,embed-emboot", help='features to the entity classifier')
        parser.add_argument("--w2v", dest='w2vfile', default='./data/vectors.txt',help='w2v embeddings pre-trained on the gigaword corpus')
        parser.add_argument("--initGiga", dest='initGigaEmbed', default=False, help='initialize emboot embeddings with gigaword embeddings [true|false]')

        ### Output files
        #########################################################################
        parser.add_argument('--logfile', dest='logfile', default='pools_output.txt', help='entities promoted per epoch')
        # parser.add_argument('--wordembs', dest='wordembs', default='word_embeddings.txt', help='word embeddings')
        # parser.add_argument('--ctxembs', dest='ctxembs', default='context_embeddings.txt', help='context embeddings')
        parser.add_argument('--gen-tsne',dest='genTsne',default=False,help='Generate t-SNR plot when true')
        parser.add_argument('--word_plot', dest='word_plot', default='%s_words_epoch_%s.pdf', help='words plot')
        parser.add_argument('--context_plot', dest='context_plot', default='%s_context_epoch_%s.pdf', help='context plot')
        parser.add_argument('--use-gpu',dest='useGpu',default="-1", help='which GPU to use (on clara)')

        self.args = parser.parse_args()

        #########################################################################
        #########################################################################

        self.console_log = open(self.args.logfile+"_console.log", 'w')
        sys.stdout = sys.stderr = self.console_log

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
            print('Promoting entities globally')

        print('Using ALL the pairs of entities in the pool for push-pull objective')

        print('Use Pattern Pool:', self.args.usePatPool)
        print('Using ALL the pairs of entities and patterns in the pool for push-pull objective')
        print('Features to the classifier: ', self.args.features_list)
        print('Pre-initializing Emboot embeddings with gigaword embeddings: ', self.args.initGigaEmbed)
        print('Generate t-SNE plots? ', self.args.genTsne)

        #########################################################################

        # read vocabularies
        print('reading vocabularies ...')
        self.entity_vocab = Vocabulary.from_file(self.args.entity_vocab)
        self.context_vocab = Vocabulary.from_file(self.args.context_vocab)

        # read training data
        print('reading training data ...')
        self.mentions, self.contexts, self.labels = Datautils.read_data(self.args.data, self.entity_vocab, self.context_vocab)

        print('preparing data for skip-gram ...')
        self.entity_ids, self.context_ids, self.mention_ids = Datautils.prepare_for_skipgram(self.mentions, self.contexts)

        print('starting negative sampling ... ')
        self.all_negatives = Datautils.collect_negatives(self.entity_ids, self.context_ids, self.entity_vocab, self.context_vocab)

        print("Building the forward and reverse indices")
        self.entityToPatternsIdx, self.patternToEntitiesIdx = Datautils.construct_indices(self.mentions, self.contexts)
        print("Done printing the indices")

        # Compute the entity-pattern counts
        entity_context_cooccurrrence = np.array(list(zip(self.entity_ids, self.context_ids)))
        entity_patterns, counts = np.unique(entity_context_cooccurrrence, axis=0, return_counts=True)
        self.entity_context_cooccurrrence_counts = {(int(ep[0]),int(ep[1])):count for ep,count in zip(entity_patterns,counts)}

        self.totalEntityPatternCount = sum(self.entity_context_cooccurrrence_counts.values())
        self.totalEntityCount = sum(self.entity_vocab.counts)
        self.totalPatternCount = sum(self.context_vocab.counts)

        print('NOTE: Model to be initialized before calling bootstrap training by invoking initialize_model() ')
        self.model = None

        # seed categories
        ## NOTE: Sort the categories to avoid random iterator
        with open(self.args.seeds_file) as sfh:
            self.categories = list(json.load(sfh).keys())
        self.categories.sort()
        print(self.categories)

        self.top_n = 10
        self.neg_k = self.args.neg_samples

        # log files
        self.pools_log = None

        print('NOTE: Seed to be initialized after Model is initialized [ initialize_model() ]')

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
        if embed_w2v_feat_name in self.feature_list or embed_w2v_global_feat_name in self.feature_list or semantic_drift_w2v_feat_name in self.feature_list or self.initGigaEmbed:
            self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_embeddings(self.args.w2vfile)

        self.genTsne = self.args.genTsne

        if self.args.useGpu == "-1":
            print("Using CPU ...")
        else:
            self.useGpu = self.args.useGpu
            print("Using GPU : " + self.useGpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = self.useGpu

        #########################################################################
        #########################################################################

    def parse_feature_list(self, feature_string):
        feature_list = feature_string.split(",")
        return feature_list

    def softmax(self, x):
        e_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
        return e_x / np.expand_dims(np.sum(e_x, axis=1), axis=1)

    def initialize_model(self):
        print('initializing Word2Vec model ...')
        self.model = Word2vec(self.entity_vocab, self.context_vocab,
                              self.args.embedding_size, self.args.neg_samples, self.args.learning_rate,
                              self.categories,
                              self.usePatPool,
                              self.initGigaEmbed, self.gigaW2vEmbed, self.lookupGiga)

    def initialize_seeds(self):
        self.entity_vocab
        print('Initializing seeds ... from file : ', self.args.seeds_file)
        with open(self.args.seeds_file) as seeds_file:
            seed_data = json.load(seeds_file)
            categories = list(seed_data.keys())
            categories.sort()
            for label in categories:
                self.model.add_seeds(label, seed_data[label])

    def do_negative_context_sampling(self):
        self.negative_context_sampling = np.random.randint(0, self.all_negatives.shape[1], size=self.all_negatives.shape)

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


    def promote_patterns_by_pmi(self, promote_global, epochId):

        time_start_patpromotion = time.clock()

        to_promote_stats = {cat:[] for cat in self.categories}

        self.pools_log_patterns.write('Epoch %s\n' % epochId)

        if promote_global != -1.0: ## Not category-wise promotion
            pmi_scores_all = dict()

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
            if promote_global == -1.0:
                to_promote = list()
                for pat, pmi in sorted_pmi_scores:
                    ## DONE: drop patterns that are already present in the pool)
                    if all(pat not in self.model.pools_contexts[c] for c in self.categories) and len(to_promote) < self.top_n:
                        to_promote.append(pat)
                        to_promote_stats[cat].append((str(pat), self.context_vocab.get_word(pat), str(pmi), str(self.context_vocab.get_count(pat))))

                self.model.pools_contexts[cat].extend(to_promote)
                contexts = [self.context_vocab.get_word(p) for p in to_promote]
                self.pools_log_patterns.write('\t'.join([cat] + contexts) + '\n')
                self.pools_log_patterns.flush()

            else:
                pmi_scores_all[cat] = sorted_pmi_scores

        if promote_global != -1.0: ## Not category-wise promotion
            pmi_scores_global = list()
            for cat, pat_pmi_scores in pmi_scores_all.items():
                for pat, pmi in pat_pmi_scores:
                    pmi_scores_global.append((pat, pmi, cat))

            ## sort in decreasing order of scores and select the top patterns
            ## top = self.top_n*len(self.categories)
            sorted_pmi_scores_global = sorted(pmi_scores_global, key=lambda tup: tup[1], reverse=True)

            num_pats_promoted = 0
            to_promote = {cat:[] for cat in self.categories}
            for (pat, pmi, cat) in sorted_pmi_scores_global:
                if all(pat not in self.model.pools_contexts[c] for c in self.categories):
                    to_promote[cat].append(pat)
                    to_promote_stats[cat].append((str(pat), self.context_vocab.get_word(pat), str(pmi), str(self.context_vocab.get_count(pat))))
                    num_pats_promoted += 1
                if num_pats_promoted >= (self.top_n*len(self.categories)):
                    break

            for cat in self.categories:
                self.model.pools_contexts[cat].extend(to_promote[cat])
                contexts = [self.context_vocab.get_word(p) for p in to_promote[cat]]
                self.pools_log_patterns.write('\t'.join([cat] + contexts) + '\n')
                self.pools_log_patterns.flush()



        pools_stats_log_1 = open(emboot.args.logfile+"_stats.1.txt", 'a')
        pools_stats_log_1.write("Epoch: " + str(epochId)+"\n")
        for cat in self.categories:
            for entid, ent, pmi, freq in to_promote_stats[cat]:
                pools_stats_log_1.write(str(entid) + "\t" + ent+"\t"+cat+"\t"+pmi+"\t"+freq+ "\n")
        pools_stats_log_1.write("------------\n")
        pools_stats_log_1.close()

        print("[pattern promotion] Time taken : " + str(time.clock() - time_start_patpromotion))

    def notContainsOrContained(self, pat1, pat2):
        pat1String = self.context_vocab.get_word(pat1)
        pat2String = self.context_vocab.get_word(pat2)

        if pat1String not in pat2String and pat2String not in pat1String:
            return True

        return False

###################################################################################################


###################################################################################################
#### FEATURE GENERATION ############
###################################################################################################

    def compute_edit_distance_feature(self, entityId, category, features):

        # positive edit-dist score
        entity = self.entity_vocab.get_word(entityId)
        posScore = max(
        [ 1
          if editdistance.eval(self.entity_vocab.get_word(e), entity)/float(len(self.entity_vocab.get_word(e))) < 0.2 ## [edit ditance] / |posEntStr|
          else
          0
         for e in self.model.pools_entities[category]
          if entityId != e ])


        if ed_global_feat_name not in self.feature_list: ## Add only if the global feature is not active
            features["EDPos_"+category] = posScore

        # negative edit-dist score
        ## Set of entities that do not belong to a category
        neg_entities = set([e ## Set of all entities in the pool
                            for sublist in self.model.pools_entities.values()
                            for e in sublist]) \
                       - \
                       set(self.model.pools_entities[category]) ## Set of entities that belong to `category`

        negScore = max(
            [ 1
              if editdistance.eval(self.entity_vocab.get_word(e), entity)/float(len(self.entity_vocab.get_word(e))) < 0.2 ## [edit ditance] / |negEntStr|
              else
              0
              for e in neg_entities
              if entityId != e ])

        if ed_global_feat_name not in self.feature_list: ## Add only if the global feature is not active
            features["EDNeg_"+category] = negScore

        return posScore , negScore
        # print("edit distance features for ", entityId)
        # print(features)

    def compute_pattern_pmi_features(self, entityID, category, features):

        patternIDs = self.model.pools_contexts[category]

        pmi_scores = list()

        for patid in patternIDs:
            if (entityID,patid) in self.entity_context_cooccurrrence_counts:
                ep_cnt = self.entity_context_cooccurrrence_counts[(entityID,patid)]
            else:
                ep_cnt = 0
            e_cnt = self.entity_vocab.get_count(entityID)
            p_cnt = self.context_vocab.get_count(patid)

            p_ep = ep_cnt / float(self.totalEntityPatternCount)
            p_e = e_cnt / float(self.totalEntityCount)
            p_p = p_cnt / float(self.totalPatternCount)

            if p_ep == 0:
                pmi = 0
            else:
                pmi = math.log(p_ep / (p_e * p_p))

            pattern = self.context_vocab.get_word(patid)
            if pmi_global_feat_name not in self.feature_list: ## Add only if the global feature is not active
                features[pattern+"_"+category+"_pmi"] = pmi
            pmi_scores.append(pmi)

        if len(pmi_scores) > 0:
            max_pmi_score = max(pmi_scores)
        else: ## case of emppty pattern pool for any category .. assign the least value so that this global feature is not added
            max_pmi_score = float('-inf')

        if len(pmi_scores) > 0:
            avg_pmi_score = sum(pmi_scores)/len(pmi_scores)
        else: ## case of emppty pattern pool for any category .. assign the least value so that this global feature is not added
            avg_pmi_score = float('-inf')

        return max_pmi_score, avg_pmi_score
        # print ("Compute pattern pmi features for ", entityID)
        # print(features)

    def compute_gigaword_embedding_features(self, entityId, category, features):
        ## Compute the features using w2v embeddings
        promoted_entities = [e for e in self.model.pools_entities[category] if entityId != e] # remove `entityId` if it is already present in the pool

        given_entity_string = self.entity_vocab.get_word(entityId)
        promoted_entities_string = [self.entity_vocab.get_word(e) for e in promoted_entities if entityId != e]

        gigaSimScores = [self.sanitiseAndFindAvgSimilarity_gigaword(given_entity_string, ent) ## find the similarity of entities in the pool with the candidate entity
                         for ent in promoted_entities_string]

        avgW2vScore = sum(gigaSimScores) / len(gigaSimScores)
        maxW2vScore = max(gigaSimScores)
        minW2vScore = min(gigaSimScores)

        if embed_w2v_global_feat_name not in self.feature_list: ## Add only if the global feature is not active
            features["avg_"+category+"_w2v"] = avgW2vScore
            features["max_"+category+"_w2v"] = maxW2vScore
            features["min_"+category+"_w2v"] = minW2vScore

        return maxW2vScore, avgW2vScore

    def compute_emboot_embedding_features(self, entityId, category, features):

        ## Compute the features using EmBoot embeddings
        all_embeddings = self.model.get_entity_embeddings()

        given_entity_embedding = Gigaword.norm(all_embeddings[entityId]) # EmBoot embedding of `entityId`

        promoted_entities = [e for e in self.model.pools_entities[category] if entityId != e] # remove `entityId` if it is already present in the pool
        promoted_entities_embeddings = Gigaword.norm(all_embeddings[promoted_entities]) # embeddings of entities in the pool

        dot_product = np.dot(promoted_entities_embeddings, given_entity_embedding) # compute the dot product of the `entityId` with every entity in the pool

        avgEmbootScore = sum(dot_product)/len(dot_product)
        maxEmbootScore = max(dot_product)
        minEmbootScore = min(dot_product)

        if embed_emboot_global_feat_name not in self.feature_list: ## Add only if the global feature is not active
            features["avg_"+category+"_emboot"] = avgEmbootScore
            features["min_"+category+"_emboot"] = minEmbootScore
            features["max_"+category+"_emboot"] = maxEmbootScore

        return maxEmbootScore, avgEmbootScore

    def compute_semantic_drift_features(self, entityId, category, features):
        # semantic drift using emboot vectors
        ## NOTE: This feature is active only when `num_ents_in_pool` >= |m|+|n|  (m,n - m closest ents to seeds, n most recent ents)
        if semantic_drift_emboot_feat_name in self.feature_list and len(self.model.pools_entities[category]) >= (num_ents_close_to_seeds + num_ents_close_to_most_recent) :
            all_embeddings = self.model.get_entity_embeddings()

            given_entity_embedding = Gigaword.norm(all_embeddings[entityId]) # EmBoot embedding of `entityId`

            promoted_entities_with_epoch = [(e,epoch) for e,epoch in self.model.pools_entities_with_epoch[category] if entityId != e] # remove `entityId` if it is already present in the pool
            promoted_entities_with_epoch_sorted = [e for e,epoch in sorted(promoted_entities_with_epoch, key=lambda x:x[1])] # sort the elements according to epoch id

            entities_close_to_seeds = promoted_entities_with_epoch_sorted[:num_ents_close_to_seeds]
            entities_most_recent = promoted_entities_with_epoch_sorted[-num_ents_close_to_most_recent:]

            entities_close_to_seeds_embeddings = Gigaword.norm(all_embeddings[entities_close_to_seeds])
            entities_most_recent_embeddings = Gigaword.norm(all_embeddings[entities_most_recent])

            dot_product_close_to_seeds = np.dot(entities_close_to_seeds_embeddings, given_entity_embedding)
            avg_dot_product_close_to_seeds = sum(dot_product_close_to_seeds)/len(dot_product_close_to_seeds)
            max_dot_product_close_to_seeds = max(dot_product_close_to_seeds)

            dot_product_most_recent = np.dot(entities_most_recent_embeddings, given_entity_embedding)
            avg_dot_product_most_recent = sum(dot_product_most_recent)/len(dot_product_most_recent)
            max_dot_product_most_recent = max(dot_product_most_recent)

            score_drift_avg = avg_dot_product_close_to_seeds / avg_dot_product_most_recent
            score_drift_max = max_dot_product_close_to_seeds / max_dot_product_most_recent

            features["avg_"+category+"_drift_emboot"] = score_drift_avg
            features["max_"+category+"_drift_emboot"] = score_drift_max
            # print("--------DRIFT FEATURE------------- Adding feature : " + "[avg_"+category+"_drift_emboot]= " + str(score_drift_avg) + "; and " + "[max_"+category+"_drift_emboot]= " + str(score_drift_max))

###################################################################################################

    #############################################################
    #### Ported from `org.clulab.embeddings.word2vec.Word2Vec`
    #############################################################
    def sanitiseAndFindAvgSimilarity_gigaword(self, entity1, entity2):

        sanitised_entity1 = [Gigaword.sanitiseWord(tok) for tok in entity1.split(" ")]
        sanitised_entity2 = [Gigaword.sanitiseWord(tok) for tok in entity2.split(" ")]

        giga_index1 = [self.lookupGiga[tok] if tok in self.lookupGiga else self.lookupGiga["<unk>"] for tok in sanitised_entity1]
        giga_index2 = [self.lookupGiga[tok] if tok in self.lookupGiga else self.lookupGiga["<unk>"] for tok in sanitised_entity2]

        embed1 = self.gigaW2vEmbed[giga_index1]
        embed2 = self.gigaW2vEmbed[giga_index2]

        ## note: can we do this faster using numpy ?
        total_sim = 0
        count = 0
        for i in embed1:
            for j in embed2:
                total_sim += np.dot(Gigaword.norm(i), Gigaword.norm(j))
                count += 1

        if count == 0:
            return 0
        else:
            return total_sim / count

###################################################################################################
#### CREATE DATASET ##########
###################################################################################################

    def create_datum_features(self, ent):

        ent_features = dict()

        posEdscore_array = list()
        negEdscore_array = list()
        max_pmi_score_array = list()
        avg_pmi_score_array = list()
        max_emboot_score_array = list()
        avg_emboot_score_array = list()
        max_w2v_score_array = list()
        avg_w2v_score_array = list()

        for cat in self.categories:
            if ed_feat_name in self.feature_list or ed_global_feat_name in self.feature_list:
                ## compute the edit distance features
                posScore, negScore = self.compute_edit_distance_feature(ent, cat, ent_features)
                posEdscore_array.append(posScore)
                negEdscore_array.append(negScore)

            if pmi_feat_name in self.feature_list or pmi_global_feat_name in self.feature_list:
                ## compute the pattern PMI features
                max_pmi_score, avg_pmi_score = self.compute_pattern_pmi_features(ent, cat, ent_features)

                max_pmi_score_array.append(max_pmi_score)
                avg_pmi_score_array.append(avg_pmi_score)

            if embed_emboot_feat_name in self.feature_list or embed_emboot_global_feat_name in self.feature_list:
                ## compute the emboot embedding based features
                max_emboot_score, avg_emboot_score = self.compute_emboot_embedding_features(ent, cat, ent_features)

                max_emboot_score_array.append(max_emboot_score)
                avg_emboot_score_array.append(avg_emboot_score)

            if embed_w2v_feat_name in self.feature_list or embed_w2v_global_feat_name in self.feature_list:
                ## compute the gigaword embedding based features
                max_w2v_score, avg_w2v_score = self.compute_gigaword_embedding_features(ent, cat, ent_features)

                max_w2v_score_array.append(max_w2v_score)
                avg_w2v_score_array.append(avg_w2v_score)

            if semantic_drift_emboot_feat_name in self.feature_list or semantic_drift_w2v_feat_name in self.feature_list:
                ## compute the semantic drift features
                self.compute_semantic_drift_features(ent, cat, ent_features)

        ## Add the overall features
        if ed_feat_name in self.feature_list or ed_global_feat_name in self.feature_list:
            lbl = self.categories[np.argmax(posEdscore_array)]
            score = np.max(posEdscore_array)
            ent_features["EDPosMax_"+lbl] = score

            lbl = self.categories[np.argmax(negEdscore_array)]
            score = np.max(negEdscore_array)
            ent_features["EDNegMax_"+lbl] = score


        if pmi_feat_name in self.feature_list or pmi_global_feat_name in self.feature_list:
            lbl = self.categories[np.argmax(max_pmi_score_array)]
            score = np.max(max_pmi_score_array)
            ent_features["PMI_Max_"+lbl] = score

            lbl = self.categories[np.argmax(avg_pmi_score_array)]
            score = np.max(avg_pmi_score_array)
            ent_features["PMI_Avg_"+lbl] = score

        if embed_emboot_feat_name in self.feature_list or embed_emboot_global_feat_name in self.feature_list:
            lbl = self.categories[np.argmax(max_emboot_score_array)]
            score = np.max(max_emboot_score_array)
            ent_features["EMBOOT_Max_"+lbl] = score

            lbl = self.categories[np.argmax(avg_emboot_score_array)]
            score = np.max(avg_emboot_score_array)
            ent_features["EMBOOT_Avg_"+lbl] = score

        if embed_w2v_feat_name in self.feature_list or embed_w2v_global_feat_name in self.feature_list:
            lbl = self.categories[np.argmax(max_w2v_score_array)]
            score = np.max(max_w2v_score_array)
            ent_features["W2V_Max_"+lbl] = score

            lbl = self.categories[np.argmax(avg_w2v_score_array)]
            score = np.max(avg_w2v_score_array)
            ent_features["W2V_Avg_"+lbl] = score

        for key, value in ent_features.items():
            if math.isnan(value) :
                print ("ERROR: NAN value in KEY ", key)

        return ent_features

    def create_training_dataset(self):
        labels = []
        features_dict = []
        for cat in self.categories:
            for ent in self.model.pools_entities[cat]:
                ent_features = self.create_datum_features(ent) ## ent_features = feature_dict
                features_dict.append(ent_features)
                labels.append(cat)

        features_dict_vec = DictVectorizer()
        return features_dict_vec.fit_transform(features_dict).toarray(), labels, features_dict_vec

    def create_test_dataset(self, candidate_entities, features_dict_vec):
        features_dict = []
        for ent in candidate_entities:
            ent_features = self.create_datum_features(ent)
            features_dict.append(ent_features)

        return features_dict_vec.transform(features_dict).toarray()

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
    def promote_entities_by_classifier(self, promote_global, epochId):

        print("[Classifer] Creating the train dataset")
        ## build a dataset given the current pools of entities and patterns
        training_dataset, training_labels, feature_vec = self.create_training_dataset()

        print("[Classifer] Training logistic regression classifier")
        ## train a logistic regression classifier on this dataset
        classifier = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
        classifier.fit(training_dataset, training_labels)

        ## run the trained classifier on candidate entities
        all_entities = set(self.mentions)
        entities_in_pool = set([e for sublist in self.model.pools_entities.values() for e in sublist])
        candidate_entities = list(all_entities - entities_in_pool)
        test_dataset = self.create_test_dataset(candidate_entities, feature_vec)

        ## promote the top candidate entities
        classes = classifier.classes_
        ## class of entities that have the highest prob scores on the test dataset
        classifier_results = classifier.predict_proba(test_dataset)
        test_predictions = classes[np.argmax(classifier_results, axis=1)]
        ## highest scores (log probs) of the predictions from the classifier
        test_predictions_maxscores = np.max(classifier_results, axis=1)

        print("[Classifer] Finished generating predictions\n------")

        ### Debug info for the classifier
        #########################################################################
        ### https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
        print("[Classifier")
        self.print_impt_features(feature_vec, classifier, classes)
        #########################################################################

        ## [[entity_ids...],[predictions from classifier]]
        candidate_entities_test_predictions = np.array([np.array(candidate_entities),test_predictions])
        ## use the log prob scores sorted in descending order as the indices to sort the above matrix
        sorting_index = np.argsort(test_predictions_maxscores)[::-1]
        ## final sorted entityids along with their class labels
        candidate_entities_test_predictions_sorted = candidate_entities_test_predictions[:,sorting_index]

        to_promote = {cat:[] for cat in self.categories}
        to_promote_stats = {cat:[] for cat in self.categories}

        if promote_global == -1.0:
            for idx,entid_pred in enumerate(candidate_entities_test_predictions_sorted.T):
                entid = int(entid_pred[0])
                cat = entid_pred[1]
                log_prob_score = test_predictions_maxscores[sorting_index[idx]]

                if all(entid not in self.model.pools_entities[c] for c in self.categories) and len(to_promote[cat]) < self.top_n:
                    to_promote[cat].append(entid)
                    to_promote_stats[cat].append((str(entid), self.entity_vocab.get_word(entid), str(log_prob_score), str(self.entity_vocab.get_count(entid))))
                if all(len(to_promote[c]) >= self.top_n for c in self.categories):
                    break

        else:
            ## Select the `top_n` * |num_categories|  (10 x 4 -- conll, 10 x 11 -- ontonotes) from the sorted predictions
            #  [self.top_n*len(self.categories)]
            num_ents_promoted = 0

            for idx, entid_pred in enumerate(candidate_entities_test_predictions_sorted.T):
                entid = int(entid_pred[0])
                cat = entid_pred[1]
                log_prob_score = test_predictions_maxscores[sorting_index[idx]]

                if all(entid not in self.model.pools_entities[c] for c in self.categories):
                    to_promote[cat].append(entid)
                    to_promote_stats[cat].append((str(entid), self.entity_vocab.get_word(entid), str(log_prob_score), str(self.entity_vocab.get_count(entid))))
                    num_ents_promoted += 1
                if num_ents_promoted >= (self.top_n*len(self.categories)):
                    break


        self.pools_log.write('Epoch %s\n' % epochId)
        for cat in self.categories:
            ee = to_promote[cat]
            self.model.pools_entities[cat].extend(ee)

            ee_with_epoch = [ (e,epochId) for e in ee] ## Note add the epoch id to the promoted entities -- for semantic drift features
            self.model.pools_entities_with_epoch[cat].extend(ee_with_epoch)

            entities = [self.entity_vocab.get_word(e) for e in ee]
            self.pools_log.write('\t'.join([cat] + entities) + '\n')
            self.pools_log.flush()

        pools_stats_log_1 = open(emboot.args.logfile+"_stats.1.txt", 'a')
        pools_stats_log_1.write("Epoch: " + str(epochId)+"\n")
        for cat in self.categories:
            for entid, ent, logprob, freq in to_promote_stats[cat]:
                pools_stats_log_1.write(str(entid) + "\t" + ent+"\t"+cat+"\t"+logprob+"\t"+freq+ "\n")
        pools_stats_log_1.write("------------\n")
        pools_stats_log_1.close()

###################################################################################################

    def training_epoch(self):

        ### Do negative sampling: TODO: Simplify this
        #########################################################################
        self.do_negative_context_sampling()
        n_negatives_sampled = 0
        #########################################################################

        self.bs_epoch+=1
        total_loss = 0.0
        mb_count = 0
        print('bootstrapping epoch', self.bs_epoch)

        ### Pushpull energies from the entities in the pool
        #########################################################################
        self.model.gather_eq_pools(self.usePatPool)
        self.model.gather_ne_pools(self.usePatPool)
        #########################################################################

        for i in range(1, self.num_epochs + 1):
            print('epoch', i)

            for mb_entities, mb_contexts in self.minibatches(self.minibatch_size):
                if self.neg_k > 0 and n_negatives_sampled > self.negative_context_sampling.shape[1]:
                    self.do_negative_context_sampling()
                    n_negatives_sampled = 0

                if self.neg_k == 0:
                    mb_negatives = self.all_negatives[mb_entities]
                else:
                    mb_negatives = np.array([self.all_negatives[wdx,self.negative_context_sampling[wdx,n_negatives_sampled:n_negatives_sampled+self.neg_k]] for wdx in mb_entities])

                # loss = self.model.train_skip_gram(mb_entities, mb_contexts, mb_negatives)
                loss = self.model.train(mb_entities, mb_contexts, mb_negatives, self.usePatPool)
                total_loss += loss[0]
                mb_count += 1
                avg_loss = total_loss / mb_count
                sys.stdout.write('average loss: %.6f current loss: %.6f  \r'%(avg_loss,loss[0]))

                n_negatives_sampled += self.neg_k

            # print("\n----------\nCompleted the skip-gram only epoch for every minimatch.\nRunning the push-pull training step\n------------")
            print("\n----------\nTraining without retrofitting")
            self.console_log.flush()

            print('\n')

    ## Run only the skip-gram style learning without promoting entities. This is to stabilize the enbeddings before relying on them to make sense predictions
    def burnin_iterations(self):
        print("Running the burnin iterations: ")
        self.do_negative_context_sampling()
        n_negatives_sampled = 0
        total_loss = 0.0
        mb_count = 0

        ### Pushpull energies from the entities in the pool
        #########################################################################
        print('Before gathering pools')
        self.model.gather_eq_pools(self.usePatPool)
        self.model.gather_ne_pools(self.usePatPool)
        #########################################################################

        for i in range(1, self.burnin_epochs + 1):
            print("\nBurnin epoch " + str(i))

            for mb_entities, mb_contexts in self.minibatches(self.minibatch_size):
                if self.neg_k > 0 and n_negatives_sampled > self.negative_context_sampling.shape[1]:
                    self.do_negative_context_sampling()
                    n_negatives_sampled = 0

                if self.neg_k == 0:
                    mb_negatives = self.all_negatives[mb_entities]
                else:
                    mb_negatives = np.array([self.all_negatives[wdx,self.negative_context_sampling[wdx,n_negatives_sampled:n_negatives_sampled+self.neg_k]] for wdx in mb_entities])

                loss = self.model.burnin_train(mb_entities, mb_contexts, mb_negatives, self.usePatPool)
                total_loss += loss[0]
                mb_count += 1
                avg_loss = total_loss / mb_count
                sys.stdout.write('average loss: %.6f current loss: %.6f  \r'%(avg_loss,loss[0]))
                n_negatives_sampled += self.neg_k

if __name__ == '__main__':
    n_trials = 1
    n_rounds = 20
    time_start = time.clock()
    emboot = Emboot()

    ## Do this step only if embed_emboot_feat or embed_emboot_global feature is set
    if embed_emboot_feat_name not in emboot.feature_list and embed_emboot_global_feat_name not in emboot.feature_list and semantic_drift_emboot_feat_name not in emboot.feature_list:
        print("NOT CALLING THE EMBOOT SKIP-GRAM TRAINING FUNCTION AS THOSE FEATURES ARE NOT SET")

    emboot.pools_log = open(emboot.args.logfile, 'w')
    if emboot.usePatPool:
        emboot.pools_log_patterns = open(emboot.args.logfile+"_patterns.txt", 'w')
    for trial in range(n_trials):

        emboot.initialize_model()
        emboot.initialize_seeds()

        # print the seeds to the output
        emboot.pools_log.write('Epoch 0\n')
        for cat in emboot.model.pools_entities:
            entities = [emboot.entity_vocab.get_word(c) for c in emboot.model.pools_entities[cat]]
            emboot.pools_log.write('\t'.join([cat] + entities) + '\n')
            emboot.pools_log.flush()

        if emboot.genTsne:
            plot_embeddings(emboot.model,"init", emboot.args)

        time_start_burnin = time.clock()
        if embed_emboot_feat_name in emboot.feature_list or embed_emboot_global_feat_name in emboot.feature_list or semantic_drift_emboot_feat_name in emboot.feature_list: ## Do this step only if embed_emboot_feat or embed_emboot_global feature is set
            emboot.burnin_iterations()

        if emboot.genTsne:
            plot_embeddings(emboot.model,"burnin", emboot.args)

        print("Burn-in completed. Time elapsed : " + str(time.clock()-time_start_burnin))

        # Bootstrapping epoch:
        #########################################################################
        for rdx in range(n_rounds):
            time_start_epoch = time.clock()

            ### Promote patterns using the same strategy as EPB
            if emboot.genTsne:
                plot_embeddings(emboot.model, rdx, emboot.args)

            if emboot.usePatPool:
                emboot.promote_patterns_by_pmi(emboot.args.promote_global, (rdx+1))
                emboot.console_log.flush()

            time_start_skip_gram = time.clock()

            ### Emboot training epoch - learning the embeddings
            if embed_emboot_feat_name in emboot.feature_list or embed_emboot_global_feat_name in emboot.feature_list or semantic_drift_emboot_feat_name in emboot.feature_list: ## Do this step only if embed_emboot_feat or embed_emboot_global feature is set
                emboot.training_epoch()

            emboot.console_log.flush()

            print("Time for skip-gram : " + str(time.clock()-time_start_skip_gram))

            time_start_promotion = time.clock()
            print("Bootstrapping Epoch ", (rdx+1))

            ### Promote entities using a multi-class classifier
            emboot.promote_entities_by_classifier(emboot.args.promote_global, (rdx+1))
            emboot.console_log.flush()

            print("Time to run classifier : " + str(time.clock()-time_start_promotion))

            print("Bootstrapping Epoch " + str(rdx+1) + "\tTime Elapsed : " + str(time.clock()-time_start_epoch))
            emboot.console_log.flush()
        #########################################################################

    emboot.pools_log.close()
    print('elapsed time:', (time.clock()-time_start))
    emboot.console_log.close()
