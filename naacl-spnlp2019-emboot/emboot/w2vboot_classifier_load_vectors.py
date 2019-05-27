import numpy as np
import math
from keras.optimizers import *
from collections import defaultdict
from itertools import product
from w2v import Gigaword
import tensorflow as tf

orig_entity_embedding = "pools_output_interpretable.txt_entemb.txt"
orig_pattern_embedding = "pools_output_interpretable.txt_ctxemb.txt"

def load_vectors(filename):
    vector_dict = {}
    with open(filename) as oee:
        ent_vecs = oee.readlines()
        for line in ent_vecs:
            line = line.strip().split("\t")
            line2 = line[1].split()
            vector_dict[line[0]] = [float(x) for x in line2]
    vector_list = []
    for item in vector_dict:
        vector_list.append(vector_dict[item])
    vector_tensor = tf.convert_to_tensor(np.asarray(vector_list), dtype=tf.float64)
    return vector_tensor

#load_vectors(orig_entity_embedding)

class Word2vec(object):

    def get_mean_entPool_embedding(self, category):
        """return the mean embedding of the entity pool of the given category"""
        pool = self.pools_entities[category]
        embeddings = self.get_entity_embeddings()
        return np.mean(Gigaword.norm(embeddings[pool]), axis=0)

    def get_entity_vocabulary(self):
        return self.in_vocabulary

    def get_context_vocabulary(self):
        return self.out_vocabulary

    def get_in_embeddings(self):
        """return normalized in embeddings as numpy array"""
        # return normalize(self.in_embeddings).eval(session=self.session)
        return self.in_embeddings.eval(session=self.session)

    def get_entity_embeddings(self):
        """return normalized word embeddings as numpy array"""
        return self.get_in_embeddings()

    def get_context_embeddings(self):
        """return normalized word embeddings as numpy array"""
        return self.out_embeddings.eval(session=self.session)

    def write_embeddings(self, entity_embedding_filename, pattern_embedding_filename):
        entity_embedding = self.get_in_embeddings()
        pattern_embedding = self.get_context_embeddings()
        np.save(entity_embedding_filename, entity_embedding)
        np.save(pattern_embedding_filename, pattern_embedding)

    def gather_eq_pools(self, usePatPool):
        self.eq1 = []
        self.eq2 = []
        for cat in self.pools_entities:
            for e1, e2 in product(self.pools_entities[cat], repeat=2):
                if e1 != e2:
                    self.eq1.append(e1)
                    self.eq2.append(e2)
        self.eq1 = np.array(self.eq1)
        self.eq2 = np.array(self.eq2)

        if usePatPool:
            self.eq_pat1 = []
            self.eq_pat2 = []
            for cat in self.categories:
                for p1, p2 in product(self.pools_contexts[cat], repeat=2):
                    if p1 != p2:
                        self.eq_pat1.append(p1)
                        self.eq_pat2.append(p2)
            self.eq_pat1 = np.array(self.eq_pat1)
            self.eq_pat2 = np.array(self.eq_pat2)

            self.eq_ep_ent = []
            self.eq_ep_pat = []
            for cat in self.categories:
                for e1,p1 in product(self.pools_entities[cat], self.pools_contexts[cat], repeat=1):
                    self.eq_ep_ent.append(e1)
                    self.eq_ep_pat.append(p1)
            self.eq_ep_ent = np.array(self.eq_ep_ent)
            self.eq_ep_pat = np.array(self.eq_ep_pat)

    def gather_ne_pools(self, usePatPool):
        self.ne1 = []
        self.ne2 = []
        for cat1 in self.pools_entities:
            for cat2 in self.pools_entities:
                if cat1 != cat2:
                    for n1, n2 in product(self.pools_entities[cat1], self.pools_entities[cat2]):
                        self.ne1.append(n1)
                        self.ne2.append(n2)
        self.ne1 = np.array(self.ne1)
        self.ne2 = np.array(self.ne2)

        if usePatPool:
            self.ne_pat1 = []
            self.ne_pat2 = []
            for cat1 in self.categories:
                for cat2 in self.categories:
                    if cat1 != cat2:
                        for n1, n2 in product(self.pools_contexts[cat1], self.pools_contexts[cat2]):
                            self.ne_pat1.append(n1)
                            self.ne_pat2.append(n2)
            self.ne_pat1 = np.array(self.ne_pat1)
            self.ne_pat2 = np.array(self.ne_pat2)

            self.ne_ep_ent = []
            self.ne_ep_pat = []
            for cat1 in self.categories:
                for cat2 in self.categories:
                    if cat1 != cat2:
                        for n1,n2 in product(self.pools_entities[cat1], self.pools_contexts[cat2]):
                            self.ne_ep_ent.append(n1)
                            self.ne_ep_pat.append(n2)
            self.ne_ep_ent = np.array(self.ne_ep_ent)
            self.ne_ep_pat = np.array(self.ne_ep_pat)

    def train(self, mb_entities, mb_contexts, neg_samples, usePatPool):
        if usePatPool:
            return self.keras_train([mb_entities, mb_contexts, neg_samples,
                                            self.eq1, self.eq2,
                                            self.ne1, self.ne2,
                                            self.eq_pat1, self.eq_pat2,
                                            self.ne_pat1, self.ne_pat2,
                                            self.eq_ep_ent, self.eq_ep_pat,
                                            self.ne_ep_ent, self.ne_ep_pat])
        else:
            return self.keras_train([mb_entities, mb_contexts, neg_samples, self.eq1, self.eq2, self.ne1, self.ne2])

    def burnin_train(self, mb_entities, mb_contexts, neg_samples, usePatPool):
        if usePatPool:
            return self.keras_burnin_train([mb_entities, mb_contexts, neg_samples,
                                            self.eq1, self.eq2,
                                            self.ne1, self.ne2,
                                            self.eq_pat1, self.eq_pat2,
                                            self.ne_pat1, self.ne_pat2,
                                            self.eq_ep_ent, self.eq_ep_pat,
                                            self.ne_ep_ent, self.ne_ep_pat])
        else:
            return self.keras_burnin_train([mb_entities, mb_contexts, neg_samples, self.eq1, self.eq2, self.ne1, self.ne2])

    def add_seeds(self, category, seeds):
        vocab = self.get_entity_vocabulary()
        seed_ids = [vocab.get_id(s) for s in seeds if vocab.contains(s)]
        self.pools_entities[category].extend(seed_ids)

        seed_ids_with_epoch = [ (s,0) for s in seed_ids ] ## NOTE: Add epoch id along with entities (for semantic drift features) -- 0 for seeds
        self.pools_entities_with_epoch[category].extend(seed_ids_with_epoch)

        self.pools_contexts[category].extend([])

    def initialize_embeddings_with_gigaword(self, gigaW2vEmbed, lookupGiga):
        entity_vocab = self.get_entity_vocabulary()
        giga_init_embed_vector = list()
        for entId in range(0, entity_vocab.size()): ## find the entity string
            entityString = entity_vocab.get_word(entId)
            sanitisedEntityString = [ Gigaword.sanitiseWord(tok) ## sanitise every token in the entity
                           for tok in entityString.split(" ")]
            giga_index = [lookupGiga[tok] ## find the index of the individual tokens in the giagword embeddings
                          if tok in lookupGiga
                          else lookupGiga["<unk>"]
                          for tok in sanitisedEntityString]
            giga_embed_entityString = Gigaword.norm(np.average(gigaW2vEmbed[giga_index], axis=0)) #TODO: Check this
            giga_init_embed_vector.append(giga_embed_entityString)

        return np.vstack(giga_init_embed_vector)

    def xavier_init(self, embedding_size):
        return math.sqrt(6) / math.sqrt(embedding_size)

    def __init__(self, word_vocabulary, context_vocabulary, embedding_size, num_neg_samples, learning_rate, categories, usePatPool, initGigaEmbed, gigaW2vEmbed, lookupGiga, pools_entities, pools_contexts, prior_entity_embedding=load_vectors(orig_entity_embedding), prior_pattern_embedding=load_vectors(orig_pattern_embedding)):

        self.in_vocabulary = word_vocabulary
        self.out_vocabulary = context_vocabulary
        self.embedding_size = embedding_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        #self.pools_entities = defaultdict(list)
        self.pools_entities = pools_entities
        self.pools_entities_with_epoch = defaultdict(list) # creating new data structure to implement the semantic drift feature. New data structure to preserve backward compatability of pools_entities with the rest of code.
        #self.pools_contexts = defaultdict(list)
        self.pools_contexts = pools_contexts
        self.categories = categories

        if initGigaEmbed == True:
            print("Initializing embeddings from the gigaword embeddings...")
            self.embedding_size = gigaW2vEmbed.shape[1] ## NOTE: Re-initialize embedding size to that of gigaword embeddings
            giga_init_embeddings = self.initialize_embeddings_with_gigaword(gigaW2vEmbed, lookupGiga)
            self.in_embeddings = K.variable(value=giga_init_embeddings)
        elif prior_entity_embedding is not None:
            self.embedding_size = prior_entity_embedding.shape[1] ## NOTE: Re-initialize embedding size to that of prior embeddings
            self.in_embeddings = K.variable(value=prior_entity_embedding)
        else:
            print("Initializing randomly (xavier initialization)... ")
            # initialize embeddings randomly using xavier initialization
            xavier = self.xavier_init(embedding_size)
            value = np.random.uniform(-xavier, xavier, (self.in_vocabulary.size(), self.embedding_size))
            self.in_embeddings = K.variable(value=value)
            print("\nself.in_embeddings:\t", self.in_embeddings)

        if prior_pattern_embedding is not None:
            # assumes entity and pattern dimensions are the same
            self.out_embeddings = K.variable(value=prior_pattern_embedding)
        else:
            xavier = self.xavier_init(embedding_size)
            value = np.random.uniform(-xavier, xavier, (self.out_vocabulary.size(), self.embedding_size))
            self.out_embeddings = K.variable(value=value)

        # placeholders for minibatch
        # they need to be ints because they are used as indices
        # they will be provided when calling keras_train() (see below)
        mb_in_indices = K.placeholder(ndim=1, dtype='int32')
        mb_out_indices = K.placeholder(ndim=1, dtype='int32')
        mb_negsamples_indices = K.placeholder(ndim=2, dtype='int32')
        pool_eq1 = K.placeholder(ndim=1, dtype='int32')
        pool_eq2 = K.placeholder(ndim=1, dtype='int32')
        pool_ne1 = K.placeholder(ndim=1, dtype='int32')
        pool_ne2 = K.placeholder(ndim=1, dtype='int32')

        # get embeddings corresponding to minibatch
        in_embs = K.gather(self.in_embeddings, mb_in_indices)
        out_embs = K.gather(self.out_embeddings, mb_out_indices)
        neg_embs = K.gather(self.out_embeddings, mb_negsamples_indices)
        eq1_embs = K.gather(self.in_embeddings, pool_eq1)
        eq2_embs = K.gather(self.in_embeddings, pool_eq2)
        ne1_embs = K.gather(self.in_embeddings, pool_ne1)
        ne2_embs = K.gather(self.in_embeddings, pool_ne2)

        ## initialize the variables to use pattern embeddings
        pool_eq_pat1 = K.placeholder(ndim=1, dtype='int32')
        pool_eq_pat2 = K.placeholder(ndim=1, dtype='int32')
        pool_ne_pat1 = K.placeholder(ndim=1, dtype='int32')
        pool_ne_pat2 = K.placeholder(ndim=1, dtype='int32')

        eq_pat1_embs = K.gather(self.out_embeddings, pool_eq_pat1)
        eq_pat2_embs = K.gather(self.out_embeddings, pool_eq_pat2)
        ne_pat1_embs = K.gather(self.out_embeddings, pool_ne_pat1)
        ne_pat2_embs = K.gather(self.out_embeddings, pool_ne_pat2)

        pool_eq_ep_ent = K.placeholder(ndim=1, dtype='int32')
        pool_eq_ep_pat = K.placeholder(ndim=1, dtype='int32')
        pool_ne_ep_ent = K.placeholder(ndim=1, dtype='int32')
        pool_ne_ep_pat = K.placeholder(ndim=1, dtype='int32')

        eq_ep_ent_embs = K.gather(self.in_embeddings, pool_eq_ep_ent) ## NOTE: in_embeddings here
        eq_ep_pat_embs = K.gather(self.out_embeddings, pool_eq_ep_pat) ## NOTE: out_embeddings here
        ne_ep_ent_embs = K.gather(self.in_embeddings, pool_ne_ep_ent) ## NOTE: in_embeddings here
        ne_ep_pat_embs = K.gather(self.out_embeddings, pool_ne_ep_pat) ## NOTE: out_embeddings here

        # we want to maximize this objective
        log_prob_positive = K.log(K.sigmoid(K.batch_dot(in_embs, out_embs, axes=1)))
        log_prob_negative = K.sum(K.log(K.sigmoid(-K.batch_dot(K.expand_dims(in_embs), neg_embs, axes=(1, 2)))), axis=2)

        ## If patPools are used we need to consider the pools of both entities and patterns for "push-pull"
        if usePatPool:
            eq1_concat = K.concatenate([eq1_embs, eq_pat1_embs, eq_ep_ent_embs], axis=0)
            eq2_concat = K.concatenate([eq2_embs, eq_pat2_embs, eq_ep_pat_embs], axis=0)
            ne1_concat = K.concatenate([ne1_embs, ne_pat1_embs, ne_ep_ent_embs], axis=0)
            ne2_concat = K.concatenate([ne2_embs, ne_pat2_embs, ne_ep_pat_embs], axis=0)

            log_prob_eq = K.log(K.sigmoid(K.batch_dot(eq1_concat, eq2_concat, axes=1)))
            log_prob_ne = K.log(K.sigmoid(-K.batch_dot(ne1_concat, ne2_concat, axes=1)))

        else: # consider only the entity pools for "push-pull"
            log_prob_eq = K.log(K.sigmoid(K.batch_dot(eq1_embs, eq2_embs, axes=1)))
            log_prob_ne = K.log(K.sigmoid(-K.batch_dot(ne1_embs, ne2_embs, axes=1)))

        objective_burnin = K.mean(log_prob_positive + log_prob_negative)
        optimizer_burnin = SGD(lr=self.learning_rate)
        params_burnin = [self.in_embeddings, self.out_embeddings]
        constraints_burnin = []
        loss_burnin = -objective_burnin # minimize loss => maximize objective
        updates_burnin = optimizer_burnin.get_updates(params_burnin, constraints_burnin, loss_burnin)

        if usePatPool:
            self.keras_burnin_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices,
                 pool_eq1, pool_eq2,
                 pool_ne1, pool_ne2,
                 pool_eq_pat1, pool_eq_pat2,
                 pool_ne_pat1, pool_ne_pat2,
                 pool_eq_ep_ent, pool_eq_ep_pat,
                 pool_ne_ep_ent, pool_ne_ep_pat],
                [loss_burnin],
                updates=updates_burnin)
        else:
            self.keras_burnin_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices, pool_eq1, pool_eq2, pool_ne1, pool_ne2],
                [loss_burnin],
                updates=updates_burnin)

        objective_train = K.mean(log_prob_positive + log_prob_negative) + K.mean(log_prob_eq) + K.mean(log_prob_ne)
        optimizer_train = SGD(lr=self.learning_rate)
        params_train = [self.in_embeddings, self.out_embeddings]
        constraints_train = []
        loss_train = -objective_train # minimize loss => maximize objective
        updates_train = optimizer_train.get_updates(params_train, constraints_train, loss_train)

        if usePatPool:
            self.keras_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices,
                 pool_eq1, pool_eq2,
                 pool_ne1, pool_ne2,
                 pool_eq_pat1, pool_eq_pat2,
                 pool_ne_pat1, pool_ne_pat2,
                 pool_eq_ep_ent, pool_eq_ep_pat,
                 pool_ne_ep_ent, pool_ne_ep_pat],
                [loss_train],
                updates=updates_train)
        else:
            self.keras_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices, pool_eq1, pool_eq2, pool_ne1, pool_ne2],
                [loss_train],
                updates=updates_train)


        # NOTE: To not allocate all the memory to a process in a multi-gpu setting
        # from https://github.com/fchollet/keras/issues/6031
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
        self.session = K.get_session()
