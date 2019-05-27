import numpy as np
import math
from keras.optimizers import *
from collections import defaultdict
from itertools import product
from w2v import Gigaword
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

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

    def prepare_pairwise_pool(self):
        self.pairwise_pool = dict()
        for cat in self.pools_entities:
            cat_ents = self.pools_entities[cat]
            self.pairwise_pool[cat] = np.array([[cat_x,cat_y] for cat_x in cat_ents for cat_y in cat_ents if cat_x != cat_y],dtype=np.int32)

    def sample_npair_loss(self, mb_size=200):
        # generate random pairs of (head,positive) pairs
        self.ent_heads,self.ent_true,self.ent_false = [],[],[]
        sampled_pairs = dict()
        for cat in self.pools_entities:
            pairwise_ents = self.pairwise_pool[cat]
            sampled_pairs[cat] = pairwise_ents[np.random.randint(pairwise_ents.shape[0],size=mb_size),:]
            self.ent_heads.extend(sampled_pairs[cat][:,0])
            true_cats = np.array([sampled_pairs[cat][:,1] for _ in range(len(self.categories)-1)]).transpose()
            self.ent_true.extend(true_cats)
        # now for each category, gather the samples from other categories, add as negative
        for cat in self.pools_entities:
            other_cats = np.array([sampled_pairs[other_cat][:,1] for other_cat in self.pools_entities if other_cat != cat]).transpose()
            self.ent_false.extend(other_cats)
        self.ent_heads,self.ent_true,self.ent_false = np.array(self.ent_heads),np.array(self.ent_true),np.array(self.ent_false)

    def train(self, mb_entities, mb_contexts, neg_samples, usePatPool):
        return self.keras_train([mb_entities, mb_contexts, neg_samples, self.ent_heads, self.ent_true, self.ent_false])

    def burnin_train(self, mb_entities, mb_contexts, neg_samples, usePatPool):
        return self.keras_burnin_train([mb_entities, mb_contexts, neg_samples, self.ent_heads, self.ent_true, self.ent_false])

    def add_seeds(self, category, seeds):
        vocab = self.get_entity_vocabulary()
        seed_ids = [vocab.get_id(s) for s in seeds if vocab.contains(s)]
        self.pools_entities[category].extend(seed_ids)
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

    def __init__(self, word_vocabulary, context_vocabulary, embedding_size, num_neg_samples, learning_rate, categories, usePatPool, initGigaEmbed, gigaW2vEmbed, lookupGiga, prior_entity_embedding=None, prior_pattern_embedding=None, cat_weight=1e-2, eps=1e-14):

        self.in_vocabulary = word_vocabulary
        self.out_vocabulary = context_vocabulary
        self.embedding_size = embedding_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.pools_entities = defaultdict(list)
        self.pools_contexts = defaultdict(list)
        self.categories = categories
        self.eps = eps
        self.cat_weight = cat_weight

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
        triplet_ent_head = K.placeholder(ndim=1, dtype='int32')
        triplet_ent_true = K.placeholder(ndim=2, dtype='int32')
        triplet_ent_false = K.placeholder(ndim=2, dtype='int32')

        # get embeddings corresponding to minibatch
        in_embs = K.gather(self.in_embeddings, mb_in_indices)
        out_embs = K.gather(self.out_embeddings, mb_out_indices)
        neg_embs = K.gather(self.out_embeddings, mb_negsamples_indices)
        triplet_ent_head_embs = K.gather(self.in_embeddings, triplet_ent_head)
        triplet_ent_true_embs = K.gather(self.in_embeddings, triplet_ent_true)
        triplet_ent_false_embs = K.gather(self.in_embeddings, triplet_ent_false)

        # we want to maximize this objective
        log_prob_positive = K.log(K.sigmoid(K.batch_dot(in_embs, out_embs, axes=1)))
        log_prob_negative = K.sum(K.log(K.sigmoid(-K.batch_dot(K.expand_dims(in_embs), neg_embs, axes=(1, 2)))), axis=2)

        # the multi-class N-pair loss objective
        head_positive = K.batch_dot(K.expand_dims(triplet_ent_head_embs), triplet_ent_true_embs, axes=(1,2))
        head_negative = K.batch_dot(K.expand_dims(triplet_ent_head_embs), triplet_ent_false_embs, axes=(1,2))
        #log_mcnp = K.log(K.mean(K.sigmoid(head_positive - head_negative), axis=2))
        log_mcnp = K.log(1+K.sum(K.exp((head_negative - head_positive)), axis=2))

        objective_burnin = K.mean(log_prob_positive + log_prob_negative)
        optimizer_burnin = SGD(lr=self.learning_rate)
        params_burnin = [self.in_embeddings, self.out_embeddings]
        constraints_burnin = []
        loss_burnin = -objective_burnin # minimize loss => maximize objective
        updates_burnin = optimizer_burnin.get_updates(params_burnin, constraints_burnin, loss_burnin)

        self.keras_burnin_train = K.function(
            [mb_in_indices, mb_out_indices, mb_negsamples_indices, triplet_ent_head,triplet_ent_true,triplet_ent_false],
            [loss_burnin],
            updates=updates_burnin)

        unsupervised_objective = K.mean(log_prob_positive + log_prob_negative)
        supervised_objective = K.mean(log_mcnp)
        objective_train = unsupervised_objective
        #objective_train = unsupervised_objective - supervised_objective
        optimizer_train = SGD(lr=self.learning_rate)
        params_train = [self.in_embeddings, self.out_embeddings]
        constraints_train = []
        loss_train = -objective_train # minimize loss => maximize objective
        updates_train = optimizer_train.get_updates(params_train, constraints_train, loss_train)

        self.keras_train = K.function(
            [mb_in_indices, mb_out_indices, mb_negsamples_indices, triplet_ent_head,triplet_ent_true,triplet_ent_false],
            [loss_train,unsupervised_objective,supervised_objective],
            updates=updates_train)


        # NOTE: To not allocate all the memory to a process in a multi-gpu setting
        # from https://github.com/fchollet/keras/issues/6031
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
        self.session = K.get_session()

