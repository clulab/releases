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

    def compute_similarity(self,emb1,emb2,sign=1.0):
        if self.cossim_margin:
            return cosine_similarity(emb1,emb2)
        # else log-likelihood margin
        return -np.log(1.0/(1.0+np.exp(sign*linear_kernel(emb1,emb2)))+1e-14)

    def compute_margin_tensor(self,head_embedding,head_pool,tail_embedding,tail_pool,head_k=None,k=8):
        # gather embeddings
        local_heads = dict()
        head_embeddings = dict()
        for cat in head_pool:
            all_head = np.array(head_pool[cat],dtype=np.int32)
            if head_k is None or head_k >= all_head.shape[0]
                local_heads[cat] = all_head
            else:
                local_heads[cat] = np.random.choice(all_head,size=head_k,replace=False)
            head_embeddings[cat] = head_embedding[local_heads[cat]]
            if head_embeddings[cat].shape[0]==0:
                return None,None
        tail_embeddings = dict()
        for cat in tail_pool:
            tail_embeddings[cat] = tail_embedding[np.array(tail_pool[cat],dtype=np.int32)]
            if tail_embeddings[cat].shape[0]==0:
                return None,None

        # compute cossine similarities within categories and cossine similarities between categories
        sim_within = dict()
        sim_between = dict()
        local_positives = dict()
        local_negatives = dict()
        for cat in head_pool:
            all_sim_within = self.compute_similarity(head_embeddings[cat],tail_embeddings[cat], sign=-1.0)
            pos_sorted_inds = np.argsort(all_sim_within,axis=-1) if k is None else np.argsort(all_sim_within,axis=-1)[:,:k]
            sim_within[cat] = np.array([all_sim_within[idx,pos_sorted_inds[idx]] for idx in range(all_sim_within.shape[0])])
            local_positives[cat] = pos_sorted_inds
            for other_cat in tail_pool:
                if cat == other_cat:
                    continue
                all_sim_between = self.compute_similarity(head_embeddings[cat],tail_embeddings[other_cat], sign=-1.0)
                neg_sorted_inds = np.argsort(all_sim_between,axis=-1) if k is None else np.argsort(all_sim_between,axis=-1)[:,-k:]
                sim_between[cat+'-'+other_cat] = np.array([all_sim_between[idx,neg_sorted_inds[idx]] for idx in range(all_sim_between.shape[0])])
                local_negatives[cat+'-'+other_cat] = neg_sorted_inds
            #
        #

        # now for each pair of categories, form 3-tensor of similarities
        margin_tensors = dict()
        for cat in head_pool:
            intra_sim_mat = sim_within[cat]
            n_cat,n_k = intra_sim_mat.shape[0],intra_sim_mat.shape[1]
            for other_cat in tail_pool:
                if cat == other_cat:
                    continue
                inter_sim_mat = sim_between[cat+'-'+other_cat]
                n_other_cat = inter_sim_mat.shape[1]

                tiled_intra_sim = np.tile(np.reshape(intra_sim_mat,(n_cat,n_k,1)), (1,1,n_other_cat))
                tiled_inter_sim = np.tile(np.reshape(inter_sim_mat,(n_cat,n_other_cat,1)), (1,1,n_k))
                tiled_inter_sim = np.swapaxes(tiled_inter_sim,1,2)
                if self.cossim_margin:
                    margin_tensors[cat+'-'+other_cat] = tiled_inter_sim - tiled_intra_sim + self.margin
                else:
                    margin_tensors[cat+'-'+other_cat] = tiled_inter_sim + tiled_intra_sim - self.margin
            #
        #

        return margin_tensors,local_heads,local_positives,local_negatives

    def preprocess_triplet(self, usePatPool, head_k=None):
        # within entities, within patterns, between entities and patterns
        entity_embeddings,pattern_embeddings = self.get_in_embeddings(),self.get_context_embeddings()
        self.entity_margin_tensor,self.entity_local_heads,self.entity_local_positives,self.entity_local_negatives = self.compute_margin_tensor(entity_embeddings,self.pools_entities,entity_embeddings,self.pools_entities,head_k=head_k)
        if usePatPool:
            self.pattern_margin_tensor,self.pattern_local_heads,self.pattern_local_positives,self.pattern_local_negatives = self.compute_margin_tensor(pattern_embeddings,self.pools_contexts,pattern_embeddings,self.pools_contexts,head_k=head_k)
            self.entity_pattern_margin_tensor,self.ep_local_heads,self.ep_local_positives,self.ep_local_negatives = self.compute_margin_tensor(entity_embeddings,self.pools_entities,pattern_embeddings,self.pools_contexts,head_k=head_k)

    def hard_negative_triplet_sampling(self, usePatPool, n_between_cat_samples=40):
        # within entity hard negatives
        self.ent_heads,self.ent_false,self.ent_true = [],[],[]
        for cat in self.pools_entities:
            cat_ents = np.array(self.pools_entities[cat],dtype=np.int32)
            local_heads = self.entity_local_heads[cat]
            local_positives = self.entity_local_positives[cat]
            for other_cat in self.pools_entities:
                if cat == other_cat:
                    continue
                other_cat_ents = np.array(self.pools_entities[other_cat],dtype=np.int32)
                local_negatives = self.entity_local_negatives[cat+'-'+other_cat]

                hard_triplets = np.argwhere(self.entity_margin_tensor[cat+'-'+other_cat]>0)
                if n_between_cat_samples < hard_triplets.shape[0]:
                    hard_triplets = hard_triplets[np.random.choice(hard_triplets.shape[0],size=n_between_cat_samples,replace=False),:]

                self.ent_heads.extend([local_heads[head] for head in hard_triplets[:,0]])

                mapped_true_ents = np.array([local_positives[head,pos] for (head,pos) in zip(hard_triplets[:,0],hard_triplets[:,1])],dtype=np.int32)
                self.ent_true.extend(cat_ents[mapped_true_ents])

                mapped_false_ents = np.array([local_negatives[head,pos] for (head,pos) in zip(hard_triplets[:,0],hard_triplets[:,2])],dtype=np.int32)
                self.ent_false.extend(other_cat_ents[mapped_false_ents])
            #
        #
        self.ent_heads,self.ent_true,self.ent_false = np.array(self.ent_heads),np.array(self.ent_true),np.array(self.ent_false)

        # within pattern hard negatives
        self.pat_heads,self.pat_false,self.pat_true = [],[],[]
        if usePatPool and self.pattern_margin_tensor is not None:
            for cat in self.pools_contexts:
                cat_pats = np.array(self.pools_contexts[cat],dtype=np.int32)
                local_heads = self.pattern_local_heads[cat]
                local_positives = self.pattern_local_positives[cat]
                for other_cat in self.pools_contexts:
                    if cat == other_cat:
                        continue
                    other_cat_pats = np.array(self.pools_contexts[other_cat],dtype=np.int32)
                    local_negatives = self.pattern_local_negatives[cat+'-'+other_cat]

                    hard_triplets = np.argwhere(self.pattern_margin_tensor[cat+'-'+other_cat]>0)
                    if n_between_cat_samples < hard_triplets.shape[0]:
                        hard_triplets = hard_triplets[np.random.choice(hard_triplets.shape[0],size=n_between_cat_samples,replace=False),:]

                    self.pat_heads.extend([local_heads[head] for head in hard_triplets[:,0]])

                    mapped_true_pats = np.array([local_positives[head,pos] for (head,pos) in zip(hard_triplets[:,0],hard_triplets[:,1])],dtype=np.int32)
                    self.pat_true.extend(cat_pats[mapped_true_pats])

                    mapped_false_pats = np.array([local_negatives[head,pos] for (head,pos) in zip(hard_triplets[:,0],hard_triplets[:,2])],dtype=np.int32)
                    self.pat_false.extend(other_cat_pats[mapped_false_pats])
                #
            #
        #

        self.pat_heads,self.pat_true,self.pat_false = np.array(self.pat_heads),np.array(self.pat_true),np.array(self.pat_false)

        # inter entity-pattern hard negatives
        self.ent_pat_heads,self.ent_pat_false,self.ent_pat_true = [],[],[]
        if usePatPool and self.pattern_margin_tensor is not None:
            for cat in self.pools_entities:
                cat_ents = np.array(self.pools_entities[cat],dtype=np.int32)
                cat_pats = np.array(self.pools_contexts[cat],dtype=np.int32)
                local_heads = self.ep_local_heads[cat]
                local_positives = self.ep_local_positives[cat]
                for other_cat in self.pools_contexts:
                    if cat == other_cat:
                        continue
                    cat_other_pats = np.array(self.pools_contexts[other_cat],dtype=np.int32)
                    local_negatives = self.ep_local_negatives[cat+'-'+other_cat]

                    hard_triplets = np.argwhere(self.entity_pattern_margin_tensor[cat+'-'+other_cat]>0)
                    if n_between_cat_samples < hard_triplets.shape[0]:
                        hard_triplets = hard_triplets[np.random.choice(hard_triplets.shape[0],size=n_between_cat_samples,replace=False),:]

                    self.ent_pat_heads.extend([local_heads[head] for head in hard_triplets[:,0]])

                    mapped_true_pats = np.array([local_positives[head,pos] for (head,pos) in zip(hard_triplets[:,0],hard_triplets[:,1])],dtype=np.int32)
                    self.ent_pat_true.extend(cat_pats[mapped_true_pats])

                    mapped_false_pats = np.array([local_negatives[head,pos] for (head,pos) in zip(hard_triplets[:,0],hard_triplets[:,2])],dtype=np.int32)
                    self.ent_pat_false.extend(cat_other_pats[mapped_false_pats])
                #
            #
        #
        self.ent_pat_heads,self.ent_pat_true,self.ent_pat_false = np.array(self.ent_pat_heads),np.array(self.ent_pat_true),np.array(self.ent_pat_false)

    def sample_triplet_losses(self, usePatPool, n_samples_per_cat=200):
        # sample entity triplet losses
        self.ent_heads,self.ent_false,self.ent_true = [],[],[]
        for cat in self.pools_entities:
            cat_ents = np.array(self.pools_entities[cat])
            rand_head_inds = np.random.choice(cat_ents.shape[0],size=n_samples_per_cat,replace=cat_ents.shape[0]<n_samples_per_cat)
            rand_true_inds = np.array(rand_head_inds)
            np.random.shuffle(rand_true_inds)

            other_cat_ents = np.concatenate([self.pools_entities[other_cat] for other_cat in self.pools_entities if other_cat != cat],axis=0)
            rand_false_inds = np.random.choice(other_cat_ents.shape[0],size=n_samples_per_cat,replace=other_cat_ents.shape[0]<n_samples_per_cat)
            self.ent_heads.extend(cat_ents[rand_head_inds])
            self.ent_true.extend(cat_ents[rand_true_inds])
            self.ent_false.extend(other_cat_ents[rand_false_inds])
        #
        self.ent_heads = np.array(self.ent_heads)
        self.ent_true = np.array(self.ent_true)
        self.ent_false = np.array(self.ent_false)

        if usePatPool:
            # sample pattern triplet losses
            self.pat_heads,self.pat_false,self.pat_true = [],[],[]
            for cat in self.pools_contexts:
                cat_pats = np.array(self.pools_contexts[cat])
                if cat_pats.shape[0]==0:
                    continue
                rand_head_inds = np.random.choice(cat_pats.shape[0],size=n_samples_per_cat,replace=cat_pats.shape[0]<n_samples_per_cat)
                rand_true_inds = np.array(rand_head_inds)
                np.random.shuffle(rand_true_inds)

                other_cat_pats = np.concatenate([self.pools_contexts[other_cat] for other_cat in self.pools_contexts if other_cat != cat],axis=0)
                rand_false_inds = np.random.choice(other_cat_pats.shape[0],size=n_samples_per_cat,replace=other_cat_pats.shape[0]<n_samples_per_cat)
                self.pat_heads.extend(cat_pats[rand_head_inds])
                self.pat_true.extend(cat_pats[rand_true_inds])
                self.pat_false.extend(other_cat_pats[rand_false_inds])
            #
            self.pat_heads = np.array(self.pat_heads)
            self.pat_true = np.array(self.pat_true)
            self.pat_false = np.array(self.pat_false)

            # sample entity-pattern triplet losses
            self.ent_pat_heads,self.ent_pat_false,self.ent_pat_true = [],[],[]
            for cat in self.pools_entities:
                cat_ents = np.array(self.pools_entities[cat])
                rand_head_inds = np.random.choice(cat_ents.shape[0],size=n_samples_per_cat,replace=cat_ents.shape[0]<n_samples_per_cat)
                cat_pats = np.array(self.pools_contexts[cat])
                if cat_pats.shape[0]==0:
                    continue
                rand_true_inds = np.random.choice(cat_pats.shape[0],size=n_samples_per_cat,replace=cat_pats.shape[0]<n_samples_per_cat)

                other_cat_pats = np.concatenate([self.pools_contexts[other_cat] for other_cat in self.pools_contexts if other_cat != cat],axis=0)
                rand_false_inds = np.random.choice(other_cat_pats.shape[0],size=n_samples_per_cat,replace=other_cat_pats.shape[0]<n_samples_per_cat)
                self.ent_pat_heads.extend(cat_ents[rand_head_inds])
                self.ent_pat_true.extend(cat_pats[rand_true_inds])
                self.ent_pat_false.extend(other_cat_pats[rand_false_inds])
            #
            self.ent_pat_heads = np.array(self.ent_pat_heads)
            self.ent_pat_true = np.array(self.ent_pat_true)
            self.ent_pat_false = np.array(self.ent_pat_false)

    def train(self, mb_entities, mb_contexts, neg_samples, usePatPool):
        if usePatPool:
            return self.keras_train([mb_entities, mb_contexts, neg_samples,
                                            self.ent_heads, self.ent_true, self.ent_false,
                                            self.pat_heads, self.pat_true, self.pat_false,
                                            self.ent_pat_heads, self.ent_pat_true, self.ent_pat_false])
        else:
            return self.keras_train([mb_entities, mb_contexts, neg_samples, self.ent_heads, self.ent_true, self.ent_false])

    def burnin_train(self, mb_entities, mb_contexts, neg_samples, usePatPool):
        if usePatPool:
            return self.keras_burnin_train([mb_entities, mb_contexts, neg_samples,
                                            self.ent_heads, self.ent_true, self.ent_false,
                                            self.pat_heads, self.pat_true, self.pat_false,
                                            self.ent_pat_heads, self.ent_pat_true, self.ent_pat_false])
        else:
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

    def __init__(self, word_vocabulary, context_vocabulary, embedding_size, num_neg_samples, learning_rate, categories, usePatPool, initGigaEmbed, gigaW2vEmbed, lookupGiga, prior_entity_embedding=None, prior_pattern_embedding=None, margin=0.5, eps=1e-14, cossim_margin=False, entity_weights=None, pattern_weights=None):

        self.in_vocabulary = word_vocabulary
        self.out_vocabulary = context_vocabulary
        self.embedding_size = embedding_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.pools_entities = defaultdict(list)
        self.pools_contexts = defaultdict(list)
        self.categories = categories
        self.margin = margin
        self.eps = eps
        self.cossim_margin = cossim_margin

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

        self.entity_weights = entity_weights
        self.pattern_weights = pattern_weights
        if entity_weights is None:
            self.entity_weights = np.ones(self.in_vocabulary.size())
        if pattern_weights is None:
            self.pattern_weights = np.ones(self.out_vocabulary.size())
        self.prior_entity_confidence = K.constant(value=self.entity_weights, shape=self.entity_weights.shape)
        self.prior_pattern_confidence = K.constant(value=self.pattern_weights, shape=self.pattern_weights.shape)

        # placeholders for minibatch
        # they need to be ints because they are used as indices
        # they will be provided when calling keras_train() (see below)
        mb_in_indices = K.placeholder(ndim=1, dtype='int32')
        mb_out_indices = K.placeholder(ndim=1, dtype='int32')
        mb_negsamples_indices = K.placeholder(ndim=2, dtype='int32')
        triplet_ent_head = K.placeholder(ndim=1, dtype='int32')
        triplet_ent_true = K.placeholder(ndim=1, dtype='int32')
        triplet_ent_false = K.placeholder(ndim=1, dtype='int32')

        # get embeddings corresponding to minibatch
        in_embs = K.gather(self.in_embeddings, mb_in_indices)
        out_embs = K.gather(self.out_embeddings, mb_out_indices)
        neg_embs = K.gather(self.out_embeddings, mb_negsamples_indices)

        triplet_ent_head_embs = K.gather(self.in_embeddings, triplet_ent_head)
        triplet_ent_true_embs = K.gather(self.in_embeddings, triplet_ent_true)
        triplet_ent_false_embs = K.gather(self.in_embeddings, triplet_ent_false)
        triplet_ent_head_conf = K.gather(self.prior_entity_confidence, triplet_ent_head)
        triplet_ent_true_conf = K.gather(self.prior_entity_confidence, triplet_ent_true)
        triplet_ent_false_conf = K.gather(self.prior_entity_confidence, triplet_ent_false)

        ## initialize the variables to use pattern embeddings
        triplet_pat_head = K.placeholder(ndim=1, dtype='int32')
        triplet_pat_true = K.placeholder(ndim=1, dtype='int32')
        triplet_pat_false = K.placeholder(ndim=1, dtype='int32')

        triplet_pat_head_embs = K.gather(self.out_embeddings, triplet_pat_head)
        triplet_pat_true_embs = K.gather(self.out_embeddings, triplet_pat_true)
        triplet_pat_false_embs = K.gather(self.out_embeddings, triplet_pat_false)
        triplet_pat_head_conf = K.gather(self.prior_pattern_confidence, triplet_pat_head)
        triplet_pat_true_conf = K.gather(self.prior_pattern_confidence, triplet_pat_true)
        triplet_pat_false_conf = K.gather(self.prior_pattern_confidence, triplet_pat_false)

        ## mixed entity/pattern placeholders - TODO: probably want all permutations of entity/patterns to avoid potential bias...
        triplet_ent_pat_head = K.placeholder(ndim=1, dtype='int32')
        triplet_ent_pat_true = K.placeholder(ndim=1, dtype='int32')
        triplet_ent_pat_false = K.placeholder(ndim=1, dtype='int32')

        triplet_ent_pat_head_embs = K.gather(self.in_embeddings, triplet_ent_pat_head)
        triplet_ent_pat_true_embs = K.gather(self.out_embeddings, triplet_ent_pat_true)
        triplet_ent_pat_false_embs = K.gather(self.out_embeddings, triplet_ent_pat_false)
        triplet_ent_pat_head_conf = K.gather(self.prior_entity_confidence, triplet_ent_pat_head)
        triplet_ent_pat_true_conf = K.gather(self.prior_pattern_confidence, triplet_ent_pat_true)
        triplet_ent_pat_false_conf = K.gather(self.prior_pattern_confidence, triplet_ent_pat_false)

        # we want to maximize this objective
        log_prob_positive = K.log(K.sigmoid(K.batch_dot(in_embs, out_embs, axes=1)))
        log_prob_negative = K.sum(K.log(K.sigmoid(-K.batch_dot(K.expand_dims(in_embs), neg_embs, axes=(1, 2)))), axis=2)

        ## If patPools are used we need to consider the pools of both entities and patterns for the triplet loss
        if usePatPool:
            head_concat = K.concatenate([triplet_ent_head_embs,triplet_pat_head_embs,triplet_ent_pat_head_embs],axis=0)
            true_concat = K.concatenate([triplet_ent_true_embs,triplet_pat_true_embs,triplet_ent_pat_true_embs],axis=0)
            false_concat = K.concatenate([triplet_ent_false_embs,triplet_pat_false_embs,triplet_ent_pat_false_embs],axis=0)

            if self.cossim_margin:
                normalized_head,normalized_true,normalized_false = K.l2_normalize(head_concat,axis=1),K.l2_normalize(true_concat,axis=1),K.l2_normalize(false_concat,axis=1)
                true_cos_sims = K.batch_dot(normalized_head,normalized_true, axes=1)
                false_cos_sims = K.batch_dot(normalized_head,normalized_false, axes=1)
                margin_loss = K.maximum(K.constant(0.0), false_cos_sims - true_cos_sims + K.constant(self.margin))
            else:
                true_head_scores = K.log(K.sigmoid(K.batch_dot(head_concat, true_concat, axes=1)) + K.constant(self.eps))
                false_head_scores = K.log(K.sigmoid(-K.batch_dot(head_concat, false_concat, axes=1)) + K.constant(self.eps))
                margin_loss = K.maximum(K.constant(0.0), -(false_head_scores + true_head_scores) - K.constant(self.margin))

        else: # triplet-based margin loss just for entities
            if self.cossim_margin:
                normalized_head,normalized_true,normalized_false = K.l2_normalize(triplet_ent_head_embs,axis=1),K.l2_normalize(triplet_ent_true_embs,axis=1),K.l2_normalize(triplet_ent_false_embs,axis=1)
                true_cos_sims = K.batch_dot(normalized_head,normalized_true, axes=1)
                false_cos_sims = K.batch_dot(normalized_head,normalized_false, axes=1)
                triplet_confidence = (triplet_ent_head_conf+triplet_ent_true_conf+triplet_ent_false_conf) / K.constant(3.0)
                the_hinge = tf.squeeze(K.maximum(K.constant(0.0), false_cos_sims - true_cos_sims + K.constant(self.margin)))
                margin_loss = tf.multiply(triplet_confidence,the_hinge)
            else:
                true_head_scores = K.log(K.sigmoid(K.batch_dot(triplet_ent_head_embs, triplet_ent_true_embs, axes=1)) + K.constant(self.eps))
                false_head_scores = K.log(K.sigmoid(-K.batch_dot(triplet_ent_head_embs, triplet_ent_false_embs, axes=1)) + K.constant(self.eps))
                margin_loss = K.maximum(K.constant(0.0), -(false_head_scores - true_head_scores) - K.constant(self.margin))

        objective_burnin = K.mean(log_prob_positive + log_prob_negative)
        optimizer_burnin = SGD(lr=self.learning_rate)
        params_burnin = [self.in_embeddings, self.out_embeddings]
        constraints_burnin = []
        loss_burnin = -objective_burnin # minimize loss => maximize objective
        updates_burnin = optimizer_burnin.get_updates(params_burnin, constraints_burnin, loss_burnin)

        if usePatPool:
            self.keras_burnin_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices,
                 triplet_ent_head,triplet_ent_true,triplet_ent_false,
                 triplet_pat_head,triplet_pat_true,triplet_pat_false,
                 triplet_ent_pat_head,triplet_ent_pat_true,triplet_ent_pat_false],
                [loss_burnin],
                updates=updates_burnin)
        else:
            self.keras_burnin_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices, triplet_ent_head,triplet_ent_true,triplet_ent_false],
                [loss_burnin],
                updates=updates_burnin)

        objective_train = K.mean(log_prob_positive + log_prob_negative) - K.mean(margin_loss)
        optimizer_train = SGD(lr=self.learning_rate)
        params_train = [self.in_embeddings, self.out_embeddings]
        constraints_train = []
        loss_train = -objective_train # minimize loss => maximize objective
        updates_train = optimizer_train.get_updates(params_train, constraints_train, loss_train)

        if usePatPool:
            self.keras_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices,
                 triplet_ent_head,triplet_ent_true,triplet_ent_false,
                 triplet_pat_head,triplet_pat_true,triplet_pat_false,
                 triplet_ent_pat_head,triplet_ent_pat_true,triplet_ent_pat_false],
                [loss_train,margin_loss],
                updates=updates_train)
        else:
            self.keras_train = K.function(
                [mb_in_indices, mb_out_indices, mb_negsamples_indices, triplet_ent_head,triplet_ent_true,triplet_ent_false],
                [loss_train,margin_loss,triplet_confidence,the_hinge],
                updates=updates_train)


        # NOTE: To not allocate all the memory to a process in a multi-gpu setting
        # from https://github.com/fchollet/keras/issues/6031
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
        self.session = K.get_session()

