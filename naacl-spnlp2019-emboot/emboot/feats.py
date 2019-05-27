import sys
import numpy as np
from w2v import Gigaword
from vocabulary import Vocabulary

class PMIFeats:
    def __init__(self,entity_context_cooccurrrence_counts,entity_vocab,context_vocab):
        self.entity_vocab = entity_vocab
        self.context_vocab = context_vocab
        self.n_entities = self.entity_vocab.size()
        self.n_patterns = self.context_vocab.size()

        self.entity_occs = np.array([self.entity_vocab.get_count(e_id) for e_id in range(self.n_entities)])
        self.context_occs = np.array([self.context_vocab.get_count(c_id) for c_id in range(self.n_patterns)])
        self.total_entity_occs = np.sum(self.entity_occs)
        self.total_context_occs = np.sum(self.context_occs)
        self.total_entity_context_occs = sum(entity_context_cooccurrrence_counts.values())

        # TODO: make it SPARSE!
        self.cooccs = np.zeros((self.n_entities,self.n_patterns))
        for e_id,p_id in entity_context_cooccurrrence_counts:
            self.cooccs[e_id,p_id] = entity_context_cooccurrrence_counts[e_id,p_id]
        print('cooccs shape:',self.cooccs.shape)
    #

    def cat_pmi(self,entity_ids,cat_pats,return_probs=False):
        cat_cooccs = self.cooccs[:,cat_pats][entity_ids,:] / self.total_entity_context_occs
        cat_entity_occs_inv = (1.0 / (self.entity_occs[entity_ids] / self.total_entity_occs)).reshape((entity_ids.shape[0],1))
        cat_context_occs_inv = (1.0 / (self.context_occs[cat_pats] / self.total_context_occs)).reshape((cat_pats.shape[0],1))

        pmi = np.log( np.maximum(1e-16,((cat_entity_occs_inv*cat_cooccs).T*cat_context_occs_inv).T) )
        pmi[cat_cooccs==0.0] = 0.0

        if return_probs:
            return pmi,cat_cooccs
        else:
            return pmi
    #

    def compute_pmi_feats(self,model,entity_ids,all_cats):
        ent_ids = np.array(entity_ids,dtype=np.int32)
        n_ents = ent_ids.shape[0]
        ent_range = np.arange(n_ents)
        cat_pmi_feats = []
        for cat in all_cats:
            patternIDs = np.array(model.pools_contexts[cat],dtype=np.int32)
            cat_pmi = self.cat_pmi(ent_ids,patternIDs)
            cat_pmi_max = np.max(cat_pmi,axis=1).reshape((n_ents,1))
            cat_pmi_mean = np.mean(cat_pmi,axis=1).reshape((n_ents,1))
            cat_pmi_feats.append(np.concatenate((cat_pmi,cat_pmi_max,cat_pmi_mean),axis=1))
        pmi_feat_mat = np.concatenate([feats[:,:-2] for feats in cat_pmi_feats],axis=1)

        pmi_max = np.concatenate([feats[:,-2:-1] for feats in cat_pmi_feats],axis=1)
        pmi_max_argmax = np.argmax(pmi_max,axis=1)
        pmi_max_feat_mat = np.zeros(pmi_max.shape)
        pmi_max_feat_mat[ent_range,pmi_max_argmax] = pmi_max[ent_range,pmi_max_argmax]

        pmi_mean = np.concatenate([feats[:,-1:] for feats in cat_pmi_feats],axis=1)
        pmi_mean_argmax = np.argmax(pmi_mean,axis=1)
        pmi_mean_feat_mat = np.zeros(pmi_mean.shape)
        pmi_mean_feat_mat[ent_range,pmi_mean_argmax] = pmi_mean[ent_range,pmi_mean_argmax]

        return np.concatenate((pmi_feat_mat,pmi_max_feat_mat,pmi_mean_feat_mat),axis=1)

class EmbeddingFeats:
    def __init__(self):
        pass

    def cat_embedding_feat(self,model,entity_ids,cat_entity_ids,is_pool_entities=False,k=None):
        all_embeddings = model.get_entity_embeddings()
        entity_embeddings = Gigaword.norm(all_embeddings[entity_ids])
        cat_entity_embeddings = Gigaword.norm(all_embeddings[cat_entity_ids])
        cos_sims = np.dot(entity_embeddings,cat_entity_embeddings.T)

        if is_pool_entities:
            # TODO: assumes at least some (or none) of entity ids are in category -> can be faster if we assume they all are, but for generality...
            mean_scores,min_scores,max_scores=[],[],[]
            for edx,ent_id in enumerate(entity_ids):
                unique_cos_sims = cos_sims[edx,cat_entity_ids!=ent_id]
                unique_cos_sims.sort()
                cos_sims_knn = unique_cos_sims if k is None else unique_cos_sims[-k:]
                mean_scores.append(np.mean(cos_sims_knn))
                min_scores.append(np.min(cos_sims_knn))
                max_scores.append(np.max(cos_sims_knn))
            return np.array(mean_scores),np.array(min_scores),np.array(max_scores)
        else:
            cos_sims.sort(axis=-1)
            cos_sims_knn = cos_sims if k is None else cos_sims[:,-k:]
            mean_scores = np.mean(cos_sims_knn,axis=1)
            min_scores = np.min(cos_sims_knn,axis=1)
            max_scores = np.max(cos_sims_knn,axis=1)
            return mean_scores,min_scores,max_scores

    def compute_embedding_feats(self,model,entity_ids,all_cats,is_pool_entities=False):
        ent_ids = np.array(entity_ids,dtype=np.int32)
        n_ents = ent_ids.shape[0]
        ent_range = np.arange(n_ents)
        cat_embedding_feats = []
        for cat in all_cats:
            cat_ent_ids = np.array(model.pools_entities[cat],dtype=np.int32)
            mean_cat_scores,min_cat_scores,max_cat_scores = self.cat_embedding_feat(model,ent_ids,cat_ent_ids,is_pool_entities)
            cat_embedding_feats.append(np.concatenate((mean_cat_scores.reshape((n_ents,1)),min_cat_scores.reshape((n_ents,1)),max_cat_scores.reshape((n_ents,1))),axis=1))
        embedding_feat_mat = np.concatenate(cat_embedding_feats,axis=1)

        embedding_mean = np.concatenate([feats[:,:1] for feats in cat_embedding_feats],axis=1)
        embedding_mean_argmax = np.argmax(embedding_mean,axis=1)
        embedding_mean_feat_mat = np.zeros(embedding_mean.shape)
        embedding_mean_feat_mat[ent_range,embedding_mean_argmax] = embedding_mean[ent_range,embedding_mean_argmax]

        embedding_max = np.concatenate([feats[:,-1:] for feats in cat_embedding_feats],axis=1)
        embedding_max_argmax = np.argmax(embedding_max,axis=1)
        embedding_max_feat_mat = np.zeros(embedding_max.shape)
        embedding_max_feat_mat[ent_range,embedding_max_argmax] = embedding_max[ent_range,embedding_max_argmax]

        return np.concatenate((embedding_feat_mat,embedding_mean_feat_mat,embedding_max_feat_mat),axis=1)

class GigawordFeats:
    def __init__(self, gigaW2vEmbed, lookupGiga, entity_vocab):
        self.giga_w2v_embedding = gigaW2vEmbed
        self.lookup_giga = lookupGiga
        self.entity_vocab = entity_vocab

        self.w2v_entity_embeddings = []
        for wdx in range(self.entity_vocab.size()):
            entity_phrase = self.entity_vocab.get_word(wdx)
            sanitised_phrase = [Gigaword.sanitiseWord(tok) for tok in entity_phrase.split(" ")]
            giga_indices = np.array([self.lookup_giga[tok] if tok in self.lookup_giga else self.lookup_giga["<unk>"] for tok in sanitised_phrase],dtype=np.int32)
            phrase_embedding = self.giga_w2v_embedding[giga_indices,:].T
            summed_embedding = np.sum(phrase_embedding,axis=1)
            summed_embedding_norm = np.linalg.norm(summed_embedding)
            if summed_embedding_norm > 0:
                summed_embedding /= summed_embedding_norm
            self.w2v_entity_embeddings.append(summed_embedding)
        self.w2v_entity_embeddings = np.array(self.w2v_entity_embeddings)

    def cat_embedding_feat(self,model,entity_ids,cat_entity_ids,is_pool_entities=False,k=None):
        entity_embeddings = Gigaword.norm(self.w2v_entity_embeddings[entity_ids,:])
        cat_entity_embeddings = Gigaword.norm(self.w2v_entity_embeddings[cat_entity_ids,:])
        cos_sims = np.dot(entity_embeddings,cat_entity_embeddings.T)

        if is_pool_entities:
            # TODO: assumes at least some (or none) of entity ids are in category -> can be faster if we assume they all are, but for generality...
            mean_scores,min_scores,max_scores=[],[],[]
            for edx,ent_id in enumerate(entity_ids):
                unique_cos_sims = cos_sims[edx,cat_entity_ids!=ent_id]
                unique_cos_sims.sort()
                cos_sims_knn = unique_cos_sims if k is None else unique_cos_sims[-k:]
                mean_scores.append(np.mean(cos_sims_knn))
                min_scores.append(np.min(cos_sims_knn))
                max_scores.append(np.max(cos_sims_knn))
            return np.array(mean_scores),np.array(min_scores),np.array(max_scores)
        else:
            cos_sims.sort(axis=-1)
            cos_sims_knn = cos_sims if k is None else cos_sims[:,-k:]
            mean_scores = np.mean(cos_sims_knn,axis=1)
            min_scores = np.min(cos_sims_knn,axis=1)
            max_scores = np.max(cos_sims_knn,axis=1)
            return mean_scores,min_scores,max_scores

    def compute_embedding_feats(self,model,entity_ids,all_cats,is_pool_entities=False):
        ent_ids = np.array(entity_ids,dtype=np.int32)
        n_ents = ent_ids.shape[0]
        ent_range = np.arange(n_ents)
        cat_embedding_feats = []
        for cat in all_cats:
            cat_ent_ids = np.array(model.pools_entities[cat],dtype=np.int32)
            mean_cat_scores,min_cat_scores,max_cat_scores = self.cat_embedding_feat(model,ent_ids,cat_ent_ids,is_pool_entities)
            cat_embedding_feats.append(np.concatenate((mean_cat_scores.reshape((n_ents,1)),min_cat_scores.reshape((n_ents,1)),max_cat_scores.reshape((n_ents,1))),axis=1))
        embedding_feat_mat = np.concatenate(cat_embedding_feats,axis=1)

        embedding_mean = np.concatenate([feats[:,:1] for feats in cat_embedding_feats],axis=1)
        embedding_mean_argmax = np.argmax(embedding_mean,axis=1)
        embedding_mean_feat_mat = np.zeros(embedding_mean.shape)
        embedding_mean_feat_mat[ent_range,embedding_mean_argmax] = embedding_mean[ent_range,embedding_mean_argmax]

        embedding_max = np.concatenate([feats[:,-1:] for feats in cat_embedding_feats],axis=1)
        embedding_max_argmax = np.argmax(embedding_max,axis=1)
        embedding_max_feat_mat = np.zeros(embedding_max.shape)
        embedding_max_feat_mat[ent_range,embedding_max_argmax] = embedding_max[ent_range,embedding_max_argmax]

        return np.concatenate((embedding_feat_mat,embedding_mean_feat_mat,embedding_max_feat_mat),axis=1)

if __name__ == '__main__':
    entity_vocab = Vocabulary.from_file(sys.argv[1])
    gigaW2vEmbed, lookupGiga = Gigaword.load_pretrained_dep_embeddings(sys.argv[2])
    filtered_w2v_file = open(sys.argv[3],'w')
    vocab_set = set()
    for wdx in range(entity_vocab.size()):
        entity_phrase = entity_vocab.get_word(wdx)
        for tok in entity_phrase.split(' '):
            sanitised_tok = Gigaword.sanitiseWord(tok)
            if sanitised_tok not in lookupGiga or sanitised_tok in vocab_set:
                continue
            vocab_set.add(sanitised_tok)
            tok_index = lookupGiga[sanitised_tok]
            tok_embedding = gigaW2vEmbed[tok_index]
            filtered_w2v_file.write(sanitised_tok)
            for coord in tok_embedding:
                filtered_w2v_file.write(' '+str(coord))
            filtered_w2v_file.write('\n')
    filtered_w2v_file.close()
