import sys
import numpy as np
from feats import PMIFeats
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class ClassBalancedMostConfident:
    def __init__(self, logistic_preds):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
    #

    def sample(self, n_subset):
        n_subset_class = n_subset // self.n_categories
        samples = []
        samples_per_class = self.n_categories*[0]
        sample_set = set()
        sorted_entropies = np.argsort(self.entropies)
        for sdx in sorted_entropies:
            pred_cat = self.class_preds[sdx]
            if samples_per_class[pred_cat] != n_subset_class:
                sample_set.add(sdx)
                samples.append(sdx)
                samples_per_class[pred_cat]+=1
            #
        # may not have the desired entities-per-class -> go through and just choose the most informative ones
        num_leftover = np.sum([n_subset_class-num_ents for num_ents in samples_per_class])
        for sdx in sorted_entropies:
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#

class ClassBalancedLeastConfident:
    def __init__(self, logistic_preds):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
    #

    def sample(self, n_subset):
        n_subset_class = n_subset // self.n_categories
        samples = []
        samples_per_class = self.n_categories*[0]
        sample_set = set()
        sorted_entropies = np.argsort(self.entropies)
        for sdx in sorted_entropies[::-1]:
            pred_cat = self.class_preds[sdx]
            if samples_per_class[pred_cat] != n_subset_class:
                sample_set.add(sdx)
                samples.append(sdx)
                samples_per_class[pred_cat]+=1
            #
        # may not have the desired entities-per-class -> go through and just choose the most informative ones
        num_leftover = np.sum([n_subset_class-num_ents for num_ents in samples_per_class])
        for sdx in sorted_entropies[::-1]:
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#

class ClassBalancedRandom:
    def __init__(self, logistic_preds):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
    #

    def sample(self, n_subset):
        n_subset_class = n_subset // self.n_categories
        # group by predicted category
        class_groups = []
        for cdx in range(self.n_categories):
            class_groups.append(np.argwhere(self.class_preds==cdx)[:,0])

        # randomly sample from within each category
        samples = []
        samples_per_class = self.n_categories*[0]
        sample_set = set()
        for cdx in range(self.n_categories):
            class_group = class_groups[cdx]
            n_class_samples = n_subset_class if class_group.shape[0] > n_subset_class else class_group.shape[0]
            class_subset = np.random.choice(class_group,size=n_class_samples,replace=False)
            samples.extend(class_subset)
            samples_per_class[cdx] = class_subset.shape[0]
            for sdx in class_subset:
                sample_set.add(sdx)
        # may not have the desired entities-per-class -> go through and just choose random samples
        num_leftover = np.sum([n_subset_class-num_ents for num_ents in samples_per_class])
        for sdx in range(self.n_samples):
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#

class ClassBalancedMostLeastConfident:
    def __init__(self, logistic_preds):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
    #

    def sample(self, n_subset):
        n_subset_class = n_subset // self.n_categories
        # group by predicted category
        class_groups = []
        for cdx in range(self.n_categories):
            class_groups.append(np.argwhere(self.class_preds==cdx)[:,0])

        # for each category take the top half and bottom half
        samples = []
        samples_per_class = self.n_categories*[0]
        sample_set = set()
        for cdx in range(self.n_categories):
            class_group = class_groups[cdx]
            class_entropies = self.entropies[class_group]
            sorted_class_entropies = np.argsort(class_entropies)
            n_class_samples = n_subset_class if class_group.shape[0] > n_subset_class else class_group.shape[0]
            balanced_samples = np.concatenate((sorted_class_entropies[-(n_class_samples//2):],sorted_class_entropies[0:(n_class_samples//2)]),axis=0)
            samples.extend(balanced_samples)
            samples_per_class[cdx] = balanced_samples.shape[0]
            for sdx in balanced_samples:
                sample_set.add(sdx)
        # may not have the desired entities-per-class -> go through and just choose random samples
        num_leftover = np.sum([n_subset_class-num_ents for num_ents in samples_per_class])
        for sdx in range(self.n_samples):
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#

class ClusterSampler:
    def __init__(self, logistic_preds, embedding, words):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
        self.embedding = embedding
        self.words = words
    #

    def sample(self, n_subset, init_with_uncertain=False):
        n_clusters = 40
        # run k-means, with high k, see what we get...
        if init_with_uncertain:
            samples_per_class = self.n_categories*[0]
            n_cluster_class = n_clusters // self.n_categories
            init_samples = []
            sorted_entropies = np.argsort(self.entropies)
            for sdx in sorted_entropies[::-1]:
                pred_cat = self.class_preds[sdx]
                if samples_per_class[pred_cat] != n_cluster_class:
                    print('uncertain seed:',self.words[sdx])
                    init_samples.append(sdx)
                    samples_per_class[pred_cat]+=1
                #
            #
            init_samples = np.array(init_samples,dtype=np.int32)
            kmeans = KMeans(n_clusters=n_clusters,verbose=2,init=self.embedding[init_samples,:],n_init=1).fit(self.embedding)
        else:
            kmeans = KMeans(n_clusters=n_clusters,verbose=2,init='k-means++',n_init=1).fit(self.embedding)
        cluster_list = list(set(kmeans.labels_))

        cluster_filename = open('cluster.txt','w')
        for cluster in cluster_list:
            cluster_inds = np.argwhere(kmeans.labels_==cluster)[:,0]
            cluster_words = [self.words[cind] for cind in cluster_inds]
            cluster_filename.write('cluster['+str(cluster)+']: '+str(cluster_inds.shape[0])+'\n')
            for wdx,word in enumerate(cluster_words):
                cluster_filename.write('   ['+str(wdx)+']: '+word+'\n')
        cluster_filename.close()

        samples = []
        n_subset_cluster = n_subset // len(cluster_list)
        for cluster in cluster_list:
            cluster_inds = np.argwhere(kmeans.labels_==cluster)[:,0]
            cluster_embedding = self.embedding[cluster_inds,:]
            cluster_center = np.mean(cluster_embedding,axis=0)
            cluster_center = cluster_center / np.linalg.norm(cluster_center)
            cluster_scores = cluster_embedding.dot(cluster_center)
            sorted_scores = np.argsort(cluster_scores)
            samples.extend(cluster_inds[sorted_scores[-n_subset_cluster:]])

        sample_set = set(samples)
        samples = list(sample_set)

        # may not have the desired entities-per-class -> go through and just choose random samples
        num_leftover = n_subset-len(samples)
        print('num leftover:',num_leftover)
        for sdx in range(self.n_samples):
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#

class ClusterDensitySampler:
    def __init__(self, logistic_preds, embedding, words):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
        self.embedding = embedding
        self.words = words

        self.density_estimate_bandwidth = 0.75
        self.band_sqd = self.density_estimate_bandwidth*self.density_estimate_bandwidth
    #

    def sample(self, n_subset, init_with_uncertain=False):
        n_clusters = 40
        # run k-means, with high k, see what we get...
        if init_with_uncertain:
            samples_per_class = self.n_categories*[0]
            n_cluster_class = n_clusters // self.n_categories
            init_samples = []
            sorted_entropies = np.argsort(self.entropies)
            for sdx in sorted_entropies[::-1]:
                pred_cat = self.class_preds[sdx]
                if samples_per_class[pred_cat] != n_cluster_class:
                    print('uncertain seed:',self.words[sdx])
                    init_samples.append(sdx)
                    samples_per_class[pred_cat]+=1
                #
            #
            init_samples = np.array(init_samples,dtype=np.int32)
            kmeans = KMeans(n_clusters=n_clusters,verbose=2,init=self.embedding[init_samples,:],n_init=1).fit(self.embedding)
        else:
            kmeans = KMeans(n_clusters=n_clusters,verbose=2,init='k-means++',n_init=1).fit(self.embedding)
        cluster_list = list(set(kmeans.labels_))

        '''
        cluster_filename = open('cluster.txt','w')
        for cluster in cluster_list:
            cluster_inds = np.argwhere(kmeans.labels_==cluster)[:,0]
            cluster_words = [self.words[cind] for cind in cluster_inds]
            cluster_filename.write('cluster['+str(cluster)+']: '+str(cluster_inds.shape[0])+'\n')
            for wdx,word in enumerate(cluster_words):
                cluster_filename.write('   ['+str(wdx)+']: '+word+'\n')
        cluster_filename.close()
        '''

        # sample each cluster proportional to its number of samples
        all_cluster_inds = []
        n_cluster_cat_freqs = self.n_categories*[0]
        n_subset_cluster = n_subset // len(cluster_list)
        for cluster in cluster_list:
            cluster_inds = np.argwhere(kmeans.labels_==cluster)[:,0]
            all_cluster_inds.append(cluster_inds)
        #
        #cluster_filename = open('cluster.txt','w')
        samples = []
        for cdx,cluster_inds in enumerate(all_cluster_inds):
            n_cluster_samples = int(np.floor(n_subset*(cluster_inds.shape[0]/self.n_samples)))
            if n_cluster_samples == 0 or n_cluster_samples >= cluster_inds.shape[0] or cluster_inds.shape[0] < 15:
                continue
            # density estimation to find most representative sample
            cluster_embedding = self.embedding[cluster_inds,:]
            sqd_dists = euclidean_distances(cluster_embedding,cluster_embedding,squared=True)
            kde = np.sum(np.exp(-sqd_dists/self.band_sqd),axis=1) / cluster_inds.shape[0]
            '''
            sorted_kde_inds = np.argsort(kde)
            top_kde = cluster_inds[sorted_kde_inds[-10:]]
            kde_maximum = top_kde[np.random.randint(10)]
            '''
            kde_maximum = cluster_inds[np.argmax(kde)]
            cluster_center = self.embedding[kde_maximum,:]
            cluster_scores = cluster_embedding.dot(cluster_center)
            sorted_scores = np.argsort(cluster_scores)
            samples.extend(cluster_inds[sorted_scores[-n_cluster_samples:]])

            # debug
            '''
            cluster_words = [self.words[cind] for cind in cluster_inds]
            cluster_filename.write('cluster['+str(cluster)+']: '+str(cluster_inds.shape[0])+' : '+str(n_cluster_samples)+' center: '+self.words[kde_maximum]+'\n')
            for wdx,word in enumerate(cluster_words):
                cluster_filename.write('   ['+str(wdx)+']: '+word+'\n')
            '''

        sample_set = set(samples)
        samples = list(sample_set)

        # may not have the desired entities-per-class -> go through and just choose random samples
        num_leftover = n_subset-len(samples)
        print('num leftover:',num_leftover)
        for sdx in range(self.n_samples):
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#
class ClassBalancedClusterSampler:
    def __init__(self, logistic_preds, embedding, representative_entities, rep_labels, words):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
        self.embedding = embedding
        self.words = words

        self.representative_entities = representative_entities
        self.rep_labels = rep_labels
        self.categories = list(set(self.rep_labels))
        self.cat_map = dict([(cat,cdx) for cdx,cat in enumerate(self.categories)])
        self.label_idxs = np.array([self.cat_map[cat] for cat in self.rep_labels],dtype=np.int32)
        self.embedding_class_preds = self.compute_embedding_class_predictions()
    #

    def compute_embedding_class_predictions(self,k=5):
        # compute distance from embedding to representative entities
        rep_scores = self.embedding.dot(self.representative_entities.T)
        # sort scores, use k-nn classifier
        sorted_rep_inds = np.argsort(rep_scores,axis=1)
        top_reps = sorted_rep_inds[:,-k:]
        top_labels = self.label_idxs[top_reps]
        emb_preds = []
        for top_label in top_labels:
            emb_preds.append(np.bincount(top_label).argmax())
        return np.array(emb_preds,dtype=np.int32)

    def sample(self, n_subset, init_with_uncertain=False):
        n_subset_class = n_subset // self.n_categories
        n_clusters = 60

        # run k-means, with high k, see what we get...
        if init_with_uncertain:
            samples_per_class = self.n_categories*[0]
            n_cluster_class = n_clusters // self.n_categories
            init_samples = []
            sorted_entropies = np.argsort(self.entropies)
            for sdx in sorted_entropies[::-1]:
                pred_cat = self.embedding_class_preds[sdx]
                if samples_per_class[pred_cat] != n_cluster_class:
                    print('certain seed:',self.words[sdx])
                    init_samples.append(sdx)
                    samples_per_class[pred_cat]+=1
                #
            #
            init_samples = np.array(init_samples,dtype=np.int32)
            kmeans = KMeans(n_clusters=n_clusters,verbose=2,init=self.embedding[init_samples,:],n_init=1).fit(self.embedding)
        else:
            kmeans = KMeans(n_clusters=n_clusters,verbose=2,init='k-means++',n_init=1).fit(self.embedding)
        cluster_list = list(set(kmeans.labels_))

        '''
        cluster_filename = open('cluster.txt','w')
        for cluster in cluster_list:
            cluster_inds = np.argwhere(kmeans.labels_==cluster)[:,0]
            cluster_words = [self.words[cind] for cind in cluster_inds]
            cluster_filename.write('cluster['+str(cluster)+']: '+str(cluster_inds.shape[0])+'\n')
            for wdx,word in enumerate(cluster_words):
                cluster_filename.write('   ['+str(wdx)+']: '+word+'\n')
        cluster_filename.close()
        '''

        all_cluster_inds = []
        all_predicted_cats = []
        n_cluster_cat_freqs = self.n_categories*[0]
        n_subset_cluster = n_subset // len(cluster_list)
        for cluster in cluster_list:
            cluster_inds = np.argwhere(kmeans.labels_==cluster)[:,0]
            cluster_class_preds = self.embedding_class_preds[cluster_inds]
            cluster_class_counts = np.bincount(cluster_class_preds)
            print('cluster class counts:',cluster_class_counts)
            pred_cat = np.argmax(cluster_class_counts)
            class_cluster_inds = cluster_inds[np.argwhere(cluster_class_preds==pred_cat)[:,0]]
            n_cluster_cat_freqs[pred_cat] += class_cluster_inds.shape[0]
            all_cluster_inds.append(class_cluster_inds)
            all_predicted_cats.append(pred_cat)
        #

        samples = []
        for cluster,class_cluster_inds,predicted_cat in zip(cluster_list,all_cluster_inds,all_predicted_cats):
            cluster_embedding = self.embedding[class_cluster_inds,:]
            cluster_center = np.mean(cluster_embedding,axis=0)
            cluster_center = cluster_center / np.linalg.norm(cluster_center)
            cluster_scores = cluster_embedding.dot(cluster_center)
            sorted_scores = np.argsort(cluster_scores)
            n_cluster_sample = int(np.ceil(n_subset_class*(class_cluster_inds.shape[0]/n_cluster_cat_freqs[predicted_cat])))
            if n_cluster_sample > class_cluster_inds.shape[0]:
                n_cluster_sample = class_cluster_inds.shape[0]
            print('for class',predicted_cat,'with',n_cluster_cat_freqs[predicted_cat],'adding',n_cluster_sample,'from',class_cluster_inds.shape[0])
            samples.extend(class_cluster_inds[sorted_scores[-n_cluster_sample:]])

        sample_set = set(samples)
        samples = list(sample_set)

        # may not have the desired entities-per-class -> go through and just choose random samples
        num_leftover = n_subset-len(samples)
        print('num leftover:',num_leftover)
        for sdx in range(self.n_samples):
            if num_leftover <= 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#

class RepresentativeUncertainSampler:
    def __init__(self, logistic_preds, sparse_npmi, candidate_entities, reference_entities, embedding, words):
        self.logistic_preds = logistic_preds
        self.n_samples = self.logistic_preds.shape[0]
        self.n_categories = self.logistic_preds.shape[1]
        self.entropies = -np.sum((self.logistic_preds*np.log2(self.logistic_preds+1e-16)), axis=1)
        self.class_preds = np.argmax(self.logistic_preds,axis=1)
        self.sparse_npmi = sparse_npmi
        self.candidate_entities = candidate_entities
        self.reference_entities = reference_entities
        self.embedding = embedding
        self.words = words
    #

    def sample(self, n_subset):
        # for each reference entity find its nearest candidate entity with respect to NPMI overlap
        representative_candidates = []
        for ref_entity in self.reference_entities:
            ref_patterns,ref_npmi = self.sparse_npmi[ref_entity]
            candidate_npmi_overlap = np.zeros(self.candidate_entities.shape[0])
            for edx,candidate_entity in enumerate(self.candidate_entities):
                cand_patterns,cand_npmi = self.sparse_npmi[candidate_entity]
                common_pats = np.intersect1d(ref_patterns,cand_patterns)
                if common_pats.shape[0]==0:
                    continue
                for common_pat in common_pats:
                    candidate_npmi_overlap[edx] += ref_npmi[np.argwhere(ref_patterns==common_pat)[0,0]]*cand_npmi[np.argwhere(cand_patterns==common_pat)[0,0]]
            candidate_npmi_overlap = candidate_npmi_overlap*self.entropies
            sorted_overlap = np.argsort(candidate_npmi_overlap)
            representative_candidates.append(sorted_overlap[-(np.random.randint(1,10))])
            '''
            print('reference entity',self.words[ref_entity])
            for cand_ind in sorted_overlap[-10:]:
                print('candidate entity:',self.words[self.candidate_entities[cand_ind]],':',candidate_npmi_overlap[cand_ind])
            '''

        # for each reference candidate, get its knn in the emboot embeddings
        representative_candidates = np.array(list(set(representative_candidates)),dtype=np.int32)
        n_subset_candidate = n_subset // representative_candidates.shape[0]
        cos_sims = np.dot(self.embedding[representative_candidates,:],self.embedding.T)
        sorted_sims = np.argsort(cos_sims,axis=1)
        samples = []
        for cdx,candidate in enumerate(representative_candidates):
            k_expanded = sorted_sims[cdx,-n_subset_candidate:]
            samples.extend(list(k_expanded))
        sample_set = set(samples)
        samples = list(sample_set)

        # may not have the desired entities-per-class -> go through and just choose random samples
        num_leftover = n_subset-len(samples)
        print('num leftover:',num_leftover)
        for sdx in range(self.n_samples):
            if num_leftover == 0:
                break
            if sdx in sample_set:
                continue
            sample_set.add(sdx)
            samples.append(sdx)
            num_leftover-=1
        #
        return samples
    #
#
