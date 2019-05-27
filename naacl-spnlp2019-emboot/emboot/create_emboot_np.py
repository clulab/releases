from w2v import Gigaword
from datautils import Datautils
from vocabulary import Vocabulary
import numpy as np

# entity_vocab_file = "./data/entity_vocabulary.emboot.filtered.txt"
# context_vocab_file = "./data/pattern_vocabulary_emboot.filtered.txt"
# data_file = "./data/training_data_with_labels_emboot.filtered.txt"
# numpy_feature_train_file = "./data/train_features_emboot.npy"
# numpy_feature_test_file = "./data/test_features_emboot.npy"
# numpy_target_train_file = "./data/train_targets_emboot.npy"
# numpy_target_test_file = "./data/test_targets_emboot.npy"
# categories = sorted(list(['PER', 'ORG', 'LOC', 'MISC']))

entity_vocab_file = "./data-ontonotes/entity_vocabulary.emboot.filtered.txt"
context_vocab_file = "./data-ontonotes/pattern_vocabulary_emboot.filtered.txt"
data_file = "./data-ontonotes/training_data_with_labels_emboot.filtered.txt"
numpy_feature_train_file = "./data-ontonotes/train_features_emboot.npy"
numpy_feature_test_file = "./data-ontonotes/test_features_emboot.npy"
numpy_target_train_file = "./data-ontonotes/train_targets_emboot.npy"
numpy_target_test_file = "./data-ontonotes/test_targets_emboot.npy"
categories = sorted(list([ "EVENT",
                           "FAC",
                           "GPE",
                           "LANGUAGE",
                           "LAW",
                           "LOC",
                           "NORP",
                           "ORG",
                           "PERSON",
                           "PRODUCT",
                           "WORK_OF_ART"]))

w2vfile = "../data/vectors.goldbergdeps.txt"

def load_emboot_data():

    print('reading vocabularies ...')
    entity_vocab = Vocabulary.from_file(entity_vocab_file)
    context_vocab = Vocabulary.from_file(context_vocab_file)

    m, c, l = Datautils.read_data(data_file, entity_vocab, context_vocab)
    return m, c, l, entity_vocab, context_vocab

def construct_patterns_embed(patternIds, context_vocab, gigaW2vEmbed, lookupGiga):

    pat_embedding_list = list()
    for pat in patternIds:
        words_in_pat = [Gigaword.sanitiseWord(w) for w in context_vocab.get_word(pat).split(" ") if w != "@ENTITY"]
        embedIds_in_pat = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_pat]
        if len(embedIds_in_pat) > 1:
            pat_embedding = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_pat]), axis=0) # avg of the embeddings for each word in the pattern
        else:
            pat_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_pat])

        pat_embedding_list.append(pat_embedding)

    pat_embedding_list_np = np.array([pe for pe in pat_embedding_list])
    return np.average(pat_embedding_list_np, axis=0)

def construct_entity_embed(entityId, entity_vocab, gigaW2vEmbed, lookupGiga):

    words_in_ent = [Gigaword.sanitiseWord(w) for w in entity_vocab.get_word(entityId).split(" ")]
    embedIds_in_ent = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_ent]
    if len(embedIds_in_ent) > 1:
        ent_embedding = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_ent]), axis=0) # avg of the embeddings for each word in the pattern
    else:
        ent_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_ent])

    return ent_embedding


if __name__ == '__main__':

    print("Loading the emboot data")
    mentions_all, contexts_all, labels_all, entity_vocab, context_vocab = load_emboot_data()
    label_ids_all = np.array([categories.index(l) for l in labels_all])

    print("Loading the gigaword embeddings ...")
    gigaW2vEmbed, lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)

    mention_embed_all = list()
    contexts_embed_all = list()
    for idx, entityId in enumerate(mentions_all):
        contextIds = contexts_all[idx]

        mention_embed = construct_entity_embed(entityId, entity_vocab, gigaW2vEmbed, lookupGiga)
        mention_embed_all.append(mention_embed)

        contexts_embed = construct_patterns_embed(contextIds, context_vocab, gigaW2vEmbed, lookupGiga)
        contexts_embed_all.append(contexts_embed)


    mention_embed_all_np = np.array([me for me in mention_embed_all])
    contexts_embed_all_np = np.array([ce for ce in contexts_embed_all])

    ############## Conll =======================
    ## full dataset = 13196 x 600
    features_all = np.hstack([mention_embed_all_np, contexts_embed_all])

    ## train = 13000 x 600, test = 196 x 600
    train_features = features_all[:13000]
    test_features = features_all[13000:]

    train_targets = label_ids_all[:13000]
    test_targets = label_ids_all[13000:]


    ############## Ontonotes =======================
    ## reduced from 67229 --> 67000
    # features_all = features_all[:67000]
    # label_ids_all = label_ids_all[:67000]

    ## train --> 53600, test --> 13400
    ## ...
    ## ...

    np.save(numpy_feature_train_file, train_features)
    np.save(numpy_feature_test_file, test_features)
    np.save(numpy_target_train_file, train_targets)
    np.save(numpy_target_test_file, test_targets)