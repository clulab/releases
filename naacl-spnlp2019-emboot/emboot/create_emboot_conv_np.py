from w2v import Gigaword
from datautils import Datautils
from vocabulary import Vocabulary
import numpy as np

entity_vocab_file = "./data/entity_vocabulary.emboot.filtered.txt"
context_vocab_file = "./data/pattern_vocabulary_emboot.filtered.txt"
data_file = "./data/training_data_with_labels_emboot.filtered.txt"
categories = sorted(list(['PER', 'ORG', 'LOC', 'MISC']))

numpy_target_train_file = "./data/train_targets_emboot_conv.npy"
# numpy_feature_train_file = "./data/train_features_emboot_conv.npy"
numpy_feature_train_file = "./data/train_features_emboot_conv_225.300.npy"

max_word_length = 9 ### max num of tokens in entity / a given pattern
max_sent_length = 24 ### max num of patterns

# entity_vocab_file = "./data-ontonotes/entity_vocabulary.emboot.filtered.txt"
# context_vocab_file = "./data-ontonotes/pattern_vocabulary_emboot.filtered.txt"
# data_file = "./data-ontonotes/training_data_with_labels_emboot.filtered.txt"
# numpy_feature_train_file = "./data-ontonotes/train_features_emboot_conv.npy"
# numpy_feature_test_file = "./data-ontonotes/test_features_emboot_conv.npy"
# numpy_target_train_file = "./data-ontonotes/train_targets_emboot_conv.npy"
# numpy_target_test_file = "./data-ontonotes/test_targets_emboot_conv.npy"
# categories = sorted(list([ "EVENT",
#                            "FAC",
#                            "GPE",
#                            "LANGUAGE",
#                            "LAW",
#                            "LOC",
#                            "NORP",
#                            "ORG",
#                            "PERSON",
#                            "PRODUCT",
#                            "WORK_OF_ART"]))

w2vfile = "../data/vectors.goldbergdeps.txt"

def load_emboot_data():

    print('reading vocabularies ...')
    entity_vocab = Vocabulary.from_file(entity_vocab_file)
    context_vocab = Vocabulary.from_file(context_vocab_file)

    m, c, l = Datautils.read_data(data_file, entity_vocab, context_vocab)
    return m, c, l, entity_vocab, context_vocab

def construct_patterns_embed_maxlength(patternIds, context_vocab, gigaW2vEmbed, lookupGiga):

    max_pattern = max([(len(context_vocab.get_word(i).split(" ")), context_vocab.get_word(i)) for  i in patternIds], key=lambda item:item[0])
    pat = max_pattern[1]

    words_in_pat = [Gigaword.sanitiseWord(w) if w != "@ENTITY" else "<entity>" for w in pat.split(" ")]
    embedIds_in_pat = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_pat]

    pad_length = max_word_length - len(words_in_pat)
    for i in range(pad_length):
        embedIds_in_pat.append(lookupGiga["<pad>"])

    pat_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_pat])

    return pat_embedding

def construct_patterns_embed(patternIds, context_vocab, gigaW2vEmbed, lookupGiga):

    pat_embedding_list = list()
    for pat in patternIds:
        words_in_pat = [Gigaword.sanitiseWord(w) if w != "@ENTITY" else "<entity>" for w in context_vocab.get_word(pat).split(" ")]
        embedIds_in_pat = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_pat]

        pad_length = max_word_length - len(words_in_pat)
        for i in range(pad_length):
            embedIds_in_pat.append(lookupGiga["<pad>"])

        pat_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_pat])

        pat_embedding_list.append(pat_embedding)

    sent_pad_length = max_sent_length - len(pat_embedding_list)
    for i in range(sent_pad_length):
        embedIds_in_pat = [lookupGiga["<pad>"]]*max_word_length
        pat_embedding_list.append(Gigaword.norm(gigaW2vEmbed[embedIds_in_pat]))

    pat_embedding_list_np = np.concatenate(pat_embedding_list, axis=0)
    return pat_embedding_list_np

def construct_entity_embed(entityId, entity_vocab, gigaW2vEmbed, lookupGiga):

    words_in_ent = [Gigaword.sanitiseWord(w) for w in entity_vocab.get_word(entityId).split(" ")]
    embedIds_in_ent = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_ent]

    pad_length = max_word_length - len(words_in_ent)
    for i in range(pad_length):
        embedIds_in_ent.append(lookupGiga["<pad>"])

    ent_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_ent])
    return ent_embedding


if __name__ == '__main__':

    print("Loading the emboot data")
    mentions_all, contexts_all, labels_all, entity_vocab, context_vocab = load_emboot_data()
    label_ids_all = np.array([categories.index(l) for l in labels_all])

    print("Loading the gigaword embeddings ...")
    gigaW2vEmbed, lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)

    features_embed_all = list()
    for idx, entityId in enumerate(mentions_all):
        contextIds = contexts_all[idx]

        mention_embed = construct_entity_embed(entityId, entity_vocab, gigaW2vEmbed, lookupGiga)
        # print("------------")
        # print(mention_embed.shape)
        contexts_embed = construct_patterns_embed(contextIds, context_vocab, gigaW2vEmbed, lookupGiga)
        # contexts_embed = construct_patterns_embed_maxlength(contextIds, context_vocab, gigaW2vEmbed, lookupGiga)
        # print(contexts_embed.shape)
        features_embed = np.concatenate([mention_embed, contexts_embed], axis=0)
        # print (features_embed.shape)
        # print("------------")
        features_embed_all.append(features_embed)

    features_embed_all_np = np.array(features_embed_all)
    print (features_embed_all_np.shape)
    ############## Conll =======================
    ## full dataset = 13196 x 25 x 9 x 300

    np.save(numpy_feature_train_file, features_embed_all_np)
    np.save(numpy_target_train_file, label_ids_all)