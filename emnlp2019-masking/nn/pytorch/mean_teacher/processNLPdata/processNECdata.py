from .w2v import Gigaword
from .vocabulary import Vocabulary
import numpy as np
from .datautils import Datautils

def tokenizer(words_str):
    words = words_str.split(" ")
    [Gigaword.sanitiseWord(w) for w in words]

def load_emboot_data(entity_vocab_file, context_vocab_file, data_file):

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

    w2vfile = "/Users/ajaynagesh/Research/code/research/clint/data/vectors.goldbergdeps.txt" # "/work/ajaynagesh/clara_expts/research-ontonotes/clint/data/vectors.goldbergdeps.txt" #
    dataset = "conll" # "onto #

    if dataset == "conll":
        ##### CoNLL dataset
        categories = sorted(list(['PER', 'ORG', 'LOC', 'MISC']))
        entity_vocab_file = "./data-local/nec/conll/entity_vocabulary.emboot.filtered.txt"
        context_vocab_file = "./data-local/nec/conll/pattern_vocabulary_emboot.filtered.txt"
        data_file = "./data-local/nec/conll/training_data_with_labels_emboot.filtered.txt"
        train_data_np_file = "./data-local/nec/conll/np_conll_nec.npy"
        train_data_np_lbls_file = "./data-local/nec/conll/np_conll_nec_labels.npy"

    else:
        #### Ontonotes dataset
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
        entity_vocab_file = "./data-local/nec/onto/entity_vocabulary.emboot.filtered.txt"
        context_vocab_file = "./data-local/nec/onto/pattern_vocabulary_emboot.filtered.txt"
        data_file = "./data-local/nec/onto/training_data_with_labels_emboot.filtered.txt"
        train_data_np_file = "./data-local/nec/onto/np_onto_nec.npy"
        train_data_np_lbl_file = "./data-local/nec/onto/np_onto_nec_labels.npy"

    print("Loading the dataset .. " )
    mentions_all, contexts_all, labels_all, entity_vocab, context_vocab = load_emboot_data(entity_vocab_file, context_vocab_file, data_file)
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
    features_all = np.hstack([mention_embed_all_np, contexts_embed_all])

    np.save(train_data_np_file, features_all)
    np.save(train_data_np_lbls_file, label_ids_all)