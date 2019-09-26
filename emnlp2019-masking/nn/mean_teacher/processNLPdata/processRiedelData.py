from w2v import Gigaword
import numpy as np

def parse_data(train_data_path):
    with open(train_data_path, 'r') as train_fh:
        train_data = []
        for line in train_fh:
            fields = line.rstrip("\n").split("\t")
            eid_1 = fields[0]
            eid_2 = fields[1]
            entity_1 = fields[2].replace('_', ' ') ## NOTE: Replacing the '_' which combines multi-token NEs with space (to help in averaging)
            entity_2 = fields[3].replace('_', ' ') ## NOTE: Replacing the '_' which combines multi-token NEs with space (to help in averaging)
            relation = fields[4]
            # NOTE: replacing the ####END#### token at the end of a sentence
            #       replacing the '_' which combines multi-token NEs with space
            #       replacing entity_1 and entity_2 also in the sentence
            sentence = fields[5]\
                            .replace(' ###END###','')\
                            .replace('_', ' ')\
                            .replace(entity_1, "<entityOne>")\
                            .replace(entity_2, "<entityTwo>")
            train_data.append((eid_1, eid_2, entity_1, entity_2, relation, sentence))

    return train_data

def create_data_for_pytorch(raw_train_data, relation_dict, lookupGiga, gigaW2vEmbed):

    relation_labels = []
    embedding_datums = []
    idx = 0
    for datum in raw_train_data:

        idx += 1
        if idx % 50000 == 0:
            print ("Completed " + str(idx) + "/" + str(len(raw_train_data)) + " number of examples .. ")

        ent1_toks = [Gigaword.sanitiseWord(w) for w in datum[2].split(" ")]

        ent2_toks = [Gigaword.sanitiseWord(w) for w in datum[3].split(" ")]
        if datum[4] in relation_dict:
            relation_id = relation_dict[datum[4]]
        else:
            print("Relation missing in train .. " + str(datum[4]))
            relation_id = relation_dict["NA"]

        sentence_toks = [Gigaword.sanitiseWord(w) for w in datum[5].split(" ") if Gigaword.sanitiseWord(w) != "" ]

        embedIds_in_ent1 = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in ent1_toks]
        embedIds_in_ent2 = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in ent2_toks]
        embedIds_in_sentence = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in sentence_toks]

        if len(embedIds_in_ent1) > 1:
            avg_embed_ent1 = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_ent1]), axis=0)
        else:
            avg_embed_ent1 = Gigaword.norm(gigaW2vEmbed[embedIds_in_ent1])

        if len(embedIds_in_ent2) > 1:
            avg_embed_ent2 = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_ent2]), axis=0)
        else:
            avg_embed_ent2 = Gigaword.norm(gigaW2vEmbed[embedIds_in_ent2])

        if len(embedIds_in_sentence) > 1:
            avg_embed_sent = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_sentence]), axis=0)
        else:
            avg_embed_sent = Gigaword.norm(gigaW2vEmbed[embedIds_in_sentence])

        embedding_datums.append(np.hstack([avg_embed_ent1, avg_embed_ent2, avg_embed_sent]))
        relation_labels.append(relation_id)

    embedding_dataset = np.array([embedding_datum for embedding_datum in embedding_datums])

    return (embedding_dataset, relation_labels)

if __name__ == '__main__':

    #### RIEDEL DATASET
    # relext_data_path = "./data-local/riedel10/raw"
    # destination_dir_path = "./data-local/riedel10"

    #### GIDS DATASET
    relext_data_path = "./data-local/gids/raw"
    destination_dir_path = "./data-local/gids"


    w2vfile = "/work/ajaynagesh/clara_expts/research-ontonotes/clint/data/vectors.goldbergdeps.txt" #"/Users/ajaynagesh/Research/code/research/clint/data/vectors.goldbergdeps.txt"

    train_data_path = relext_data_path + "/train.txt" #"/Users/ajaynagesh/Research/LadderNetworks/relext_data/Riedel2010dataset/RE/train.txt"
    test_data_path = relext_data_path + "/test.txt"   #/Users/ajaynagesh/Research/LadderNetworks/relext_data/Riedel2010dataset/RE/test.txt"


    train_data_np_file = destination_dir_path + "/train/np_relext.npy" # "/Users/ajaynagesh/Research/LadderNetworks/mean-teacher/pytorch/processNLPdata/np_relext.npy"
    train_data_np_lbls_file = destination_dir_path + "/train/np_relext_labels.npy" #"/Users/ajaynagesh/Research/LadderNetworks/mean-teacher/pytorch/processNLPdata/np_relext_labels.npy"
    test_data_np_file = destination_dir_path + "/val/np_relext.npy"  #"/Users/ajaynagesh/Research/LadderNetworks/mean-teacher/pytorch/processNLPdata/np_relext.npy"
    test_data_np_lbls_file = destination_dir_path + "/val/np_relext_labels.npy" #"/Users/ajaynagesh/Research/LadderNetworks/mean-teacher/pytorch/processNLPdata/np_relext_labels.npy"

    print("Parsing the raw dataset ... from ... " + train_data_path)
    raw_train_data = parse_data(train_data_path)
    print("Parsing the raw dataset ... from ... " + test_data_path)
    raw_test_data = parse_data(test_data_path)

    ## Relation labels
    relation_dict =  dict((lbl, id) for (id, lbl) in enumerate(sorted(list({datum[4] for datum in raw_train_data}))))
    print("---------------------------")
    print ("Relation Mappings : ")
    print("---------------------------")
    print (relation_dict)
    print ("---------------------------")

    print("Loading the gigaword embeddings ...")
    gigaW2vEmbed, lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)

    embedding_dataset_train, relation_labels_train = create_data_for_pytorch(raw_train_data, relation_dict, lookupGiga, gigaW2vEmbed)
    embedding_dataset_test, relation_labels_test = create_data_for_pytorch(raw_test_data, relation_dict, lookupGiga, gigaW2vEmbed)

    np.save(train_data_np_file, embedding_dataset_train)
    np.save(train_data_np_lbls_file, np.array(relation_labels_train))

    np.save(test_data_np_file, embedding_dataset_test)
    np.save(test_data_np_lbls_file, np.array(relation_labels_test))