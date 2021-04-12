import pickle
import numpy as np

def load_raw_squad_data(saved_squad_file_pcikle = "data_raw/squad_retrieval_data.pickle"):
    with (saved_squad_file_pcikle, "rb") as handle:
        squad_raw_data = pickle.load(handle)

    return squad_raw_data

def load_squad_query_embeddings(saved_squad_embds_train = "data_raw/ques_train_embds.npy",
                                saved_squad_embds_dev = "data_raw/ques_dev_embds.npy"):
    train_embds = np.load(saved_squad_embds_train)

    dev_embds = np.load(saved_squad_embds_dev)

    return train_embds, dev_embds

def squad_dev_id_to_index_in_train():
    query_id2idx_dict = {}

    with open("/Users/zhengzhongliang/NLP_Research/2020_HybridRetrieval/HybridRetrieval/data_generated/squad_useqa/squad_retrieval_data.pickle","rb") as handle:
        instances_list = pickle.load(handle)

    for i, instance in enumerate(instances_list["train_list"]):
        query_id2idx_dict[instance["id"]] = i

    # Load squad dev query id
    with open("/Users/zhengzhongliang/NLP_Research/2020_HybridRetrieval/IR_BM25/data/squad/raw_data/squad_dev_id.txt","r") as handle:
        query_id_list = handle.read().split("\n")[:-1]  # [-1] is to remove the last line, which is empty

    assert(len(query_id_list)==10000)

    with open(
            "/Users/zhengzhongliang/NLP_Research/2020_HybridRetrieval/IR_BM25/data/squad/raw_data/squad_dev_label.txt",
            "r") as handle:
        query_label_list = handle.read().split("\n")[:-1]

    assert (len(query_label_list) == 10000)

    dev_query_idx_in_train_list = []
    for i, query_id in enumerate(query_id_list):
        dev_query_idx_in_train_list.append(query_id2idx_dict[query_id])
        assert (int(query_label_list[i]) == instances_list["train_list"][dev_query_idx_in_train_list[-1]][
            "response"])

    np.save("data_raw/squad/squad_dev_ques_idx_in_train.npy", np.array(dev_query_idx_in_train_list))

    return 0

def generate_squad_dev_embds():
    dev_ques_idx_in_train = np.load("data_raw/squad/squad_dev_ques_idx_in_train.npy")

    embds_list = []

    squad_embds = np.load("data_raw/squad/squad_ques_train_embds_full.npy")

    for idx in dev_ques_idx_in_train:
        embds_list.append(squad_embds[idx])

    embds_list = np.array(embds_list)
    np.save("data_raw/squad/squad_ques_train_embds.npy", embds_list)

    print(embds_list.shape)

    return 0

def generate_squad_probe_data():
    # This should be in a similar format as openbook data.
    with open("/Users/zhengzhongliang/NLP_Research/2020_HybridRetrieval/HybridRetrieval/data_generated/squad_useqa/squad_retrieval_data.pickle","rb") as handle:
        instances_list = pickle.load(handle)

    doc_list = instances_list["doc_list"]
    resp_list = instances_list["resp_list"]

    dev_ques_idx_in_train = np.load("data_raw/squad/squad_dev_ques_idx_in_train.npy")

    # kb: only collect the docs. Because we want to reconstruct lexical occurence, and sentence must be in the doc.
    # so we do not need to store sentences and docs separately. Only doc is enough.
    kb_doc_indices = {}
    kb_doc = []

    # generate dev instances:
    probe_instance_train = []
    for i, idx in enumerate(dev_ques_idx_in_train):
        id = instances_list["train_list"][idx]["id"]
        query = instances_list["train_list"][idx]["question"]
        doc_id = instances_list["train_list"][idx]["document"]
        if doc_id not in kb_doc_indices:
            kb_doc_indices[doc_id] = len(kb_doc_indices)
            kb_doc.append(instances_list["doc_list"][doc_id])

        doc_id_new = kb_doc_indices[doc_id]
        facts = doc_id_new

        probe_instance_train.append({"id":id, "query":query, "doc_id_":doc_id, "doc_id_new":doc_id_new, "facts": facts})

        if i<2:
            print("="*20)
            print("\tquery:", probe_instance_train[i]["query"])
            print("\tdoc:", instances_list["doc_list"][probe_instance_train[i]["doc_id_"]])
            print("\tdoc:", kb_doc[probe_instance_train[i]["facts"]])

        assert(instances_list["doc_list"][probe_instance_train[i]["doc_id_"]] == kb_doc[probe_instance_train[i]["facts"]])

    probe_instance_dev = []
    for idx, instance in enumerate(instances_list["dev_list"]):
        id = instance["id"]
        query = instance["question"]
        doc_id = instance["document"]

        if doc_id not in kb_doc_indices:
            kb_doc_indices[doc_id] = len(kb_doc_indices)
            kb_doc.append(instances_list["doc_list"][doc_id])

        doc_id_new = kb_doc_indices[doc_id]
        facts = doc_id_new

        probe_instance_dev.append(
            {"id": id, "query": query, "doc_id_": doc_id, "doc_id_new": doc_id_new, "facts": facts})

        if idx < 2:
            print("=" * 20)
            print("\tquery:", probe_instance_dev[idx]["query"])
            print("\tdoc:", instances_list["doc_list"][probe_instance_dev[idx]["doc_id_"]])
            print("\tdoc:", kb_doc[probe_instance_dev[idx]["facts"]])

        assert (instances_list["doc_list"][probe_instance_dev[idx]["doc_id_"]] == kb_doc[
            probe_instance_dev[idx]["facts"]])

    with open("data_raw/squad/squad_probe_data_raw.pickle", "wb") as handle:
        pickle.dump({"train_list": probe_instance_train, "dev_list": probe_instance_dev, "kb": kb_doc} , handle)

    return 0

def load_squad_probe_raw_data():
    with open("data_raw/squad/squad_probe_data_raw.pickle", "rb") as handle:
        all_data = pickle.load(handle)

    train_list = all_data["train_list"]
    dev_list = all_data["dev_list"]
    kb = all_data["kb"]

    print("squad probe raw data loaded!")
    print("train list size:",len(train_list))
    print("dev list size:", len(dev_list))
    print("kb size:", len(kb))

    return train_list, dev_list, kb

