import sys
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # used for compute cosine similarity for sparse matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression


parent_folder_path = str(Path('.').absolute().parent)
utils_folder_path = parent_folder_path+'/utils'
models_folder_path = parent_folder_path+'/models'
data_folder_path = parent_folder_path + '/data'

sys.path+=[parent_folder_path, utils_folder_path, models_folder_path, data_folder_path]

import numpy as np

import pickle
import utils_probe_squad
import utils_dataset_squad
import torch
import time
import torch.nn as nn
import sklearn.preprocessing
import random
import datetime
from utils import get_map
from utils import get_ppl


class Experiment():
    def __init__(self, vocab_dict, tfidf_vectorizer, probe_data_dir, seed, model_type = "useqa", input_type = "query_useqa_embd", label_type = "gold", device = "cpu"):
        self.seed = seed
        torch.manual_seed(seed)

        self.probe_data_dir = probe_data_dir
        self.vocab_dict = vocab_dict
        self.target_vocab_size = len(vocab_dict)
        self.tfidf_vocab_size = len(tfidf_vectorizer.vocabulary_)

        self.model_type = model_type
        self.input_type = input_type
        self.label_type = label_type
        self.query_indices = "lemma_query_indices_"+label_type
        self.fact_indices = "lemma_fact_indices_"+label_type
        self.negative_indices = "lemma_negative_indices_"+label_type

        if model_type=="useqa":
            self.linear = nn.Linear(512, self.target_vocab_size).to(device)
        elif model_type=="tfidf":
            self.linear = nn.Linear(self.tfidf_vocab_size, self.target_vocab_size).to(device)

        self.optimizer = torch.optim.Adam(self.linear.parameters())
        self.criterion = nn.BCELoss(reduction = "none")

        self.label_binarizer = sklearn.preprocessing.LabelBinarizer()
        self.label_binarizer.fit(range(self.target_vocab_size))

        self.device = device

    def get_training_labels(self, query_indices, fact_indices, negative_indices):
        label_masks = list(set(query_indices+fact_indices+negative_indices))
        label_masks_onehot = np.sum(self.label_binarizer.transform(label_masks), axis=0)

        labels = list(set(query_indices+fact_indices))
        labels_onehot = np.sum(self.label_binarizer.transform(labels), axis=0)

        return labels_onehot, label_masks_onehot, np.array(labels), np.array(label_masks)

    def get_loss(self, prediction, target, mask):
        loss = torch.sum(self.criterion(prediction, target)*mask)/torch.sum(mask)
        return loss

    def train_epoch_linear_probe(self, train_list, epoch, save_folder_path):

        self.linear.train()
        total_loss = 0
        random.shuffle(train_list)
        map_list = list([])
        ppl_list = list([])
        for i, instance in enumerate(train_list):
            self.optimizer.zero_grad()
            labels_onehot, masks_onehot, labels, label_masks = self.get_training_labels(instance[self.query_indices], instance[self.fact_indices], instance[self.negative_indices])

            output_ = self.linear(instance[self.input_type].to(self.device))  # output size is (6600)
            output = nn.functional.sigmoid(output_)

            loss = self.get_loss(output, torch.tensor(labels_onehot, dtype = torch.float32).to(self.device), torch.tensor(masks_onehot, dtype = torch.float32).to(self.device))
            loss.backward()
            self.optimizer.step()

            total_loss+=loss.detach().cpu().numpy()

            map_list.append(get_map(output.detach().cpu().numpy(), labels))
            ppl_list.append(get_ppl(output.detach().cpu().numpy(), labels))

            # if (i + 1) % 10 == 0:
            #     print("\tsample ",i+1, " loss:", total_loss/(i+1))

        print("epoch ", epoch,"\tbert total training loss:", total_loss/len(train_list))

        return total_loss/len(train_list)

    def eval_epoch_linear_probe(self, eval_list, epoch, vocab_dict, print_sample =False):
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(self.target_vocab_size))
        vocab_dict_rev = {v: k for k, v in vocab_dict.items()}
        self.linear.eval()
        total_loss = list([])

        map_list = list([])
        ppl_list = list([])
        query_map_list = list([])
        query_ppl_list = list([])
        target_map_list = list([])
        target_ppl_list = list([])

        overflow_count = 0
        with torch.no_grad():
            for i, instance in enumerate(eval_list):
                labels_onehot, masks_onehot, labels, _ = self.get_training_labels(instance[self.query_indices],
                                                                                  instance[self.fact_indices],
                                                                                  instance[self.negative_indices])

                output_ = self.linear(instance[self.input_type].to(self.device))  # output size is (6600)
                output = nn.functional.sigmoid(output_)

                loss = self.get_loss(output, torch.tensor(labels_onehot, dtype=torch.float32).to(self.device),
                                     torch.tensor(masks_onehot, dtype=torch.float32).to(self.device))

                total_loss.append(loss.detach().cpu().numpy())

                map_list.append(get_map(output.detach().cpu().numpy(), labels))
                ppl_list.append(get_ppl(output.detach().cpu().numpy(), labels))

                query_labels_eval =  np.array(instance[self.query_indices])
                target_labels_eval = np.array(list(set(instance[self.fact_indices])-set(instance[self.query_indices])))

                if len(query_labels_eval)>0:
                    query_map_list.append(get_map(output.detach().cpu().numpy(), query_labels_eval))
                    query_ppl = get_ppl(output.detach().cpu().numpy(), query_labels_eval)
                    if ~np.isinf(query_ppl):
                        query_ppl_list.append(query_ppl)
                    else:
                        overflow_count+=1
                if len(target_labels_eval)>0:
                    target_map_list.append(get_map(output.detach().cpu().numpy(), target_labels_eval))
                    target_ppl = get_ppl(output.detach().cpu().numpy(), target_labels_eval)
                    if ~np.isinf(target_ppl):
                        target_ppl_list.append(target_ppl)
                    else:
                        overflow_count+=1

                if len(target_ppl_list)>0 and np.isinf(target_ppl_list[-1]):
                    print("="*20)
                    print("\tquery", instance["query"])
                    print("\toutput sum", torch.sum(output))
                    print("\tlabel sum:", np.sum(target_labels_eval))
                    print("\t", self.query_indices)
                    print("\t", self.fact_indices)
                    input("AA")

        result_dict = {"eval_loss":total_loss,
                       "avg map":map_list,
                       "avg ppl":ppl_list,
                       "query map:": query_map_list,
                       "query ppl:": query_ppl_list,
                       "target map:": target_map_list,
                       "target ppl:": target_ppl_list}
        print("-" * 20)
        result_summary = {x:sum(result_dict[x])/len(result_dict[x]) for x in result_dict.keys()}

        print(result_summary)
        print(overflow_count)

        return result_summary , result_dict

    def train_all(self, train_list, eval_list, vocab_dict, n_epoch):
        ###########################################
        # train and evaluate useqa linear classifier
        ###########################################
        save_folder_path = self.probe_data_dir+self.input_type+"_"+self.label_type+"_result_seed_"+str(self.seed)+"/"
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        print("="*20)
        useqa_results_list = list([])
        best_loss = 1000
        for epoch in range(n_epoch):
            train_loss = self.train_epoch_linear_probe(train_list, epoch, save_folder_path)
            result_summary, result_dict = self.eval_epoch_linear_probe( eval_list, epoch, vocab_dict)
            result_summary["train_loss"] = train_loss
            useqa_results_list.append(result_summary)
            print("-" * 20)

            if result_summary["eval_loss"]<best_loss:
                best_loss= result_summary["eval_loss"]

                # Save the linear probe:
                torch.save(self.linear, save_folder_path + "best_linear_prober")

                # save the total results:
                with open(save_folder_path + "best_epoch_result.pickle", "wb") as handle:
                    pickle.dump(result_dict, handle)

        with open(save_folder_path+"result_summary.pickle", "wb") as handle:
            pickle.dump(useqa_results_list, handle)


def experiments_squad(device):
    useqa_embds_paths = {
        "train":"data_generated/squad/squad_ques_train_embds.npy",
        "dev":"data_generated/squad/squad_ques_dev_embds.npy"
    }

    saved_data_folder = 'data_generated/squad/'
    if not os.path.exists(saved_data_folder):
        os.mkdir(saved_data_folder)

    train_list, dev_list, kb = utils_dataset_squad.load_squad_probe_raw_data()
    vocab_dict, tfidf_vectorizer = utils_probe_squad.get_vocabulary(train_list, kb, saved_data_folder+"squad_vocab_dict.pickle", saved_data_folder+"squad_tfidf_vectorizer.pickle")
    instances_all_seeds = utils_probe_squad.get_probe_dataset(train_list, dev_list, kb, useqa_embds_paths, vocab_dict, tfidf_vectorizer, saved_data_folder, "squad_probe.pickle")

    now = datetime.datetime.now()
    date_time = str(now)[:10] + '_' + str(now)[11:13] + str(now)[14:16] + str(now)[17:19]
    saved_result_folder_path = saved_data_folder + '/probe_experiment_' + date_time + "/"

    if not os.path.exists(saved_result_folder_path):
        os.mkdir(saved_result_folder_path)

    for random_seed in range(5):
        # Exp1: useqa trained embedding and gold probe label. Best @ epoch 2
        experiment = Experiment(vocab_dict, tfidf_vectorizer, saved_result_folder_path, random_seed, model_type = "useqa",
                                input_type = "query_useqa_embd", label_type = "gold", device = device)
        experiment.train_all(instances_all_seeds[random_seed]["train"], instances_all_seeds[random_seed]["dev"],
                             vocab_dict, 5)


        # Exp2: useqa random embedding and gold probe label. Best @ epoch 0
        experiment = Experiment(vocab_dict, tfidf_vectorizer, saved_result_folder_path, random_seed, model_type="useqa",
                                input_type="query_random_embd", label_type="gold", device=device)
        experiment.train_all(instances_all_seeds[random_seed]["train"], instances_all_seeds[random_seed]["dev"],
                             vocab_dict, 4)


        # Exp3: tf-idf embedding and gold probe label.  Best @ epoch epoch 2
        experiment = Experiment(vocab_dict, tfidf_vectorizer, saved_result_folder_path, random_seed, model_type="tfidf",
                                input_type="query_tfidf_embd", label_type="gold", device=device)
        experiment.train_all(instances_all_seeds[random_seed]["train"], instances_all_seeds[random_seed]["dev"],
                             vocab_dict, 6)


        # Exp4: useqa trained embedding and question shuffled probe label. Best @ epoch
        experiment = Experiment(vocab_dict, tfidf_vectorizer, saved_result_folder_path, random_seed, model_type="useqa",
                                input_type="query_useqa_embd", label_type="ques_shuffle", device=device)
        experiment.train_all(instances_all_seeds[random_seed]["train"], instances_all_seeds[random_seed]["dev"],
                             vocab_dict, 5)


        # Exp5: useqa trained embedding and token remapped probe label.
        # experiment = Experiment(vocab_dict, tfidf_vectorizer, saved_result_folder_path, random_seed, model_type="useqa",
        #                         input_type="query_useqa_embd", label_type="token_remap", device=device)
        # experiment.train_all(instances_all_seeds[random_seed]["train"], instances_all_seeds[random_seed]["dev"],
        #                      vocab_dict, 20)

    return 0

def experiments_squad_manual_check(device, data_partition = "train", print_text = False, embd_type = "useqa", label_type = "gold", seed = 0, epoch = 1):

    def get_training_labels(label_binarizer, query_indices, fact_indices, negative_indices):
        label_masks = list(set(query_indices+fact_indices+negative_indices))
        label_masks_onehot = np.sum(label_binarizer.transform(label_masks), axis=0)

        labels = list(set(query_indices+fact_indices))
        labels_onehot = np.sum(label_binarizer.transform(labels), axis=0)

        return labels_onehot, label_masks_onehot, np.array(labels), np.array(label_masks)


    def get_loss(criterion, prediction, target, mask):
        loss = torch.sum(criterion(prediction, target)*mask)/torch.sum(mask)
        return loss

    probe_model_root_path = "data_generated/squad/probe_experiment_2020-05-30_215643/"
    input_type = "query_"+embd_type+"_embd"
    probe_model_path = probe_model_root_path+"query_"+embd_type+"_embd_"+label_type+"_result_seed_"+str(seed)+"/best_linear_prober"
    saved_data_folder = 'data_generated/squad/'

    train_list, dev_list, kb = utils_dataset_squad.load_squad_probe_raw_data()
    vocab_dict, tfidf_vectorizer = utils_probe_squad.get_vocabulary(train_list, kb,
                                                                saved_data_folder + "squad_vocab_dict.pickle",
                                                                saved_data_folder + "squad_tfidf_vectorizer.pickle")

    instances_all_seeds = utils_probe_squad.get_probe_dataset(train_list, dev_list, kb, "",
                                                          vocab_dict, tfidf_vectorizer, saved_data_folder,
                                                          "squad_probe.pickle")


    linear_probe = torch.load(probe_model_path).to(device)
    linear_probe.eval()
    criterion = nn.BCELoss(reduction = "none")

    target_vocab_size = len(vocab_dict)
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(target_vocab_size))
    vocab_dict_rev = {v: k for k, v in vocab_dict.items()}

    query_indices = "lemma_query_indices_"+label_type
    fact_indices = "lemma_fact_indices_"+label_type
    negative_indices ="lemma_negative_indices_"+label_type

    data_list = instances_all_seeds[seed][data_partition]

    total_loss = 0
    map_list = list([])
    ppl_list = list([])
    query_map_list = list([])
    query_ppl_list = list([])
    target_map_list = list([])
    target_ppl_list = list([])

    pred_score_dict = {}
    target_occur_dict = {}
    with torch.no_grad():
        for i, instance in enumerate(data_list):
            labels_onehot, masks_onehot, labels, label_masks = get_training_labels(label_binarizer,
                                                                         instance[query_indices],
                                                                         instance[fact_indices],
                                                                         instance[negative_indices])

            output_ = linear_probe(instance[input_type].to(device))  # output size is (6600)
            output = nn.functional.sigmoid(output_)

            if print_text:
                output_numpy = output.detach().cpu().numpy()
                top_preds = np.flip(np.argsort(output_numpy))
                print("="*20)
                print("\tquery:", instance["lemmas_query"])
                print("\tfact:", instance["lemmas_fact"])
                print('\ttop pred lemma:', [vocab_dict_rev[idx] for idx in top_preds[:20]])
                input("A")

            loss = get_loss(criterion, output, torch.tensor(labels_onehot, dtype=torch.float32).to(device),
                                 torch.tensor(masks_onehot, dtype=torch.float32).to(device))

            total_loss += loss.detach().cpu().numpy()

            map_list.append(get_map(output.detach().cpu().numpy(), labels))
            ppl_list.append(get_ppl(output.detach().cpu().numpy(), labels))

            query_map_list.append(get_map(output.detach().cpu().numpy(), np.array(instance[query_indices])))
            query_ppl_list.append(get_ppl(output.detach().cpu().numpy(), np.array(instance[query_indices])))

            if len(set(instance[fact_indices]) - set(instance[query_indices]))>0:
                target_map_list.append(get_map(output.detach().cpu().numpy(), np.array(
                    list(set(instance[fact_indices]) - set(instance[query_indices])), dtype = np.int64)))
                target_ppl_list.append(get_ppl(output.detach().cpu().numpy(), np.array(
                    list(set(instance[fact_indices]) - set(instance[query_indices])), dtype = np.int64)))

            for pred_lemma_indices in list(set(instance[fact_indices]) - set(instance[query_indices])):
                pred_lemma = vocab_dict_rev[pred_lemma_indices]
                if pred_lemma not in pred_score_dict:
                    pred_score_dict[pred_lemma] = 0
                    target_occur_dict[pred_lemma] = 0
                target_occur_dict[pred_lemma]+=1
                pred_score_dict[pred_lemma]+=output[pred_lemma_indices].item()


            if print_text:
                print("="*20)
                print("query:",instance["lemmas_query"])
                print("fact",instance["lemmas_fact"])
                print("negative",instance["lemmas_negative"])

                print("positive token reconstructed:", [vocab_dict_rev[lemma_idx] for lemma_idx in labels])
                print("negative token reconstructed:", [vocab_dict_rev[lemma_idx] for lemma_idx in list(set(label_masks)-set(labels))])
                print("query reconstructed", [vocab_dict_rev[lemma_idx] for lemma_idx in instance[query_indices]])
                print("fact alone reconstructed:", [vocab_dict_rev[lemma_idx] for lemma_idx in instance[fact_indices]])

                input("--------")

    result_dict = {"eval_loss": total_loss / len(dev_list),
                   "avg map": sum(map_list) / len(map_list),
                   "avg ppl": sum(ppl_list) / len(ppl_list),
                   "query map:": sum(query_map_list) / len(query_map_list),
                   "query ppl:": sum(query_ppl_list) / len(query_ppl_list),
                   "target map:": sum(target_map_list) / len(target_map_list),
                   "target ppl:": sum(target_ppl_list) / len(target_ppl_list)}
    print("-" * 20)
    print(result_dict)

    print("-"*20)
    pred_freq_dict_avg = {}
    for k in pred_score_dict.keys():
        pred_freq_dict_avg[k] = pred_score_dict[k]/target_occur_dict[k]

    tokens_sorted_by_occur = sorted(target_occur_dict.items(), key=lambda kv: kv[1])
    for histo_tuple in list(reversed(tokens_sorted_by_occur)):
        print("token:", histo_tuple[0], "\tn occur:", histo_tuple[1],"\tavg prob:",pred_freq_dict_avg[histo_tuple[0]] )


    return 0

def main():
    cuda_num = "0"

    print('threads before set:', torch.get_num_threads())
    torch.set_num_threads(1)
    device = torch.device("cuda:" + cuda_num if torch.cuda.is_available() else "cpu")
    print(device)
    print('threads after set:', torch.get_num_threads())

    experiments_squad(device)
    #experiments_squad_manual_check(device, data_partition="dev", print_text = True, label_type = "gold")

    return 0

main()

