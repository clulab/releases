import torch.nn as nn
import numpy as np
import torch
import torch.functional as F
import pickle
import os
import json
import time

from sklearn.metrics import precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from pathlib import Path

from transformers import BertTokenizer

class BiLSTMLM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                 n_layers, dropout_p):
        super(BiLSTMLM, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # |input| = (batch_size, max_seq_len)

        embeds = self.embedding(input)
        # |embeds| = (batch_size, max_seq_len, embedding_size)

        lstm_out, hidden = self.lstm(embeds)
        # If bidirectional=True, num_directions is 2, else it is 1.
        # |lstm_out| = (batch_size, max_seq_len, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)

        forward_out = lstm_out[:, :-2, :self.hidden_size]
        backward_out = lstm_out[:, 2:, self.hidden_size:]
        # |forward_out| = (batch_size, max_seq_len-2, hidden_size)
        # |backward_out| = (batch_size, max_seq_len-2, hidden_size)

        context = torch.cat((forward_out, backward_out), dim=-1)
        # |context| = (batch_size, max_seq_len-2, 2*hidden_size)

        fc_out = self.fc2(self.fc(context))
        # |fc_out| = (batch_size, max_seq_len-2, hidden_size//2)
        output = self.softmax(self.fc3(fc_out))
        # |output| = (batch_size, max_seq_len-2, output_size)

        return output

class LSTMTokenizer:

    def __init__(self, token_to_index):
        self.vocab = token_to_index
        self.index_to_token = {v: k for (k, v) in token_to_index.items()}

    def convert_ids_to_tokens(self, list_of_ids):
        return [self.index_to_token[idx] for idx in list_of_ids]

class EmbeddingUtils:
    '''
    This class has the functions to initialize the embedding layer of the LSTM network.
    '''

    # w2v_general: uses the w2v embeddings trained on pubmed by clulab
    # w2v_in_domain: uses the w2v embeddings trained on 10,000 pubmed papers about interactions.
    # bert_pubmed_10000: uses the bert tokenizer, but limit the size of vocab to be 10,000.
    all_embd_opts = ["w2v_general", "w2v_in_domain", "bert_pubmed_10000"]

    @classmethod
    def load_w2v_embeddings(cls, w2v_path):
        '''
        Loads the w2v embeddings, and save it as the dictionary of {term: np_array}
        :param w2v_path:
        :return:
        '''
        assert(w2v_path.endswith(".txt"))

        word2vec_dict = {}
        with open(w2v_path, "r") as f:
            all_vocabs = f.readlines()

        for i in range(1, len(all_vocabs)):  # skip the first line because it is not an actual entry.
            term_entry = all_vocabs[i].split(" ")
            term = term_entry[0]
            embd = np.array([float(num) for num in term_entry[1:-1]])
            word2vec_dict[term] = embd

        return word2vec_dict

    @classmethod
    def get_lstm_embeddings(cls,
                            embd_opt,
                            input_sequences,
                            max_vocab_size,
                            vocab_min_num_occur,
                            embd_dim=100,
                            ):
        '''
        Gets the actual embedding layer for the LSTM.
        :param embd_opt: the supported types of embedding.
        :param input_sequences: the input sequences, which are used for building the vocabulary. Should be from the training set.
        :param vocab_min_num_occur: the minimum number of occurrences of a token to be considered to add to the vocabulary
        :param embd_dim: set to 100 due to our pretrained embeddings.
        :return:
        '''
        assert embd_opt in cls.all_embd_opts, "lstm embedding option not implementation!"
        if embd_opt == "w2v_general" or embd_opt == "w2v_in_domain":
            # For these two options, the vocabulary is built from the input sequences.
            if embd_opt == "w2v_general":
                alix_w2v_path = "/home/zhengzhongliang/CLU_Projects/embeddings_november_2016.txt"
                clara_w2v_path = "/work/zhengzhongliang/2020_ASKE/embeddings_november_2016.txt"
                w2v_path = alix_w2v_path if os.path.exists(alix_w2v_path) else clara_w2v_path
            else:
                # This file is on clara.
                w2v_path = "/work/zhengzhongliang/2020_ASKE/20211124_train_pubmed_embeddings/20211124_pubmed_for_causal_detection_10k_article_w2v_vectors.txt"

            w2v_dict = cls.load_w2v_embeddings(w2v_path)

            # This dictionary counts the number of occurrences of the tokens
            token_count_dict = {}
            for seq in input_sequences:
                seq_tokens = seq["seg1_tokens"] + seq["seg2_tokens"] + seq["seg3_tokens"]
                for token in seq_tokens:
                    if token.lower() not in token_count_dict:
                        token_count_dict[token.lower()] = 1
                    else:
                        token_count_dict[token.lower()] += 1

            # Sort the tokens from the most frequent to the least frequent.
            token_count_dict_list = list(
                (k, v) for k, v in sorted(token_count_dict.items(), key=lambda item: -item[1]) if v >= vocab_min_num_occur)

            # This builds the actual map between each token and the index.
            token_to_index = {}
            token_to_index["[PAD]"] = 0
            token_to_index["unk"] = 1  # The unknown token is set to this because it matches the w2v embeddings.
            token_to_index["[SEP]"] = 2
            for i, (token, count) in enumerate(token_count_dict_list):
                if i >= max_vocab_size-3:
                    break
                else:
                    token_to_index[token] = i + 3
            tokenizer = LSTMTokenizer(token_to_index)

            # Initialize the embedding layer and initialize the weights with the w2v embeddings.
            embedding_layer = nn.Embedding(min(max_vocab_size, len(token_to_index)), embd_dim)
            num_token_of_embd_layer = embedding_layer.weight.size()[0]

            weight_matrix = []
            n_vocab_missing_from_word2vec = 0
            for (term, term_idx) in list(token_to_index.items())[:num_token_of_embd_layer]:
                if term in w2v_dict:
                    weight_matrix.append(w2v_dict[term])
                else:
                    weight_matrix.append(np.random.rand(embd_dim))
                    n_vocab_missing_from_word2vec += 1

            weight_matrix = np.array(weight_matrix)

            embedding_layer.weight = nn.Parameter(torch.tensor(weight_matrix, dtype=torch.float32))

            if not embedding_layer.weight.requires_grad:
                embedding_layer.weight.requires_grad = True

            print("embed layer built from train data! vocab size: ", len(token_to_index))
            print("top 10 token:", list(token_to_index.items())[:10])
            print("n token missing from word2vec:", n_vocab_missing_from_word2vec)

        else:
            # First load token to index
            # project_folder_path = Path(__file__).resolve().parents[1].absolute()  # Two levels up
            alix_project_folder_path = "/home/zhengzhongliang/CLU_Projects/2020_ASKE/ASKE_2020_CausalDetection"
            clara_project_folder_path = "/work/zhengzhongliang/2020_ASKE/neural_baseline/ASKE_2020_CausalDetection"
            project_folder_path = alix_project_folder_path if os.path.exists(alix_project_folder_path) else clara_project_folder_path

            vocab_file_folder_path = str(project_folder_path) + "/Experiments2/truncate_vocab/bert_base_uncased_pubmed_10000/"

            tokenizer = BertTokenizer.from_pretrained(vocab_file_folder_path)

            # Then load the embedding layer.
            model_path = str(project_folder_path) + "/Experiments2/saved_models_20211229_lstm_language_model/20211215_bilstm_vocab_bert_10000_hidden_700/bilstm_lm_epoch_8"
            bilstm_language_model = torch.load(model_path)

            embedding_layer = bilstm_language_model.embedding

            if not embedding_layer.weight.requires_grad:
                embedding_layer.weight.requires_grad = True

        return tokenizer, embedding_layer

    @classmethod
    def check_embedding_loading(cls):
        a = cls.get_lstm_embeddings(embd_opt="bert_pubmed_10000",
                                    input_sequences=[],
                                    max_vocab_size=20000,
                                    vocab_min_num_occur=2,
                                    embd_dim=100)

        print(a)


class LSTM1Base(nn.Module):
    def __init__(self,
                 embd_opt="w2v_general",
                 vocab_size=10000,
                 vocab_min_num_occur=2,
                 input_dim=100,
                 hidden_dim=100,
                 train_list=[],
                 device=torch.device("cuda:0")):

        super().__init__()

        self.vocab_size = vocab_size
        self.input_dim = input_dim  # This 100 should be the word2vec size
        self.hidden_dim = hidden_dim

        self.tokenizer, self.embedding_layer = EmbeddingUtils.get_lstm_embeddings(embd_opt=embd_opt,
                                                                                       input_sequences=train_list,
                                                                                       max_vocab_size=vocab_size,
                                                                                       vocab_min_num_occur=vocab_min_num_occur,
                                                                                       embd_dim=input_dim)

        if embd_opt == "bert_pubmed_10000":
            alix_project_folder_path = "/home/zhengzhongliang/CLU_Projects/2020_ASKE/ASKE_2020_CausalDetection"
            clara_project_folder_path = "/work/zhengzhongliang/2020_ASKE/neural_baseline/ASKE_2020_CausalDetection"
            project_folder_path = alix_project_folder_path if os.path.exists(
                alix_project_folder_path) else clara_project_folder_path

            model_path = str(
                project_folder_path) + "/Experiments2/saved_models_20211229_lstm_language_model/20211215_bilstm_vocab_bert_10000_hidden_700/bilstm_lm_epoch_8"
            bilstm_lm = torch.load(model_path)
            self.bilstm = bilstm_lm.lstm  # I am sure the weights require grad here.

        else:
            self.bilstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True, batch_first=True)

        self.device = device


class LSTM1Seg(LSTM1Base):
    def __init__(self,
                 embd_opt="w2v_general",
                 vocab_size=10000,
                 vocab_min_num_occur=2,
                 input_dim=300,
                 hidden_dim=50,
                 mlp_hidden_dim=100,
                 train_list=[],
                 device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")):
        super().__init__(embd_opt, vocab_size, vocab_min_num_occur, input_dim, hidden_dim, train_list, device)
        self.mlp_hidden_dim = mlp_hidden_dim

        self.linear_bottom = nn.Linear(hidden_dim*2, mlp_hidden_dim)
        self.relu = nn.ReLU()
        self.linear_top = nn.Linear(mlp_hidden_dim, 3)

        self.device = device

    def forward(self, data_batch):

        # The input should be a list of input sequences (represented by indices, not padded)
        input_lengths = torch.tensor([len(sample) for sample in data_batch], dtype=torch.int64).to(self.device)

        padded_embedding_tensor = pad_sequence([self.embedding_layer(sample.to(self.device)) for sample in data_batch],
                                               batch_first=True).to(self.device)

        packed_padded_embedding_tensor = pack_padded_sequence(padded_embedding_tensor,
                                                              input_lengths,
                                                              batch_first=True,
                                                              enforce_sorted=False)

        lstm_output_packed, _ = self.bilstm(packed_padded_embedding_tensor)

        lstm_output_padded, input_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_packed, batch_first=True)

        lstm_output = torch.stack([torch.max(lstm_output_padded[i, :input_lengths[i], :], dim=0)[0] for i in range(len(input_lengths))], dim=0)

        mlp_output = self.linear_top(self.relu(self.linear_bottom(lstm_output)))

        return mlp_output


class LSTM3Seg(LSTM1Base):
    def __init__(self, vocab_size = 10000, input_dim = 300, hidden_dim = 50, mlp_hidden_dim = 50, pretrained_embd = False, train_list = [], device = torch.device("cuda:0")):
        super().__init__(vocab_size, input_dim, hidden_dim, pretrained_embd, train_list, device)
        self.mlp_hidden_dim = mlp_hidden_dim

        self.linear_bottom = nn.Linear(hidden_dim * 2, mlp_hidden_dim)
        self.relu = nn.ReLU()
        self.linear_top = nn.Linear(mlp_hidden_dim, 3)

    def forward(self, data_batch):
        # The input should be a list of input sequences (represented by indices, not padded)

        input_lengths = torch.tensor([len(sample) for sample in data_batch], dtype=torch.int64)
        padded_embedding_tensor = pad_sequence([self.embedding_layer(sample) for sample in data_batch],
                                               batch_first=True)
        packed_padded_embedding_tensor = pack_padded_sequence(padded_embedding_tensor, input_lengths, batch_first=True,
                                                              enforce_sorted=False)

        lstm_output_packed, _ = self.bilstm(packed_padded_embedding_tensor)
        lstm_output_padded, input_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_packed, batch_first=True)

        lstm_output = torch.stack(
            [torch.max(lstm_output_padded[i, :input_lengths[i], :], dim=0)[0] for i in range(len(input_lengths))],
            dim=0)

        mlp_output = self.linear_top(self.relu(self.linear_bottom(lstm_output)))

        return mlp_output

class LSTM3SegAtt(LSTM1Base):
    def __init__(self, vocab_size=20000,
                 input_dim=300,
                 hidden_dim=64,
                 mlp_hidden_dim=64,
                 pretrained_embd=False,
                 train_list=[],
                 device=torch.device("cuda:0")):
        super().__init__(vocab_size, input_dim, hidden_dim, pretrained_embd, train_list, device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.mlp_hidden_dim = mlp_hidden_dim

        self.linear_bottom = nn.Linear(hidden_dim * 2, mlp_hidden_dim)
        self.relu = nn.ReLU()
        self.linear_top = nn.Linear(mlp_hidden_dim, 3)

    def forward(self, data_batch):
        # The input should be a list of input sequences (represented by indices, not padded)

        input_lengths = torch.tensor([len(sample) for sample in data_batch], dtype=torch.int64)
        padded_embedding_tensor = pad_sequence([self.embedding_layer(sample) for sample in data_batch],
                                               batch_first=True)
        packed_padded_embedding_tensor = pack_padded_sequence(padded_embedding_tensor, input_lengths, batch_first=True,
                                                              enforce_sorted=False)

        lstm_output_packed, _ = self.bilstm(packed_padded_embedding_tensor)
        lstm_output_padded, input_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_packed, batch_first=True)

        lstm_outputs = [lstm_output_padded[i, :input_lengths[i], :].unsqueeze(0) for i in range(len(input_lengths))]
        self_att_outputs = [self.transformer_encoder(lstm_output).squeeze(0) for lstm_output in lstm_outputs]

        self_att_max_output = torch.stack([torch.max(self_att_output, dim=0)[0] for self_att_output in self_att_outputs],
            dim=0)

        mlp_output = self.linear_top(self.relu(self.linear_bottom(self_att_max_output)))

        return mlp_output


class LSTMExperiment:
    def __init__(self,
                 embd_opt="w2v_general",
                 model_name="LSTM1Seg",
                 vocab_min_num_occur=2,
                 hidden_dim=750,  # Number of lstm hidden units.
                 n_epoch=20,
                 train_list=[],
                 lr=0.0001,
                 device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")):

        self.model_name = model_name
        self.device = device
        # If the case flag is set to False, the lstm uses uncased models, and we might also want to use word2vec to initialize the embedding layer.

        if model_name == "LSTM1Seg":
            self.lstm_classifier = LSTM1Seg(embd_opt=embd_opt,
                                            vocab_size=10000,  # This is fixed for now.
                                            vocab_min_num_occur=vocab_min_num_occur,
                                            input_dim=100,  # This is fixed for now.
                                            hidden_dim=hidden_dim,
                                            mlp_hidden_dim=100,  # This is fixed for now.
                                            train_list=train_list,
                                            device=device)
        else:
            assert False, "specified model not implemented!"

        self.n_epoch = n_epoch
        self.adamOpt = torch.optim.Adam(params=self.lstm_classifier.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss()
        self.distill_criterion = torch.nn.MSELoss()

        self.lstm_classifier.to(device)

        assert(lr in [0.001, 0.0001, 0.00002])
        print("total number of params of lstm:", self.get_num_paras())

    def get_num_paras(self):
        total_num_params = 0
        for name, param in self.lstm_classifier.named_parameters():
            if param.requires_grad:
                param_shape = param.data.size()
                #print(name, param_shape)
                if len(param_shape) == 1:
                    total_num_params += param.data.size()[0]
                else:
                    total_num_params += param.data.size()[0] * param.data.size()[1]

        return total_num_params

    def save_classifier(self):

        return 0

    def train_epoch(self, batches, tokenizer, debug_flag=False, print_every=20):
        self.lstm_classifier.train()

        if debug_flag:
            print("="*20)

        print_loss_total = 0
        for batch_idx, batch in enumerate(batches):
            self.adamOpt.zero_grad()

            output = self.lstm_classifier(batch["token_indices"])

            loss = self.criterion(output, batch["target"].to(self.device))

            if debug_flag:
                print("\toutput:", output, " target:", batch["target"], " loss:", loss)

                if batch_idx < 3:
                    print("\tseg1  tokens:", batch["seg1_tokens"])
                    print("\tseg2  tokens:", batch["seg2_tokens"])
                    print("\tseg3  tokens:", batch["seg3_tokens"])
                    print("\tindices:", batch["token_indices"])
                    print("\treconstructed token:", [tokenizer.convert_ids_to_tokens([token_index.item() for token_index in indices]) for indices in batch["token_indices"]])
                    input("\n wait for next example")

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if batch_idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("\tbatch num:", batch_idx, " average loss:", print_loss_avg)
                print_loss_total = 0

        if debug_flag:
            input("-"*20)

        return 0

    def train_distill_epoch(self,
                            lstm_batches,
                            tokenizer,
                            debug_flag=False,
                            print_every=20):

        self.lstm_classifier.train()

        print_loss_total = 0
        for batch_idx, batch in enumerate(lstm_batches):
            self.adamOpt.zero_grad()

            lstm_output = self.lstm_classifier(batch["token_indices"])

            loss = self.distill_criterion(lstm_output, batch["teacher_pred_scores"].to(self.device))

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if batch_idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("\tbatch num:", batch_idx, " average loss:", print_loss_avg)
                print_loss_total = 0

            if debug_flag:
                print("="*20)
                print("lstm output:", lstm_output)
                print("teacher score:", batch["teacher_pred_scores"])
                print("gold flag:", batch["gold_label_flag"])
                print("target:", batch["target"])
                input("-"*20)

        return 0

    def train_distill_rule_feature_epoch(self, lstm_batches, ce_loss_coef, debug_flag = False, print_every = 20):
        self.distillation_ce_loss_coef = ce_loss_coef

        self.lstm_classifier.train()

        if debug_flag:
            print("="*20)

        print_loss_total = 0
        for batch_idx, batch in enumerate(lstm_batches):
            self.adamOpt.zero_grad()

            lstm_output = self.lstm_classifier(batch["token_indices"])

            # This is fixed to 1 recognize the labeled and unlabeled data easier.
            assert(len(batch["target"]) == 1)

            if batch["gold_label_flag"][0] == 1:  # if it is 1, it means it has the gold label.
                # If the instance has a gold label, the loss does not have to consider the coefficient.
                loss = self.criterion(lstm_output, batch["target"])

            else:
                # This is the situation where the instance is unlabeled.
                loss = self.criterion(lstm_output, batch["target"]) * self.distillation_ce_loss_coef

            if debug_flag:
                print("\tlstm output:", lstm_output, " target:", batch["target"], " gold label flag:", batch["gold_label_flag"])

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if batch_idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("\tbatch num:", batch_idx, " average loss:", print_loss_avg)
                print_loss_total = 0

        if debug_flag:
            input("-"*20)

        return 0

    def eval_epoch(self, batches, tokenizer, debug_flag=False):
        self.lstm_classifier.eval()

        if debug_flag:
            print("-"*20)

        label_list = []
        pred_list = []
        logits_list = []
        time_record = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                self.adamOpt.zero_grad()

                if self.device != torch.device("cpu"):  # This is important!
                    torch.cuda.synchronize()
                start_time = time.time()
                output = self.lstm_classifier(batch["token_indices"])

                if self.device != torch.device("cpu"):  # This is important!
                    torch.cuda.synchronize()
                end_time = time.time()
                time_record.append(end_time - start_time)

                pred_labels = torch.max(output, dim=1)[1]
                logits_list.append(output)

                if debug_flag:
                    print("\tlstm output:", output, " pred:", pred_labels, " label:", batch["target"])

                    if batch_idx<3:
                        print("\tseg1  tokens:", batch["seg1_tokens"])
                        print("\tseg2  tokens:", batch["seg2_tokens"])
                        print("\tseg3  tokens:", batch["seg3_tokens"])
                        print("\tindices:", batch["token_indices"])
                        print("\treconstructed token:",
                              [tokenizer.convert_ids_to_tokens([token_index.item() for token_index in indices])
                               for indices in batch["token_indices"]])
                        input("\n wait for next example")

                # print(pred_labels, batch["target"])
                # print(pred_labels==batch["target"])
                # input("-------")

                # n_hit += torch.sum(pred_labels==batch["target"]).detach().cpu().numpy()
                # n_total_sample += len(batch["target"])

                label_list.extend(batch["target"].cpu().tolist())
                pred_list.extend(pred_labels.detach().cpu().tolist())

            precision, recall, f1 = self.calculate_p_r_f1(label_list, pred_list)
            print("\tp:", precision, "r:", recall, "f1:", f1)

        if debug_flag:
            input("-"*20)

        #print(logits_list)
        # for i in range(len(label_list)):
        #     print("label:", label_list[i], " pred:", pred_list[i])

        return f1, label_list, pred_list, time_record


    def eval_manual(self):


        return 0

    @classmethod
    def calculate_p_r_f1(cls, label_list, pred_list):
        tp = 0
        fp = 0
        fn = 0

        smooth = 1e-7

        for i in range(len(label_list)):
            if pred_list[i]!=0 and label_list[i]==pred_list[i]:
                tp +=1

            if pred_list[i]!=0 and label_list[i]!=pred_list[i]:
                fp +=1

            if pred_list[i]==0 and label_list[i]!=pred_list[i]:
                fn +=1

        precision = tp/(tp+fp+smooth)
        recall = tp/(tp+fn+smooth)
        f1 = 2*precision*recall/(precision + recall+smooth)

        return precision, recall, f1

