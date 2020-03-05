import numpy as np
import pickle
import re
import torch
#from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import nn
from torch.nn import functional as F

from bert_features import from_tensor_list_to_one_tensor
from char_embeddings import CharEmbeddings, CNNCharEmbeddings
from embeddings import *
from positional_embeddings import PositionalEmbeddings
from utils import first
import networkx as nx
from collections import defaultdict


class Pat(nn.Module):

    def __init__(self, args, word_vocab, tag_vocab, pos_vocab, deprel_vocab, char_vocab):
        super().__init__()

        self.i=0
        self.args = args
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.pos_vocab = pos_vocab
        self.deprel_vocab = deprel_vocab
        self.char_vocab = char_vocab

        self.device = torch.device(f'cuda:{args.which_cuda}' if torch.cuda.is_available() else 'cpu')

        self.glove_emb = args.glove_emb
        self.word_emb_size = args.word_emb_size
        self.tag_emb_size = args.tag_emb_size
        self.bilstm_hidden_size = args.bilstm_hidden_size
        self.bilstm_num_layers = args.bilstm_num_layers

        self.mlp_output_size = args.mlp_output_size

        self.bilstm_input_size = self.word_emb_size + self.tag_emb_size
        # char embeddings
        self.char_emb = args.char_emb
        self.char_emb_hidden_size = args.char_emb_hidden_size
        self.char_emb_size = args.char_emb_size
        # elmo
        #self.elmo_opts = args.elmo_opts
        #self.elmo_weights = args.elmo_weights
        # position embeddings
        self.position_emb_max_pos = args.position_emb_max_pos
        self.position_emb = args.position_emb
        self.position_emb_size = args.position_emb_size
        # bert
        self.bert = args.bert
        self.bert_hidden_size = args.bert_hidden_size
        #polyglot embeddings
        self.polyglot = args.polyglot
        self.polyglot_size = 64 # pretrained model has standard length 64 (no other variants)
        #cnn char encoding
        self.cnn_ce = args.cnn_ce
        self.cnn_embeddings_size = args.cnn_embeddings_size
        self.cnn_ce_kernel_size = args.cnn_ce_kernel_size
        self.cnn_ce_out_channels = args.cnn_ce_out_channels

        # Use head for predicting label
        self.use_head = args.use_head

        #dropout
        self.dropout = nn.Dropout(p=args.dropout)
        self.bilstm_dropout = args.bilstm_dropout

        self.mlp_hidden_size = args.mlp_hidden_size

        self.partofspeech_type = args.part_of_speech

        self.nr_of_cycles = 0

        self.loss_weight_factor = args.loss_weight_factor


        if self.polyglot:
            # Load embeddings
            words, embeddings = pickle.load(open(self.polyglot, 'rb'), encoding='latin1')

            # build a dictionary for fast access
            self.polyglot_dictionary = {}
            for i, w in enumerate(words):
                self.polyglot_dictionary[w]=embeddings[i]

            # Digits are replaced with # in this embedding
            self.polyglot_digit_transformer = re.compile('[0-9]', re.UNICODE)

            # Increase input size accordingly
            self.bilstm_input_size += self.polyglot_size


        if self.bert:
            self.bilstm_input_size = self.bilstm_input_size + self.bert_hidden_size

        if self.position_emb:
            self.bilstm_input_size = self.bilstm_input_size + self.position_emb_size

        # sum output of char embedding to bilstm
        # it is *2 bec a bilstm is used
        if self.char_emb:
            self.bilstm_input_size = self.bilstm_input_size + self.char_emb_hidden_size*2

        if self.cnn_ce:
            self.bilstm_input_size += self.cnn_ce_out_channels

        # if elmo files are set
        #if self.elmo_opts:
        #    print('using elmo')
        #    self.elmo = Elmo(
        #        self.elmo_opts,
        #        self.elmo_weights,
        #        num_output_representations=1,
        #        dropout=0
        #    ).to(self.device)
        #    # increace size of embedding with elmo's size
        #    self.bilstm_input_size = self.bilstm_input_size + self.elmo.get_output_dim()

        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.word_vocab),
            embedding_dim=self.word_emb_size,
            padding_idx=self.word_vocab.pad,
        )

        if self.position_emb:
            self.positional_embedding = PositionalEmbeddings(
                emb_size=self.position_emb_size,
                max_position=self.position_emb_max_pos,
                pad_index=self.word_vocab.pad
            ).to(self.device)

        self.char_embedding = CharEmbeddings(
            char_vocab=self.char_vocab,
            embedding_dim=self.char_emb_size,
            hidden_size=self.char_emb_hidden_size
        ).to(self.device)

        self.cnn_char_embedding = CNNCharEmbeddings(
            char_vocab=self.char_vocab,
            cnn_embeddings_size=self.cnn_embeddings_size,
            cnn_ce_kernel_size=self.cnn_ce_kernel_size,
            cnn_ce_out_channels=self.cnn_ce_out_channels,
            which_cuda=args.which_cuda
        ).to(self.device)

        # if glove is defined
        if self.glove_emb:
            # load glove
            glove = load_glove(self.glove_emb)
            # convert to matrix
            glove_weights = from_vocab_to_weight_matrix(self.word_vocab, glove)
            # load matrix into embeds
            self.word_embedding.load_state_dict({'weight': glove_weights})

        self.tag_embedding = nn.Embedding(
            num_embeddings=len(self.tag_vocab),
            embedding_dim=self.tag_emb_size,
            padding_idx=self.tag_vocab.pad,
        )

        self.bilstm = nn.LSTM(
            input_size=self.bilstm_input_size,
            hidden_size=self.bilstm_hidden_size,
            num_layers=self.bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.bilstm_dropout
        )

        self.bilstm_to_hidden1 = nn.Linear(
            in_features=self.bilstm_hidden_size * 2,
            out_features=self.mlp_hidden_size,
        )

        self.hidden1_to_hidden2 = nn.Linear(
            in_features=self.mlp_hidden_size,
            out_features=self.mlp_output_size,
        )

        self.hidden2_to_pos = nn.Linear(
            in_features=self.mlp_output_size,
            out_features=len(self.pos_vocab),
        )

        self.hidden2_to_dep = nn.Linear(
            in_features=self.mlp_output_size * 2 if self.use_head else self.mlp_output_size, # Depending on whether the head is used or not
            out_features=len(self.deprel_vocab),
        )

        # init embeding weights only if glove is not defined
        if self.glove_emb == None:
            nn.init.xavier_normal_(self.word_embedding.weight)
        nn.init.xavier_normal_(self.tag_embedding.weight)
        nn.init.xavier_normal_(self.bilstm_to_hidden1.weight)
        nn.init.xavier_normal_(self.hidden1_to_hidden2.weight)
        nn.init.xavier_normal_(self.hidden2_to_pos.weight)
        nn.init.xavier_normal_(self.hidden2_to_dep.weight)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def load(name, device):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            print(params)
            params[0].which_cuda = device.index
            pat = Pat(*params)
            pat.load_state_dict(torch.load(f'{name}.model', map_location=device), strict=True)
        return pat

    def save(self, name):
        torch.save(self.state_dict(), f'{name}.model')
        with open(f'{name}.pickle', 'wb') as f:
            params = (self.args, self.word_vocab, self.tag_vocab, self.pos_vocab, self.deprel_vocab, self.char_vocab)
            pickle.dump(params, f)

    # receives an array of Conll objects
    # returns:
    # 1. a matrix word indexes padded
    # 2. a matrix of tag indexes padded
    # 3. an array with sentence lengths
    def sentence2tok_tags(self, sentences):
        w = [[e.norm for e in sentence] for sentence in sentences]
        w, x_lengths = self.prepare(w, self.word_vocab)
        t = [[e.get_partofspeech_tag(self.partofspeech_type) for e in sentence] for sentence in sentences]
        t, _ = self.prepare(t, self.tag_vocab)
        return w, t, x_lengths

    def train_conll(self, sentences):
        # get targets from sentences
        y1 = [[str(e.pos) for e in sentence] for sentence in sentences]
        y1, _ = self.prepare(y1, self.pos_vocab)

        y2 = [[str(e.deprel) for e in sentence] for sentence in sentences]
        y2, _ = self.prepare(y2, self.deprel_vocab)

        # flatten array
        y1 = y1.contiguous().view(-1)
        y2 = y2.contiguous().view(-1)

        # get preds from network
        y_pred1, y_pred2 = self.forward(sentences)
        # flatten preds
        y_pred1 = y_pred1.contiguous().view(-1, len(self.pos_vocab))

        y_pred2 = y_pred2.contiguous().view(-1, len(self.deprel_vocab))
        loss1 = F.cross_entropy(y_pred1, y1, ignore_index=self.pos_vocab.pad)
        loss2 = F.cross_entropy(y_pred2, y2, ignore_index=self.deprel_vocab.pad)

        return loss1+self.loss_weight_factor*loss2

    # graph -> graph with no cycles
    # head_values -> a topk result applied on a 1d tensor. pair of original values and indices (therefore it is sorted in descendent order)
    # Returns the index of first value from head_values which when added to graph doesn't add a cycle, entry_id and pos_value
    def first_not_cycle(self, graph, head_values, entry_id, sentence_length):
        debug=[]
        _, indices = head_values
        for value in indices:
            word=self.pos_vocab[int(value)]
            if word != '<pad>':
                copy_graph = graph.copy()
                pos_value = 0 if word == '<unk>' else int(word)
                if 0 <= entry_id + pos_value < sentence_length:
                    copy_graph.add_edge(entry_id + pos_value if pos_value != 0 else 0, entry_id)
                    if len(list(nx.simple_cycles(copy_graph))) == 0:
                        return entry_id + pos_value if pos_value != 0 else 0, entry_id, pos_value
                debug.append(f"Tried to add  {entry_id + pos_value}, but it adds a cycle or is outside {entry_id + pos_value}")


        # Not possible to add an edge without creating a cycle. This should not be possible
        print(graph.edges())
        print(graph.nodes)
        print(head_values)
        print(entry_id)
        print(debug)
        raise ValueError("Not possible to add an edge without creating a cycle. This should not be possible, because E=V+1 (for this particular problem) and there are V^2 possible edges to add")


    # Receives a matrix which is similar to an adjacency matrix, but the weights represent the probability.
    # Take the one that is most likely to be the root first
    # Returns a dictionary with the most likely tree
    def optimal_no_cycles(self, sentence, y):

        G = nx.DiGraph()
        sentence_length = len(sentence)

        for j, entry in enumerate(sentence):
            # Construct graph, ignoring pads
            # if j != root[0] and entry.id != 0: # Do not add root or fake-root yet. Add every node with no root at the end
            if entry.id != 0:
                for k, probability in enumerate(y[j]):
                    word = self.pos_vocab[k]
                    if word != '<pad>': # Skip pad
                        pos = int(word) if word != '<unk>' else 0
                        if 0 <= entry.id + pos < sentence_length: # make sure is between limits
                            G.add_edge(entry.id + pos if pos != 0 else 0, entry.id, weight=probability)

        edmond = nx.algorithms.tree.branchings.Edmonds(G)
        result = list(edmond.find_optimum(style='arborescence', kind='max').edges())
        result = [(x[1], x[0]) for x in result]
        result = dict(result)

        return result

    # Produce the results
    # Results can be with cycles or not.
    # Two strategies for removing cycles: greedy and optimal
    def parse_conll(self, sentences):
        y1, y2 = self.forward(sentences)
        max = y1.shape[2]
        self.i = 0
        if self.mode == 'evaluation' and self.no_cycles:  # self.no_cycles is dynamically added in predict.py. self.mode is dynamically added in predict.py or train,py
            # For each word in each line ([i][j]), use y1[i][j].topk(<length_of_tensor>) which returns a pair of two tensors: sorted values and indices in original array
            y2 = torch.argmax(y2, dim=2)
            # if running on cuda, copy y from gpu memmory to cpu
            if next(self.parameters()).is_cuda:
                y1 = y1.cpu()
                y2 = y2.cpu()
            y2 = y2.numpy()
            if self.no_cycles_strategy == 'greedy':
                for i, sentence in enumerate(sentences):
                    sentence_length = len(sentence)
                    G = nx.DiGraph()
                    result = defaultdict(list)
                    # iterate in descending order, from the word for which there is the most confident output
                    for index_in_array in torch.max(y1[i][:sentence_length], 1)[0].sort(descending=True)[1]:
                        entry = sentence[index_in_array]
                        values = y1[i][index_in_array].topk(max)
                        if entry.id != 0:
                            edge_no_cycle = self.first_not_cycle(G, values, entry.id, sentence_length)
                            result[int(index_in_array)] = list(edge_no_cycle)
                            G.add_edge(edge_no_cycle[0], edge_no_cycle[1])
                        else:
                            result[int(index_in_array)] = ['_', '_', '_']  # head of entry with id = 0 is 0 (itself)

                    for j, entry in enumerate(sentence):
                        deprel = self.deprel_vocab[int(y2[i, j])]
                        entry.head = result[j][0]
                        entry.pos = result[j][2]
                        entry.deprel = deprel
            elif self.no_cycles_strategy == 'optimal':  # Uses Liu-Chen-Edmonds algorithm
                for i, sentence in enumerate(sentences):
                    torch.set_printoptions(threshold=5000)
                    result = self.optimal_no_cycles(sentence, y1[i])
                    for j, entry in enumerate(sentence):
                        deprel = self.deprel_vocab[int(y2[i, j])]
                        if entry.id not in result:
                            entry.head = entry.id

                        else:
                            entry.head = result[entry.id]
                        entry.pos = '_'
                        entry.deprel = deprel

        else:

            # After it was checked that ind is not a pad
            def get_pos(ind):
                word_of_ind = self.pos_vocab[int(ind)]
                if word_of_ind != '<pad>':
                    pos_of_ind = 0 if word_of_ind == '<unk>' else int(word_of_ind)
                    return pos_of_ind

                raise ValueError("No index is valid. Maybe there is a mistake somewhere else?")

            y2 = torch.argmax(y2, dim=2)

            # if running on cuda, copy y from gpu memmory to cpu
            if next(self.parameters()).is_cuda:
                y1 = y1.cpu()
                y2 = y2.cpu()

            y2 = y2.numpy()
            for i, sentence in enumerate(sentences):
                sentence_length = len(sentence)
                for j, entry in enumerate(sentence):
                    # Skip over 'fake' root
                    if entry.id != 0:
                        # Indices from biggest to smallest (total = k) from y1[i][j] (current entry)
                        y1_topk = y1[i][j].topk(max)[1]
                        # Take first index (descending order) that is not '<pad>' and inside sentence
                        index = first(y1_topk, lambda ind: self.pos_vocab[int(ind)] != '<pad>' and (0 <= (entry.id + get_pos(ind)) < sentence_length))

                        pos = get_pos(index)
                        entry.head = entry.id + pos if pos != 0 else 0
                        entry.pos = pos
                    else:
                        entry.pos = 0
                        entry.head = 0

                    deprel = self.deprel_vocab[int(y2[i, j])]
                    entry.deprel = deprel

        # Printing the number of cycles. Only on evaluation
        if self.mode == 'evaluation':
            if self.print_nr_of_cycles:
                for sentence in sentences:
                    G = nx.DiGraph()
                    for entry in sentence:
                        if entry.id != 0: # Might be that root.head is equal to root.id (fake root, with id=0)
                            G.add_edge(entry.head, entry.id)
                    if len(list(nx.simple_cycles(G))) > 0:
                        self.nr_of_cycles += 1


    def prepare(self, sentences, vocab):
        x = [torch.tensor([vocab[w] for w in sentence]).to(self.device) for sentence in sentences]
        x_lengths = np.array([len(sentence) for sentence in x])
        padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        return padded_x, x_lengths

    #def get_elmo_embeddings(self, orig_w):
    #    # elmo uses non-normalized words
    #    #w = [[e.form for e in sentence] for sentence in sentences]
    #    # elmo from batch/sentence to batch/character_ids
    #    elmo_ids = batch_to_ids(orig_w).to(self.device)
    #    # get elmo_embeddings
    #    elmo_embeds = self.elmo(elmo_ids)
    #    # elmo_embeds is a dict with elmo_representations and mask
    #    # first dimention contains the output of each elmo layer (in this project 1 layer)
    #    elmo_embeds = elmo_embeds['elmo_representations'][0]
    #    # (batch_size, seq_len, embedding_dim)
    #    return elmo_embeds

    def get_polyglot_embedding(self, word):
        # print("Got word: ", word, "\n")
        if word.isdigit():
            processed = self.polyglot_digit_transformer.sub('#', word)
            if processed in self.polyglot_dictionary:
                return self.polyglot_dictionary[self.polyglot_digit_transformer.sub('#', word)]
            else:
                return self.polyglot_dictionary['<UNK>']
        elif word not in self.polyglot_dictionary:
            return self.polyglot_dictionary['<UNK>']
        else:
            return self.polyglot_dictionary[word]


    def get_polyglot_embeddings(self, orig_w):
        return [[torch.tensor(self.get_polyglot_embedding(word)) for word in sentence] for sentence in orig_w]

    def forward(self, sentences):
        orig_w = [[e.form for e in sentence] for sentence in sentences]

        w, t, x_lengths = self.sentence2tok_tags(sentences)

        batch_size, seq_len = w.size()
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        we = self.word_embedding(w)
        t = self.tag_embedding(t)

        if self.bert:
            # get bert features from model
            bert_features_list = [[e.bert for e in sentence] for sentence in sentences]
            # convert list to one tensor
            bert_features_tensor = from_tensor_list_to_one_tensor(bert_features_list, self.bert_hidden_size).to(self.device)
            # concat the tensor with the the rest of the word embeddings
            we = torch.cat((bert_features_tensor, we), 2)

        if self.position_emb:
            # get positional embeddings
            position = self.positional_embedding(w)
            # concat positional embeddings with word embeddings

            we = torch.cat((position, we), 2)

        # concat tags embeddings and word embeddings
        x = torch.cat((we, t), 2)

        #if self.elmo_opts:
        #    elmo_embeds = self.get_elmo_embeddings(orig_w)
        #    x = torch.cat([x, elmo_embeds], 2)

        if self.char_emb:
            c = self.char_embedding(orig_w)
            x = torch.cat([x, c], 2)

        if self.cnn_ce:
            c = self.cnn_char_embedding(orig_w)
            x = torch.cat([x, c], 2)

        if self.polyglot:
            polyglot_features_list = self.get_polyglot_embeddings(orig_w)
            polyglot_features_tensor = from_tensor_list_to_one_tensor(polyglot_features_list, self.polyglot_size).to(self.device)
            x = torch.cat([x, polyglot_features_tensor], 2)

        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, n_lstm_units)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x, _ = self.bilstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # (batch_size, seq_len, n_lstm_units) -> (batch_size * seq_len, n_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.bilstm_to_hidden1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.hidden1_to_hidden2(x)
        x = F.relu(x)

        y1 = self.hidden2_to_pos(x)
        # print(x.shape)
        if self.mode == 'training':
            if self.use_head:
                heads = [[e.head for e in sentence] for sentence in sentences]
                maximum = max([len(z) for z in heads])
                heads = [z + (maximum - len(z)) * [0] for z in heads]  # pads
                heads = torch.tensor(heads)

                heads = heads.view(1, -1)[0].to(self.device)

                # each offset is of length (seq_len) in the end
                # Creates offsets: [0,0,0..,0], [seq_length, seq_length, .., seq_length], [2*seq_length, 2*seq_length, .., 2*seq_length] etc that are used for fast access in a tensor of shape (batch_size * seq_length, pos_hidden_size)
                offsets = (torch.arange(batch_size).repeat(seq_len).view(seq_len, batch_size).transpose(0,1) * seq_len).contiguous().view(1,-1)[0].to(self.device)
                indices = heads + offsets

                heads = x[indices]


        elif self.mode == 'evaluation':
            if self.use_head:
                # (batch_size * seq_len, n_tags)
                ids = [[e.id for e in sentence] for sentence in sentences]
                maximum = max([len(z) for z in ids])
                ids = [z + (maximum - len(z)) * [0] for z in ids]
                ids = torch.tensor(ids).to(self.device)
                ids = ids.view(1, -1)[0]

                heads = torch.zeros(ids.shape[0]).long().to(self.device)

                maxes = torch.argmax(y1, dim=1)

                offsets = (torch.arange(batch_size).repeat(seq_len).view(seq_len, batch_size).transpose(0,1) * seq_len).contiguous().view(1,-1)[0].to(self.device)
                for i in range(heads.shape[0]):
                    if ids[i] != 0:
                        word = self.pos_vocab[int(maxes[i])]
                        pos = 0 if word == '<unk>' else int(word)
                        heads[i] = (0 if pos+ids[i] > maximum else torch.clamp(pos + ids[i], min=0)) if pos != 0 else 0
                    else:
                        heads[i] = 0
                indices = heads + offsets
                heads = x[indices]
        else:
            exit("Unknown mode")

        if self.use_head:
            x = torch.cat([x,heads], 1)

        y2 = self.hidden2_to_dep(x)


        if self.mode == 'evaluation':
            y1 = F.softmax(y1, dim=1)
            y2 = F.softmax(y2, dim=1)


        # (batch_size * seq_len, n_lstm_units) -> (batch_size, seq_len, n_tags)
        y1 = y1.view(batch_size, seq_len, len(self.pos_vocab))
        y2 = y2.view(batch_size, seq_len, len(self.deprel_vocab))

        return y1, y2
