
import sys
import math
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import export, parameter_count



@export
def simple_MLP_embed_RTE(word_vocab_size, num_classes, wordemb_size, pretrained=True, word_vocab_embed=None, hidden_size=200, update_pretrained_wordemb=False):

    model = FeedForwardMLPEmbed_RTE(word_vocab_size, wordemb_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model

class FeedForwardMLPEmbed_RTE(nn.Module):
    def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz, word_vocab_embed, update_pretrained_wordemb):
        super().__init__()
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(word_vocab_size, embedding_size)
        print(f"inside architectures.py line 26 at 1 value of self.embeddings.weight is {self.embeddings.weight.shape} ")


        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222
        if word_vocab_embed is not None:  # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            print("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            print(f"at 2 value of self.embeddings.weight is {self.embeddings.weight} ")

            if update_pretrained_wordemb is False:
                # NOTE: do not update the emebddings
                # https://discuss.pytorch.org/t/how-to-exclude-embedding-layer-from-model-parameters/1283
                print("NOT UPDATING the word embeddings ....")
                self.embeddings.weight.detach_()
            else:
                print("UPDATING the word embeddings ....")
                print(f"at 2 value of self.embeddings.weight is {self.embeddings.weight} ")



        #todo: pass them from somewhere...maybe command line or config file
        self.NUM_CLASSES = 3
        self.CODE_PRCT_DROPOUT, self.COMM_PRCT_DROPOUT = 0.1, 0.1
        self.CODE_HD_SZ, self.COMM_HD_SZ = 50,50
        self.CODE_NUM_LAYERS, self.COMM_NUM_LAYERS = 2, 2


        # Creates a bidirectional LSTM for the code input
        self.lstm = nn.LSTM(embedding_size,  # Size of the code embedding
                                 self.CODE_HD_SZ,  # Size of the hidden layer
                                 num_layers=self.CODE_NUM_LAYERS,
                                 dropout=self.CODE_PRCT_DROPOUT,
                                 batch_first=True,
                                 bidirectional=True)

        # Size of the concatenated output from the 2 LSTMs
        self.CONCAT_SIZE = (self.CODE_HD_SZ + self.COMM_HD_SZ) * 2

        # FFNN layer to transform LSTM output into class predictions
        self.lstm2hidden = nn.Linear(self.CONCAT_SIZE, 50)
        self.hidden2label = nn.Linear(50, self.NUM_CLASSES)

        #todo: might have to add a softmax. look at what loss function you are using.-CrossEntropyLoss




    def forward(self, claim, evidence, claim_lengths, evidence_lengths):

        # keep track of how code and comm were sorted so that we can unsort them later
        # because packing requires them to be in descending order
        claim_lengths, claim_sort_order = claim_lengths.sort(descending=True)
        evidence_lengths, ev_sort_order = evidence_lengths.sort(descending=True)
        claim_inv_order = claim_sort_order.sort()[1]
        ev_inv_order = ev_sort_order.sort()[1]

        # Encode the batch input using word embeddings
        claim_encoding = self.embeddings(claim[claim_sort_order])
        ev_encoding = self.embeddings(evidence[ev_sort_order])

        # pack padded input
        claim_enc_pack = torch.nn.utils.rnn.pack_padded_sequence(claim_encoding, claim_lengths, batch_first=True)
        evidence_enc_pack = torch.nn.utils.rnn.pack_padded_sequence(ev_encoding, evidence_lengths, batch_first=True)

        # Run the LSTMs over the packed input
        #:claim_h_n hidden states at each word- will be used later when we have to get output of bilstm.

        #claim_c_n = context states at each word
        claim_enc_pad, (claim_h_n, claim_c_n) = self.lstm(claim_enc_pack)
        ev_enc_pad, (ev_h_n, ev_c_n) = self.lstm(evidence_enc_pack)

        # back to padding
        code_vecs, _ = torch.nn.utils.rnn.pad_packed_sequence(claim_enc_pad, batch_first=True)
        comm_vecs, _ = torch.nn.utils.rnn.pad_packed_sequence(ev_enc_pad, batch_first=True)

        # Concatenate the final output from both LSTMs
        # therefore claim_h_n[0]= hidden states at the end of forward lstm pass.
        # therefore claim_h_n[1]= hidden states at the end of backward lstm pass.
        recurrent_vecs = torch.cat((claim_h_n[0, claim_inv_order], claim_h_n[1, claim_inv_order],
                                    ev_h_n[0, ev_inv_order], ev_h_n[1, ev_inv_order]), 1)

        # Transform recurrent output vector into a class prediction vector
        y = F.relu(self.lstm2hidden(recurrent_vecs))
        y = self.hidden2label(y)

        return y

@export
def da_RTE(word_vocab_size, num_classes, wordemb_size, pretrained=True, word_vocab_embed=None, hidden_size=200, update_pretrained_wordemb=False,para_init=0.1,use_gpu=False):

    model = DecompAttnLibowenCode(word_vocab_size, wordemb_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb,para_init,num_classes,use_gpu)
    return model

class DecompAttnLibowenCode(nn.Module):
    # num_embeddings, embedding_size, hidden_size, para_init):

    def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz, word_vocab_embed,
                 update_pretrained_wordemb,para_init,num_classes,use_gpu):

        super(DecompAttnLibowenCode, self).__init__()

        self.input_encoder = encoder(word_vocab_size, embedding_size, hidden_sz,para_init)
        self.input_encoder.embedding.weight.data.copy_(torch.from_numpy(word_vocab_embed))
        self.input_encoder.embedding.weight.requires_grad = update_pretrained_wordemb
        self.inter_atten = atten(hidden_sz, num_classes, para_init)

        device = None
        if (use_gpu) and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.input_encoder.to(device)
        self.inter_atten.to(device)

        self.para1 = filter(lambda p: p.requires_grad, self.input_encoder.parameters())
        self.para2 = self.inter_atten.parameters()


    def forward(self, claim, evidence, claim_lengths, evidence_lengths):
        train_src_linear, train_tgt_linear = self.input_encoder(
            claim, evidence)
        log_prob = self.inter_atten(train_src_linear, train_tgt_linear)
        return log_prob


class encoder(nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_size, para_init):
        super(encoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.para_init = para_init

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''

        batch_size = sent1.size(0)
        sent1 = self.embedding(sent1)
        sent2 = self.embedding(sent2)



        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        sent1_linear = self.input_linear(sent1).view(
            batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(
            batch_size, -1, self.hidden_size)

        return sent1_linear, sent2_linear

class atten(nn.Module):
    '''
        intra sentence attention
    '''

    def __init__(self, hidden_size, label_size, para_init):
        super(atten, self).__init__()

        self.hidden_size = hidden_size
        self.label_size = label_size
        self.para_init = para_init

        self.mlp_f = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size)

        self.final_linear = nn.Linear(
            self.hidden_size, self.label_size, bias=True)

        self.log_prob = nn.LogSoftmax()

        '''initialize parameters'''
        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''attend'''

        f1 = self.mlp_f(sent1_linear.view(-1, self.hidden_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.hidden_size))

        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.hidden_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.hidden_size))
        g1 = g1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        h = self.final_linear(h)

        # print 'final layer'
        # print h.data

        log_prob = self.log_prob(h)

        return log_prob
