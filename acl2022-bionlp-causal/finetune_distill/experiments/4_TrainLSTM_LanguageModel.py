import os
import json
import sys
from pathlib import Path
import math
import numpy as np
import random

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from transformers import BertTokenizer

parent_folder_path = str(Path('.').absolute().parent)
data_folder_path = parent_folder_path+"/DataRaw"
data_processing_path = parent_folder_path+"/DataProcessing"
model_path = parent_folder_path+"/ModelClasses"

sys.path += [parent_folder_path, data_folder_path, data_processing_path, model_path]

from LSTMClass import BiLSTMLM

'''
Modified from: https://github.com/lyeoni/pretraining-for-language-understanding
'''

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# TODO: we should not make the tokenization to be in this Dataset class, but should make it in the PadCollate class
# Becuase in that way we can handle sequences with various lengths.
class Corpus(Dataset):
    def __init__(self, corpus_path, tokenizer, model_type, cuda):
        self.corpus = []
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.cuda = cuda

        with open(corpus_path, 'r', encoding='utf8') as reader:
            for li, line in enumerate(reader):
                self.corpus.append(line.strip())

    def __getitem__(self, index):
        return self.corpus[index]

    def __len__(self):
        return len(self.corpus)


class PadCollate:
    def __init__(self, tokenizer):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.tokenizer = tokenizer

    def pad_collate(self, batch):
        """
        Return inputs, targets tensors used for model training.
        The size of returned tensor is as follows.
        if self.model_type == 'LSTM':
            |inputs|    = (batch_size, max_seq_len-1)
            |targets|   = (batch_size, max_seq_len-1)

        elif self.model_type == 'BiLSTM':
            |inputs|    = (batch_size, max_seq_len)
            |targets|   = (batch_size, max_seq_len-2)
        """

        # The max length of the tokenized sequence is set to 50 (The block size).
        # This block size and setting is the same as what we had in the transformer experiment.
        tokens_indices = self.tokenizer(batch,  # tokenize the list of lines in this batch.
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=50)["input_ids"]
        tokens_indices = tokens_indices.cuda()

        # The return format should be (input, target)
        batch_input = tokens_indices[:, 1:]  # eliminate the [CLS] embedding for lstm language model
        batch_target = tokens_indices[:, 2:-1]  # For BiLSTM, the length of target should be 2 less than the length of input

        return batch_input, batch_target

    def __call__(self, batch):
        return self.pad_collate(batch)


class TrainLSTMLanguageModel:

    '''
    Train the LSTM language model using the code in this repo:
    https://github.com/lyeoni/pretraining-for-language-understanding/blob/master/lm_trainer.py

    Default values of the pre-training:
    $ python lm_trainer.py --train_corpus build_corpus/corpus.train.txt --vocab vocab.train.pkl --model_type BiLSTM --n_layers 1 --multi_gpu
    Namespace(batch_size=512, clip_value=10, cuda=True, dropout_p=0.2, embedding_size=256, epochs=10, hidden_size=1024, is_tokenized=False, max_seq_len=32, model_type='BiLSTM', multi_gpu=True, n_layers=1, shuffle=True, test_corpus=None, tokenizer='mecab', train_corpus='build_corpus/corpus.train.txt', vocab='vocab.train.pkl')
    =========MODEL=========
     DataParallelModel(
      (module): BiLSTMLM(
        (embedding): Embedding(271503, 256)
        (lstm): LSTM(256, 1024, batch_first=True, dropout=0.2, bidirectional=True)
        (fc): Linear(in_features=2048, out_features=1024, bias=True)
        (fc2): Linear(in_features=1024, out_features=512, bias=True)
        (fc3): Linear(in_features=512, out_features=271503, bias=True)
        (softmax): LogSoftmax()
      )
    )
    '''
    vocab_file_dir = "truncate_vocab/pubmed_10000_vocab_by_freq_token_to_original_id.json"

    clip_value = 10
    model_type = "BiLSTM"
    max_seq_len = 50  # This is set to match the sequence length of BERT. If it does not work, set it back later.
    dropout_p = 0.2
    n_layers = 1
    batch_size = 64  # This is enough for minibert to be trained on a 8G GPU.
    cuda = True

    n_epochs = 12  # This is fixed to be 12 to match what we had for BERT.

    embedding_size = 100
    hidden_size = 700   # When hidden size=700, the LSTM1Seg classifier has 5.6M parameters.

    # This uses the same pubmed corpus as MiniBERT.
    train_corpus = "/home/zhengzhongliang/CLU_Projects/2021_MiniBERT_Pubmed/pubmed_10000/pubmed_10000_train.txt"
    test_corpus = "/home/zhengzhongliang/CLU_Projects/2021_MiniBERT_Pubmed/pubmed_10000/pubmed_10000_test.txt"

    save_model_folder_dir = "saved_models_20211229_lstm_language_model/"
    model_folder_path = save_model_folder_dir + "20211215_bilstm_vocab_bert_10000_hidden_" + str(
        hidden_size) + "/"

    if not os.path.exists(model_folder_path):
        model_folder_path_ = Path(model_folder_path)
        model_folder_path_.mkdir(parents=True, exist_ok=True)  # This makes the dir recursively

    @classmethod
    def load_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("truncate_vocab/bert_base_uncased_pubmed_10000/")
        return tokenizer

    @classmethod
    def train_epoch(cls, model, tokenizer, train_loader, optimizer, loss_fn, epoch, debug_flag=False):
        n_batches, n_samples = len(train_loader), len(train_loader.dataset)

        model.train()
        total_loss, total_ppl = 0, 0
        for iter_, batch in enumerate(tqdm(train_loader)):
            inputs, targets = batch
            # |inputs|, |targets| = (batch_size, seq_len), (batch_size, seq_len)

            preds = model(inputs)
            # |preds| = (batch_size, seq_len, len(vocab))

            preds = preds.contiguous().view(-1, len(tokenizer.vocab))
            # |preds| = (batch_size*seq_len, len(vocab))
            # The original code call contiguous after view, but it gives error.
            # According to this post: https://github.com/cezannec/capsule_net_pytorch/issues/4
            # Calling contiguous before view should fix this.

            old_targets = targets  # This stores the targets before reshape. Used for debugging.
            targets = targets.contiguous().view(-1)
            # |targets| = (batch_size*seq_len)

            loss = loss_fn(preds, targets)
            total_loss += loss.item()
            total_ppl += np.exp(loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cls.clip_value)
            optimizer.step()

            if debug_flag:
                print("=" * 40)
                print("input size:", inputs.size())
                print("target size after view:", targets.size())
                print("pred size after view:", preds.size())
                print("input text:", tokenizer.convert_ids_to_tokens(inputs[0]))  # Only check the first example in each batch
                print("output text:", tokenizer.convert_ids_to_tokens(old_targets[0]))
                input("-" * 40)

            if iter_ % (n_batches // 300) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.4f} \tPerplexity: {:.5f}'.format(
                    epoch, iter_, n_batches, 100. * iter_ / n_batches, loss.item(), np.exp(loss.item())))

        print('====> Train Epoch: {} Average loss: {:.4f} \tPerplexity: {:.5f}'.format(
            epoch, total_loss / n_batches, total_ppl / n_batches))

        # Save model
        torch.save(model, cls.model_folder_path + "bilstm_lm_epoch_" + str(epoch))

        return total_ppl/n_batches

    @classmethod
    def eval_epoch(cls, model, tokenizer, test_data_loader, loss_fn):
        '''
        The default evaluation function does not return the loss. So we write our own evaluation function instead.
        :param model:
        :param test_data_loader:
        :param vocab_dict:
        :param loss_fn:
        :return:
        '''
        model.eval()

        total_loss = []
        with torch.no_grad():
            for iter_, batch in enumerate(tqdm(test_data_loader)):
                inputs, targets = batch
                # |inputs|, |targets| = (batch_size, seq_len), (batch_size, seq_len)

                preds = model(inputs)
                # |preds| = (batch_size, seq_len, len(vocab))

                preds = preds.contiguous().view(-1, len(tokenizer.vocab))
                # |preds| = (batch_size*seq_len, len(vocab))

                targets = targets.contiguous().view(-1)
                # |targets| = (batch_size*seq_len)

                loss = loss_fn(preds, targets)

                total_loss.append(loss.detach().cpu().tolist())

        epoch_average_loss = sum(total_loss)/len(total_loss)
        print("Evaluation average batch loss:", epoch_average_loss, " ppl:", math.exp(epoch_average_loss))

        return math.exp(epoch_average_loss)  # return the epoch evaluation perplexity

    @classmethod
    def train_lstm_language_model(cls):
        # Select tokenizer
        tokenizer = cls.load_tokenizer()

        # Build dataloader
        train_loader = DataLoader(dataset=Corpus(corpus_path=cls.train_corpus,
                                                 tokenizer=tokenizer,
                                                 model_type=cls.model_type,
                                                 cuda=cls.cuda),
                                  collate_fn=PadCollate(tokenizer=tokenizer),
                                  batch_size=cls.batch_size,
                                  shuffle=True,
                                  drop_last=True)

        test_loader = DataLoader(dataset=Corpus(corpus_path=cls.test_corpus,
                                                tokenizer=tokenizer,
                                                model_type=cls.model_type,
                                                cuda=cls.cuda),
                                 collate_fn=PadCollate(tokenizer=tokenizer),
                                 batch_size=cls.batch_size,
                                 shuffle=False,
                                 drop_last=True)

        model = BiLSTMLM(input_size=len(tokenizer.vocab),
                         embedding_size=cls.embedding_size,
                         hidden_size=cls.hidden_size,
                         output_size=len(tokenizer.vocab),
                         n_layers=cls.n_layers,
                         dropout_p=cls.dropout_p)

        loss_fn = nn.NLLLoss(ignore_index=tokenizer.vocab[tokenizer.pad_token])
        # This learning rate is given based on my previous experience using LSTM
        # Empirically I think the learning rate should be below 1e-3 but higher than 1e-4
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        model = model.cuda()
        loss_fn = loss_fn.cuda()

        print('=========MODEL=========\n', model)

        # Train
        train_ppl_list = []
        eval_ppl_list = []
        for epoch in range(1, cls.n_epochs + 1):
            train_ppl_epoch = cls.train_epoch(model, tokenizer, train_loader, optimizer, loss_fn, epoch, debug_flag=False)
            eval_ppl_epoch = cls.eval_epoch(model, tokenizer, test_loader, loss_fn)

            train_ppl_list.append(train_ppl_epoch)
            eval_ppl_list.append(eval_ppl_epoch)

        with open(cls.model_folder_path + "perplexity_history.json", "w") as handle:
            json.dump({"train_ppl": train_ppl_list,
                       "eval_ppl": eval_ppl_list}, handle)

TrainLSTMLanguageModel.train_lstm_language_model()
