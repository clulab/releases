import torch
from torch.nn import functional as F
from torch import nn

class CharEmbeddings(nn.Module):
    def __init__(self, char_vocab, embedding_dim, hidden_size, which_cuda=0):
        super().__init__()

        self.device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else 'cpu')

        self.embedding_dim = embedding_dim
        self.vocab = char_vocab
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(
            num_embeddings = len(self.vocab),
            embedding_dim = embedding_dim,
            padding_idx = self.vocab.pad
        )

        self.bilstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = self.hidden_size,
            num_layers = 1,
            bidirectional = True
        )

    
    def forward(self, sentence_batch):
        # char2index + padding
        words, lengths, unsort_idx = self.prepare(sentence_batch)
        # words -> (n_pad_words, max_word_length)
        # sort + remove 0 lenth words (pads)
        non_zero_words, non_zero_lengths = self.remove_pad_words(words, lengths)

        # non_zero_words -> (n_nonpad_words, max_word_length)
        #
        embeddings = self.embeddings(non_zero_words).to(self.device)
        # embeddings -> (n_nonpad_words, max_word_length, embeddings_dim)
        # pack
        x = torch.nn.utils.rnn.pack_padded_sequence(embeddings, non_zero_lengths, batch_first=True)
        # pass through lstm
        output, hidden = self.bilstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # embeddings -> (n_nonpad_words, max_word_length, hidden_size*2)

        # take the output of the lstm correctly
        # see https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        ## filter idx is the id of the last character in each word
        ## this aims to find the output of the lshtm on the last non pad char
        ## ex: ['h', 'e', 'l', 'l', 'o', '<pad>']
        ## instead of getting the output of '<pad>' we want the output of 'o'
        filter_idx = non_zero_lengths.long().view(-1, 1, 1).expand(-1, 1, self.hidden_size*2) -1
        # filter_idx -> (n_nonpad_words), max_word_length, hidden_size*2)
        forward_out = x.gather(1, filter_idx).squeeze(1)[:,:self.hidden_size]
        # filter_idx -> (n_nonpad_words, hidden_size)

        # get the output of the first character
        backward_out = x[:,0,self.hidden_size:]
        # concat first char's output last hidden state part with the last char's output first hidden state part
        x = torch.cat([forward_out, backward_out], 1)
        # x -> (n_nonpad_words, hidden_size*2)
        #
        x = torch.cat([x, torch.zeros(len(words)-len(non_zero_words), self.hidden_size*2).to(self.device)],0)
        # x -> (n_pad_words, hidden_size*2)
        # unsort
        x = x[unsort_idx]
        # reshape to sentence size
        x = x.view(len(sentence_batch), -1 , self.hidden_size*2)
        # x -> (batch_size, max_sentence_size, hidden_size*2)
        return x
    
    # receibes a batch of sentences and return a batch of words and their lengths sorted in decreasing order
    def prepare(self, sentence_batch):

        # get max word length to pad with empty tensors
        sentence_lengths = [len(s) for s in sentence_batch]
        max_word_length = max([len(w) for s in sentence_batch for w in s])
        # from sentence to words
        # from words to vec of idx
        sentences = [[torch.LongTensor([int(self.vocab.w2i.get(c, self.vocab.unk)) for c in w]) for w in s] for s in sentence_batch]
        # pad with empty tensors
        padded_sentences = [s+(max(sentence_lengths)-len(s))*[torch.Tensor()] for s in sentences]
        # flatten_padded_sentences
        # this is needed to prepare use pad_sequence with words
        flatten_padded_sentences = [w for s in padded_sentences for w in s]
        # get word lengths to pack padded
        word_lengths = [len(w) for w in flatten_padded_sentences]
        # sort
        sort_word_len, sort_idx = torch.LongTensor(word_lengths).to(self.device).sort(0, descending=True)
        #print(flatten_padded_sentences)
        flatten_padded_sentences = [flatten_padded_sentences[i] for i in sort_idx]
        # get unsort_index
        _, unsort_idx = sort_idx.sort(0)
        # padd words
        padded_words = torch.nn.utils.rnn.pad_sequence(flatten_padded_sentences, batch_first=True).to(self.device)

        return padded_words, sort_word_len, unsort_idx
    
    # this function remove pad words from a batch of words
    # note that the index returned is the size of the batch before filtering pads
    # so, before unsorting, padded values must be re-added to the tensor
    def remove_pad_words(self, padded_words, word_lengths):
        
        # count how many zeroes
        num_zeroes = sum([1 if length == 0 else 0 for length in word_lengths])
        # remove zeroes
        padded_words = padded_words[:len(word_lengths)-num_zeroes]
        sort_word_len = word_lengths[:len(word_lengths)-num_zeroes]
        
        return padded_words, sort_word_len
    
#model = CharEmbeddings(char_vocab, 5, 5)

class CNNCharEmbeddings(CharEmbeddings):
    def __init__(self, char_vocab, cnn_embeddings_size, cnn_ce_kernel_size, cnn_ce_out_channels, which_cuda=0):
        torch.backends.cudnn.deterministic = True
        CharEmbeddings.__init__(self, char_vocab, cnn_embeddings_size, 1, which_cuda=which_cuda)

        self.embedding_dim = cnn_embeddings_size
        self.vocab = char_vocab

        self.cnn_ce_kernel_size = cnn_ce_kernel_size
        self.cnn_ce_out_channels = cnn_ce_out_channels

        if self.cnn_ce_kernel_size % 2 != 1:
            exit("Kernel size must be odd")

        self.cnn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.cnn_ce_out_channels,
            kernel_size=cnn_ce_kernel_size,
            stride=1,
            padding=int((cnn_ce_kernel_size-1)/2),
        )

    def forward(self, sentence_batch):

        words, lengths, unsort_idx = self.prepare(sentence_batch)
        non_zero_words, non_zero_lengths = self.remove_pad_words(words, lengths)
        embeddings = self.embeddings(non_zero_words).to(self.device)

        x = self.cnn(embeddings.transpose(1,2))
        mask = (torch.arange(x.shape[2]).expand(x.shape).to(self.device) < non_zero_lengths.unsqueeze(1).unsqueeze(1).to(self.device)).float()
        x_min = (torch.arange(x.shape[2]).expand(x.shape).to(self.device) >= non_zero_lengths.unsqueeze(1).unsqueeze(1).to(self.device)).float() * torch.min(x)
        x_min = x_min.detach().float()
        x = x * mask + x_min
        x = F.relu(x)
        x = nn.MaxPool1d(
            kernel_size=x.shape[2]
        )(x)

        x=x.view([x.shape[0], x.shape[1]])
        x = torch.cat([x, torch.zeros(len(words)-len(non_zero_words), self.cnn_ce_out_channels).to(self.device)],0)

        x = x[unsort_idx]
        x = x.view(len(sentence_batch), -1 , self.cnn_ce_out_channels)

        return x
