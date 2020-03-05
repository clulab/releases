import torch
from torch.nn import functional as F
from torch import nn

class PositionalEmbeddings(nn.Module):
    def __init__(self, emb_size, max_position, pad_index):
        super().__init__()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.emb_size = emb_size
        self.max_position = max_position
        self.pad_index = pad_index
        
        self.embeddings = nn.Embedding(
            num_embeddings=self.max_position,
            embedding_dim=self.emb_size,
            padding_idx=0,
        )

    def forward(self, batch):
        # get positions ignoring pads
        positions = self.get_positions(batch, self.pad_index)
        # get embeddings
        embeddings = self.embeddings(positions)
        #
        return embeddings

    def get_positions(self, batch, pad_index):
        batch_size, sentence_max_length = batch.shape
        # get positions
        positions = torch.arange(1,sentence_max_length+1).expand(batch_size,-1).long().to(self.device)
        # get mask from tensor
        mask = positions*0 + pad_index
        # fill mask
        mask = 1-mask.ne(batch)
        # mask pad words
        positions[mask]=0
        #
        return positions

#model = PositionalEmbeddings(5, 100, 0)
