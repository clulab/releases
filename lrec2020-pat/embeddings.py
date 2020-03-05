import numpy as np
from torch import nn
import torch

# dowwnloaded glove from http://nlp.stanford.edu/data/glove.6B.zip

# receives a path
# returns a dictionary with word -> embeddings
def load_glove(path):
    glove = {}
    with open(path, 'rb') as f:
        for l in f:
            # split lines
            line = l.decode().split()
            # first part is word
            word = line[0]
            # the rest is the embeddings
            vec = np.array(line[1:]).astype(float)
            # feed dict
            glove[word] = vec
    return glove


# receives a @Vocabulary object and a dictionary of embeddings (word -> embeds)
def from_vocab_to_weight_matrix(vocab, embeddings):
    print('creating embeddigns matrix')
    # get the embeds size
    embeddings_size = len(next(iter(embeddings.values())))
    # get vocab size
    vocab_size = len(vocab)
    # create weight matrixs
    weights_matrix = np.zeros((vocab_size, embeddings_size))
    words_not_found = 0

    # create a matrix where each line corresponds to the embeds of ith word
    for word in vocab:
        i = vocab.w2i[word]
        try:
            weights_matrix[i] = embeddings[word]
        except KeyError:
            # if word is not in pretrained embeds, initialize them with random words
            words_not_found += 1
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embeddings_size, ))
    print(words_not_found, 'words not found')
    return torch.from_numpy(weights_matrix)

