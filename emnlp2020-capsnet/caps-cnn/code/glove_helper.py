from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import collections
import zipfile

import numpy as np

from common import vocabulary

def download_glove(output_dir="data"):
    import wget
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # NOTE: these are uncased vectors from 6B tokens from Wikipedia + Gigaword
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    return wget.download(url, out=output_dir)

def archive_line_iter(archive_path, inner_path):
    with zipfile.ZipFile(archive_path) as arx:
        with arx.open(inner_path) as fd:
            for line in fd:
                yield line

def parse_glove_file(archive_path, ndim, vector_path):
    # File path inside archive
    # inner_path = "glove.6B.{:d}d.txt".format(ndim)
    inner_path = vector_path.format(ndim)

    print("Parsing file: {:s}:{:s}".format(archive_path, inner_path))
    # Count lines to pre-allocate memory
    line_count = 0
    for line in archive_line_iter(archive_path, inner_path):
        line_count += 1
    print("Found {:,} words.".format(line_count))
    
    # Pre-allocate vectors as a contiguous array
    # Add three for for <s>, </s>, and <unk>
    W = np.zeros((3+line_count, ndim), dtype=np.float32)
    words = ["<s>", "</s>", "<unk>"]

    print("Parsing vectors... ", end="")
    line_iter = archive_line_iter(archive_path, inner_path)
    for i, line in enumerate(line_iter):
        word, numbers = line.split(maxsplit=1)
        words.append(word.decode('utf-8'))
        W[3+i] = np.fromstring(numbers, dtype=np.float32, sep=" ")
    
    print("Done! (W.shape = {:s})".format(str(W.shape)))
    return words, W


class Hands(object):
    """Helper class to manage GloVe vectors."""
    
    _AVAILABLE_DIMS = { 50, 100, 200, 300 }

    def __init__(self, vector_zip, ndim=300):
        assert(ndim in self._AVAILABLE_DIMS)

        self.vocab = None
        self.W = None
        # self.zipped_filename = "data/glove/glove.6B.zip"
        self.zipped_filename = vector_zip

        # Download datasets
        if not os.path.isfile(self.zipped_filename):
            data_dir = os.path.dirname(self.zipped_filename)
            print("Downloading GloVe vectors to {:s}".format(data_dir))
            self.zipped_filename = download_glove(data_dir)
        print("Loading vectors from {:s}".format(self.zipped_filename))

        # Get the actual .vec file inside the archive. Yes it's dumb.
        #vec_filename = self.zipped_filename.split("/")[-1][:-4]+".vec"
        vec_filename = self.zipped_filename.split("/")[-1][:-4]+".vec"

        words, W = parse_glove_file(self.zipped_filename, ndim, vec_filename)
        # Set nonzero value for special tokens
        mean_vec = np.mean(W[3:], axis=0)
        for i in range(3):
            W[i] = mean_vec
        self.W = W
        self.vocab = vocabulary.Vocabulary(words[3:])
        assert(self.vocab.size == self.W.shape[0])

    @property
    def shape(self):
        return self.W.shape

    @property
    def nvec(self):
        return self.W.shape[0]

    @property
    def ndim(self):
        return self.W.shape[1]
    
    def get_vector(self, word, strict=True):
        """Get the vector for a given word. If strict=True, will not replace 
        unknowns with <unk>."""
        if strict: 
            assert word in self.vocab, "Word '{:s}' not found in vocabulary.".format(word)
        id = self.vocab.word_to_id.get(word, self.vocab.UNK_ID)
        assert(id >= 0 and id < self.W.shape[0])
        return self.W[id]

    def __contains__(self, word):
        return word in self.vocab

    def __getitem__(self, word):
        return self.get_vector(word)
