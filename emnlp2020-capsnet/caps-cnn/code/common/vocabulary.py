from __future__ import print_function
from __future__ import division

from collections import defaultdict, Counter

from . import constants

class Vocabulary(object):

    START_TOKEN = constants.START_TOKEN
    END_TOKEN   = constants.END_TOKEN
    UNK_TOKEN   = constants.UNK_TOKEN

    def __init__(self, tokens, size=None,
                 progressbar=lambda l:l):
        """Create a Vocabulary object.

        Args:
            tokens: iterator( string )
            size: None for unlimited, or int > 0 for a fixed-size vocab.
                  Vocabulary size includes special tokens <s>, </s>, and <unk>
            progressbar: (optional) progress bar to wrap iterator.
        """
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(lambda: Counter())
        prev_word = None
        for word in progressbar(tokens):  # Make a single pass through tokens
            self.unigram_counts[word] += 1
            self.bigram_counts[prev_word][word] += 1
            prev_word = word
        self.bigram_counts.default_factory = None  # make into a normal dict

        # Leave space for "<s>", "</s>", and "<unk>"
        top_counts = self.unigram_counts.most_common(None if size is None else (size - 3))
        vocab = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] +
                 [w for w,c in top_counts])

        # Assign an id to each word, by frequency
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}
        self.size = len(self.id_to_word)
        if size is not None:
            assert(self.size <= size)

        # For convenience
        self.wordset = set(self.word_to_id.keys())

        # Store special IDs
        self.START_ID = self.word_to_id[self.START_TOKEN]
        self.END_ID = self.word_to_id[self.END_TOKEN]
        self.UNK_ID = self.word_to_id[self.UNK_TOKEN]

    @property
    def num_unigrams(self):
        return len(self.unigram_counts)

    @property
    def num_bigrams(self):
        return len(self.bigram_counts)

    def __contains__(self, key):
        if isinstance(key, int):
            return (key > 0 and key < self.size)
        else:
            return key in self.word_to_id

    def words_to_ids(self, words):
        return [self.word_to_id.get(w, self.UNK_ID) for w in words]

    def ids_to_words(self, ids):
        return [self.id_to_word[i] for i in ids]

    def ids_to_feats(self, ids):
        returnedList = []
        for i in ids:
            wordFeats = []
            for item in i:
                wordFeats.append(self.id_to_word[item])
            returnedList.append(wordFeats)
        return returnedList
        # return [self.id_to_word[i] for i in ids]

    def pad_sentence(self, words, use_eos=True):
        ret = [self.START_TOKEN] + words
        if use_eos:
          ret.append(self.END_TOKEN)
        return ret

    def sentence_to_ids(self, words, use_eos=True):
        return self.words_to_ids(self.pad_sentence(words, use_eos))

    def ordered_words(self):
        """Return a list of words, ordered by id."""
        return self.ids_to_words(range(self.size))

    def write_flat_file(self, filename):
        """Write the vocabulary list to a flat file."""
        ordered_words = self.ids_to_words(range(self.size))
        with open(filename, 'w') as fd:
            for word in ordered_words:
                fd.write(word + "\n")
        print("Vocabulary ({:,} words) written to '{:s}'".format(len(ordered_words),
                                                               filename))

    def write_projector_config(self, checkpoint_dir, tensor_name):
        """Write metadata for TensorBoard Embeddings Projector."""
        import os
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        metadata_file = os.path.join(checkpoint_dir, "metadata.tsv")
        self.write_flat_file(metadata_file)
        # Write projector config pb
        projector_config_file = os.path.join(checkpoint_dir,
                                             "projector_config.pbtxt")
        with open(projector_config_file, 'w') as fd:
            contents = """embeddings {
              tensor_name: "%s"
              metadata_path: "metadata.tsv"
            }""" % tensor_name
            fd.write(contents)
        print("Projector config written to {:s}".format(projector_config_file))


