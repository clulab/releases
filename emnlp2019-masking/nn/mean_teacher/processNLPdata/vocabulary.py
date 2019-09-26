import io
import numpy as np

DEFAULT_ENCODING = 'utf8'
EOS = '</s>'


class Vocabulary(object):

    def __init__(self):
        self.word_to_id = {}
        self.words = []
        self.counts = []
        # EOS token should always be on top
        self.add(EOS, 0)

    # returns number of words in the vocabulary
    def size(self):
        return len(self.words)

    # add word to vocabulary, increasing its count by the provided `count`
    def add(self, word, count=1):
        if word not in self.word_to_id:
            self.word_to_id[word] = len(self.words)
            self.words.append(word)
            self.counts.append(0)
        word_id = self.word_to_id[word]
        self.counts[word_id] += count
        return word_id

    def contains(self, word):
        return word in self.word_to_id

    # get the id of a word
    def get_id(self, word):
        if (word in self.word_to_id):
            return self.word_to_id.get(word)
        else:
            return self.word_to_id.get(EOS)

    # get the word corresponding to an id
    def get_word(self, id):
        return self.words[id]

    # get the frequency of a word
    def get_count(self, id):
        return self.counts[id]

    def prepare(self, min_count=None):
        # convert to numpy arrays (excluding EOS)
        eos_count = self.counts[0]
        words = np.array(self.words)[1:]
        counts = np.array(self.counts)[1:]
        # sort by count
        indices = np.argsort(counts)
        # remove words with frequency less than min_count
        if min_count is not None:
            indices = indices[counts[indices] >= min_count]
        indices = indices[::-1] # in descending order
        words = words[indices]
        counts = counts[indices]
        # replace object's fields
        self.words = np.insert(words, 0, EOS)
        self.counts = np.insert(counts, 0, eos_count)
        self.word_to_id = {k:v for v,k in enumerate(words)}
        print('num words:',self.words.shape[0])

    # returns a string representation of the vocabulary
    def to_string(self):
        output = ''
        for (word, count) in zip(self.words, self.counts):
            output += '%s\t%s\n' % (word, count)
        return output

    # write vocabulary to file
    def to_file(self, file, encoding=DEFAULT_ENCODING):
        with io.open(file, 'w+', encoding=encoding) as f:
            f.write(self.to_string())

    # read vocabulary from file
    @classmethod
    def from_file(cls, file, encoding=DEFAULT_ENCODING):
        with io.open(file, encoding=encoding) as f:
            vocabulary = Vocabulary()
            for line in f:
                [word, count] = line.split('\t')
                vocabulary.add(word, int(count))

            return vocabulary
