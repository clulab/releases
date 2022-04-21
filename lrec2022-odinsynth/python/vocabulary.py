import config

from itertools import takewhile 

class Vocabulary:

    def __init__(self, i2t=None, counts=None):
        self.i2t = list() if i2t is None else i2t
        self.t2i = {t:i for i,t in enumerate(self.i2t)}
        self.counts = counts

        self.pad_token = config.PAD_TOKEN
        self.cls_token = config.CLS_TOKEN
        self.sep_token = config.SEP_TOKEN
        self.unk_token = config.UNK_TOKEN

        if i2t is None:
            # special tokens
            self.pad_token_id = self.add_token(self.pad_token)
            self.cls_token_id = self.add_token(self.cls_token)
            self.sep_token_id = self.add_token(self.sep_token)
            self.unk_token_id = self.add_token(self.unk_token)
            # Reset the counts of special tokens to 0
            self.counts = [0] * len(self.i2t)
        else:
            self.pad_token_id = self.t2i[self.pad_token]
            self.cls_token_id = self.t2i[self.cls_token]
            self.sep_token_id = self.t2i[self.sep_token]
            self.unk_token_id = self.t2i[self.unk_token]

    def __len__(self):
        return len(self.i2t)

    def save(self, filename):
        with open(filename, 'w') as f:
            for t in self.i2t:
                f.write(f'{t}\t{self.lookup_count_for_word(t)}\n')

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [x.split('\t') for x in lines]
            words = [x[0].strip() for x in lines]
            counts = [int(x[1].strip()) for x in lines]
            return cls(words, counts)

    @classmethod
    def load_without_counts(cls, filename):
        with open(filename) as f:
            return cls(f.read().splitlines())

    def _add_count(self, index, new=False):
        if new:
            self.counts[index] = self.counts[index] + 1
        else:
            self.counts.append(1)

    def add_token(self, token):
        if token in self.t2i:
            index = self.t2i[token]
            self._add_count(index, False)
        else:
            index = len(self.i2t)
            self.i2t.append(token)
            self.t2i[token] = index
            self._add_count(index, True)
        return index

    def add_tokens(self, tokens):
        return [self.add_token(t) for t in tokens]

    def lookup_token(self, token):
        return self.t2i.get(token, self.unk_token_id)

    def lookup_tokens(self, tokens):
        return [self.lookup_token(t) for t in tokens]

    def lookup_index(self, index):
        if index < 0:
            raise IndexError(f"the index ({index}) can't be negative")
        if index >= len(self.i2t):
            raise IndexError(f'the index ({index}) is not in the Vocabulary')
        return self.i2t[index]

    def lookup_indices(self, indices):
        return [self.lookup_index(i) for i in indices]

    # Returns count of unk_token if the word is not in the vocabulary (unk_token count is 0)
    def lookup_count_for_word(self, word):
        return self.counts[self.lookup_token(word)]

    def prune(self, threshold: int, filter=None):
        target_tokens = list(takewhile(lambda x: self.lookup_count_for_word(x) == 0, self.i2t))
        target_counts = [0] * len(target_tokens)
        if filter is not None:
            for idx, token in enumerate(self.i2t):
                if self.counts[idx] > threshold and filter(token):
                    target_tokens.append(token)
                    target_counts.append(self.counts[idx])
        else:
            for idx, token in enumerate(self.i2t):
                if self.counts[idx] > threshold:
                    target_tokens.append(token)
                    target_counts.append(self.counts[idx])
        return Vocabulary(target_tokens, target_counts)
