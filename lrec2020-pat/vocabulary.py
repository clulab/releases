def make_word_vocabulary(sentences, prune=0, normalize=True):
    """make word vocabulary from a list of sentences"""
    vocab = Vocabulary()
    for sentence in sentences:
        words = [e.norm if normalize else e.form for e in sentence]
        vocab.count_words(words)
    if prune > 0:
        vocab = vocab.prune(prune)
    return vocab



def make_tag_vocabulary(sentences, partofspeech_type, prune=0):
    """make part-of-speech vocabulary from a list of sentences"""
    vocab = Vocabulary()
    for sentence in sentences:
        tags = [e.get_partofspeech_tag(partofspeech_type) for e in sentence]
        vocab.count_words(tags)
    if prune > 0:
        vocab = vocab.prune(prune)
    return vocab



def make_deprel_vocabulary(sentences, prune=0):
    """make dependency relation vocabulary from a list of sentences"""
    vocab = Vocabulary()
    for sentence in sentences:
        deprels = [e.deprel for e in sentence]
        vocab.count_words(deprels)
    if prune > 0:
        vocab = vocab.prune(prune)
    return vocab



def make_pos_vocabulary(sentences, prune=0, left_threshold = -50, right_threshold = 50):
    """make position vocabulary from a list of sentences"""
    vocab = Vocabulary()
    for sentence in sentences:
        positions = [str(e.pos) for e in sentence if left_threshold < e.pos < right_threshold]
        vocab.count_words(positions)
    if prune > 0:
        vocab = vocab.prune(prune)
    return vocab



def make_char_vocabulary(sentences, prune=0, normalize=False):
    """make character vocabulary from a list of sentences"""
    vocab = Vocabulary()
    for sentence in sentences:
        for entry in sentence:
            word = entry.norm if normalize else entry.form
            vocab.count_words(list(word))
    if prune > 0:
        vocab = vocab.prune(prune)
    return vocab



class Vocabulary:
    """Keeps a mapping between words and ids. Also keeps word counts."""

    def __init__(self):
        self.pad = 0
        self.unk = 1
        self.i2w = ['<pad>', '<unk>']
        self.w2i = {w:i for i,w in enumerate(self.i2w)}
        self.counts = [0] * len(self.i2w)

    def __len__(self):
        return len(self.i2w)

    def __iter__(self):
        return iter(self.i2w)

    def __contains__(self, word):
        if isinstance(word, int):
            return 0 <= word < len(self.i2w)
        else:
            return word in self.w2i

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.w2i:
                return self.index(key)
            else:
                return self.unk
        elif isinstance(key, (int, slice)):
            return self.i2w[key]
        elif key is None:
            return self.unk
        else:
            raise KeyError(f'invalid key: {type(key)} {key} {self.w2i} {self.i2w}')

    def prune(self, threshold):
        """returns a new vocabulary populated with elements
        from `self` whose count is greater than `threshold`"""
        v = Vocabulary()
        for i, w in enumerate(self.i2w):
            c = self.count(i)
            if c > threshold:
                v.count_word(w, c)
        return v

    def count_words(self, words):
        for w in words:
            self.count_word(w)

    def count_word(self, word, count=1):
        """increments the given word's count"""
        if word not in self.w2i:
            self.w2i[word] = len(self.i2w)
            self.i2w.append(word)
            self.counts.append(count)
        else:
            self.counts[self.w2i[word]] += count

    def word(self, i):
        """gets an index and returns a word"""
        return self.i2w[i]

    def index(self, w):
        """gets a word and returns an index"""
        return self.w2i.get(w, self.unk)

    def count(self, i):
        """gets an index (or word) and returns the corresponding count"""
        if isinstance(i, str):
            # seems like i is a word, not an index
            i = self.index(i)
        return self.counts[i] if 0 <= i < len(self.counts) else 0

    def histogram(self):
        """returns a list of (word, count) tuples"""
        return list(zip(self.i2w, self.counts))

    def __str__(self):
        return str(self.pad) + "\t" + str(self.unk) + "\t" + str(self.i2w) + "\t" + str(self.w2i) + "\t" + str(self.counts) + "\n"
