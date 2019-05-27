import numpy as np
from keras import backend as K
from keras.optimizers import SGD



def unigram_probabilities(counts, power=1):
    """gets an array of counts and returns an unigram probability
    raised to some power"""
    counts = np.array(counts)
    counts = np.power(counts, power)
    return counts / np.sum(counts)



def normalize(embeddings):
    """gets an embedding matrix where every row is an embedding
    and returns a matrix of normalized embeddings"""
    norm = K.sqrt(K.sum(K.square(embeddings), axis=1, keepdims=True))
    return embeddings / norm



class Word2vec(object):

    def __init__(self, word_vocabulary, context_vocabulary, algorithm, embedding_size, num_neg_samples, prob_power, learning_rate):

        self.algorithm = algorithm
        if self.algorithm == 'skipgram':
            self.in_vocabulary = word_vocabulary
            self.out_vocabulary = context_vocabulary
        elif self.algorithm == 'cbow':
            self.in_vocabulary = context_vocabulary
            self.out_vocabulary = word_vocabulary
        self.embedding_size = embedding_size
        self.num_neg_samples = num_neg_samples
        self.prob_power = prob_power
        self.learning_rate = learning_rate
        self.uni_probs = unigram_probabilities(self.out_vocabulary.counts, self.prob_power)

        # initialize embeddings randomly
        value = np.random.uniform(-1, 1, (self.in_vocabulary.size(), self.embedding_size))
        self.in_embeddings = K.variable(value=value)
        value = np.random.uniform(-1, 1, (self.out_vocabulary.size(), self.embedding_size))
        self.out_embeddings = K.variable(value=value)

        if self.algorithm == 'skipgram':
            # In skipgram the input is a word
            # and the output is one of the words surrounding it.
            # The input is an embedding and the output is an embedding.
            in_dim = 1

        elif self.algorithm == 'cbow':
            # In cbow the output is a word
            # and the inputs are the words surrounding it.
            # The input is a matrix where each row is an embedding
            # and the output is an embedding.
            in_dim = 2

        # placeholders for minibatch
        # they need to be ints because they are used as indices
        # they will be provided when calling keras_train() (see below)
        mb_in_indices = K.placeholder(ndim=in_dim, dtype='int32')
        mb_out_indices = K.placeholder(ndim=1, dtype='int32')
        mb_negsamples_indices = K.placeholder(ndim=2, dtype='int32')

        # get embeddings corresponding to minibatch
        in_embs = K.gather(self.in_embeddings, mb_in_indices)
        out_embs = K.gather(self.out_embeddings, mb_out_indices)
        neg_embs = K.gather(self.out_embeddings, mb_negsamples_indices)

        if self.algorithm == 'cbow':
            # input is average of all words in window
            in_embs = K.mean(in_embs, axis=1)

        # we want to maximize this objective
        log_prob_positive = K.log(K.sigmoid(K.batch_dot(in_embs, out_embs, axes=1)))
        log_prob_negative = K.sum(K.log(K.sigmoid(-K.batch_dot(K.expand_dims(in_embs), neg_embs, axes=(1, 2)))), axis=2)
        objective = K.mean(log_prob_positive + log_prob_negative)

        # make an optimizer to update the embeddings
        optimizer = SGD(lr=self.learning_rate)
        params = [self.in_embeddings, self.out_embeddings]
        constraints = []
        loss = -objective # minimize loss => maximize objective
        updates = optimizer.get_updates(params, constraints, loss)

        # this function gets the minibatch
        # including the negative samples
        # and returns the loss for the minibatch
        # and updates the embeddings accordingly
        self.keras_train = K.function([mb_in_indices, mb_out_indices, mb_negsamples_indices], [loss], updates=updates)

        self.session = K.get_session()

    def train(self, mb_words, mb_contexts):
        """train embeddings on a minibatch and return the loss"""
        # find minibatch size
        mb_size = len(mb_words)
        # set in and out
        if self.algorithm == 'skipgram':
            mb_in = mb_words
            mb_out = mb_contexts
        elif self.algorithm == 'cbow':
            mb_in = mb_contexts
            mb_out = mb_words
        # get negative samples
        n = self.out_vocabulary.size()
        size = (mb_size, self.num_neg_samples)
        neg_samples = np.random.choice(n, size=size, p=self.uni_probs)
        # train
        return self.keras_train([mb_in, mb_out, neg_samples])

    def get_word_vocabulary(self):
        if self.algorithm == 'skipgram':
            return self.in_vocabulary
        elif self.algorithm == 'cbow':
            return self.out_vocabulary

    def get_context_vocabulary(self):
        if self.algorithm == 'skipgram':
            return self.out_vocabulary
        elif self.algorithm == 'cbow':
            return self.in_vocabulary

    def get_in_embeddings(self):
        """return normalized in embeddings as numpy array"""
        return normalize(self.in_embeddings).eval(session=self.session)

    def get_out_embeddings(self):
        """return normalized out embeddings as numpy array"""
        return normalize(self.out_embeddings).eval(session=self.session)

    def get_word_embeddings(self):
        """return normalized word embeddings as numpy array"""
        if self.algorithm == 'skipgram':
            return self.get_in_embeddings()
        elif self.algorithm == 'cbow':
            return self.get_out_embeddings()

    def get_context_embeddings(self):
        """return normalized context embeddings as numpy array"""
        if self.algorithm == 'skipgram':
            return self.get_out_embeddings()
        elif self.algorithm == 'cbow':
            return self.get_in_embeddings()
