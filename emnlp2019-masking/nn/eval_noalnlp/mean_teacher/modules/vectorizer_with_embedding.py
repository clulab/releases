from collections import Counter
from .vocabulary import Vocabulary,SequenceVocabulary
from mean_teacher.utils.logger import LOG
import numpy as np
import string


# ### The Vectorizer


class VectorizerWithEmbedding(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, claim_ev_vocab, labels_vocab):
        self.claim_ev_vocab = claim_ev_vocab
        self.label_vocab = labels_vocab

    def vectorize(self, input_sentence, vector_length=-1):
        """
        Args:
            input_sentence (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        #indices = [self.claim_ev_vocab.begin_seq_index]
        indices=[]
        indices.extend(self.claim_ev_vocab.lookup_token(token)
                       for token in input_sentence.split(" "))
        #indices.append(self.claim_ev_vocab.end_seq_index)

        #if we have not found or are providing the length of the input with maximum length.
        if vector_length < 0:
            vector_length = len(indices)

        #Pad the rest of the sentence with masking token
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.claim_ev_vocab.mask_index

        return out_vector

    def update_word_count(self, sentence,word_counts):
            for word in sentence.split(" "):
                #if word not in string.punctuation:
                    word_counts[word] += 1
            return word_counts

    @classmethod
    def create_vocabulary(cls, claim_ev_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            claim_ev_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """

        claim_ev_vocab = SequenceVocabulary()
        word_counts = Counter()

        for claim in (claim_ev_df.claim):
            word_counts=cls.update_word_count(cls,claim,word_counts)

        for ev in (claim_ev_df.evidence):
            word_counts=cls.update_word_count(cls, ev,word_counts)


        for word, count in word_counts.items():
            # removing cutoff for the time being- to check if it increases accuracy
            # if count > cutoff:
                claim_ev_vocab.add_token(word)

        labels_vocab = Vocabulary(add_unk=False)
        for label in sorted(set(claim_ev_df.label)):
            labels_vocab.add_token(label)
        LOG.debug(f"size of claim_ev_vocab is {len(claim_ev_vocab)}")
        LOG.debug(f"size of labels_vocab is {len(labels_vocab)}")


        return cls(claim_ev_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        claim_ev_vocab_ser = SequenceVocabulary.from_serializable(contents['claim_ev_vocab'])
        label_vocab_ser = Vocabulary.from_serializable(contents['label_vocab'])
        return cls(claim_ev_vocab=claim_ev_vocab_ser, labels_vocab=label_vocab_ser)

    def to_serializable(self):
        return {'claim_ev_vocab': self.claim_ev_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}