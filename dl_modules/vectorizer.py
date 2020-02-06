import numpy as np
from vocabulary import *

class Vectorizer(object):
    """
    Will take the tokens and return a fixed size vector using dictionary.
    """
    # pass
    def __init__(self, vocab):
        """
        maps words to integers
        :param vocab: the dictionary containing vocabulary of words, defined in the class Vocabulary
        """
        self.vocab = vocab

    def vectorize(self, question):
        """
        Create a vector representation of the questions
        :param question: a list of tokens
        :return: one-hot representation of the questions.
        """
        one_hot_vector = np.zeros((len(self.vocab), 1), dtype=np.float32)
        # one_hot_vector[index,0] = 1 for index in
        for token in question:
            one_hot_vector[self.vocab_lookup_token(token), 0] = 1

        return one_hot_vector

    @classmethod
    def from_dataframe(cls, vocab_df, cutoff=25):
        """
        Create a vocabulary from the vocabulary dataframe,
        :param vocab_df: the input dataframe of tokens/words to be used as Vocabulary for the Model
        :param cutoff: for a frequency based vocabulary, use only words appearing more than cutoff time.
        :return:
        """
        vocab = Vocabulary(add_unk=True)

        # populate Vocabulary if the word count is more than cutoff
        token_count = {}
        for token in vocab_df.tokens:
            token_count[token] += 1

        for token, count in token_count.items():
            if count > cutoff:
                vocab.add_token(token)

        return cls(vocab)

    @classmethod
    def from_list(cls, vocab_list, cutoff=25):
        """
        Create a vocabulary from the list of tokens
        :param vocab_list: list of tokens
        :param cutoff: cutoff to include a token in vocab
        :return: return Vocabulary object
        """
        vocab = Vocabulary(add_unk=True)

        # populate Vocabulary if the word count is more than cutoff
        token_count = {}
        for token in vocab_list:
            token_count[token] += 1

        for token, count in token_count.items():
            if count > cutoff:
                vocab.add_token(token)

        return cls(vocab)

    @classmethod
    def from_serializable(cls, vocab_dict):
        """
        :param vocab_dict:
        :return:
        """
        vocab = Vocabulary.from_serializable(vocab_dict['tokens'])

        return cls(vocab)

    def to_serializable(self):
        """
        serialize to cache the vocabulary
        :return:
        """
        return {'vocab': self.vocab.to_serializable()}