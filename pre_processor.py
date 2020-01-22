from nl_utils import *
from ug_utils import *
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
from pprint import pprint
import string
from io import IOBase

"""
Design Choice: gensim is a grate package for handling large file, provide support for word2vec and topic modelling
"""

class PreProcessor(object):
    """
    Takes input either a question or a list of question or a file containing
    question and does the pre-processing to be passed on to the parser for semantic parsing
    """

    def __init__(self, nlqs):
        """
        Tokenize sentence/documents into words
        :param nlqs:
        """
        if isinstance(nlqs, list):
            # split sentence into token using nltk.word_tokenize(), this will have punctuations as separate tokens
            tokens_mat = [[token for token in word_tokenize(nl_question)] for nl_question in nlqs ]

            # filer out punctuations from each word
            table = str.maketrans('', '', string.punctuation)
            tokens_mat = [[token.translate(table) for token in token_row] for token_row in tokens_mat]

            # remove token which are not alpha-numeric
            tokens_mat = [[token for token in tokens_row if token.isalpha()] for tokens_row in tokens_mat]

            # convert to lower case
            tokens_mat = [[token.lower() for token in tokens_row] for tokens_row in tokens_mat]

            # creating gensim dictonary
            self.dictionary = corpora.Dictionary(tokens_mat)

        ## using simple_preprocess
        # that will take file_object
        elif isinstance(nlqs, IOBase):
            self.dictionary = corpora.Dictionary(
                simple_preprocess(line, deacc=True) for line in nlqs)
        print(self.dictionary)


    @classmethod
    def from_file(cls, file_obj: object):
        """
        takes file_object containing questions
        :param file_obj:
        :return:
        """
        # return cls(file_obj.readlines())
        return cls(file_obj)

    @classmethod
    def from_list(cls, list_obj):
        """
        takes list of questions
        :param list_obj:
        :return:
        """
        return cls(list_obj)





