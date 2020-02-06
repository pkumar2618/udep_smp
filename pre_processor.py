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
    Takes as input a list of questions (one or more) or a file containing
    question and does their pre-processing and would save it as a list of NLQuestion objects.

    """
    def __init__(self, nlqs, batch_size=1):
        """
        Tokenize a list of sentences (one or more) into words
        :param nlqs: list of sentences
        """
        assert isinstance(nlqs, list), "Questions should be inside a list"

        # split sentence into token using nltk.word_tokenize(), this will have punctuations as separate tokens
        pre_proc_questions_list = [NLQuestion([token for token in word_tokenize(nl_question)]) for nl_question in nlqs ]

        # filter out punctuations from each word
        table = str.maketrans('', '', string.punctuation)
        pp_questions_list = [NLQuestion([token.translate(table) for token in pp_question.question]) for pp_question in pre_proc_questions_list]

        # remove token which are not alpha-numeric
        pp_questions_list = [NLQuestion([token for token in pp_question.question if token.isalpha()]) for pp_question in pp_questions_list]

        # convert to lower case
        pp_questions_list = [NLQuestion([token.lower() for token in pp_question.question]) for pp_question in pp_questions_list]

        self.pp_questions_list = pp_questions_list

        # creating gensim dictonary
        # self.dictionary = corpora.Dictionary(tokens_mat)

        ## using simple_preprocess
        # that will take file_object
        # elif isinstance(nlqs, IOBase):
        #     self.dictionary = corpora.Dictionary(
        #         simple_preprocess(line, deacc=True) for line in nlqs)
        # print(self.dictionary)

    def get_pp_questions(self):
        return self.pp_questions_list

    @classmethod
    def from_file(cls, file_obj: object):
        """
        takes file_object containing questions
        :param file_obj:
        :return:
        """
        return cls(file_obj.readlines())
        # return cls(file_obj)

    @classmethod
    def from_list(cls, list_obj):
        """
        takes list of questions
        :param list_obj:
        :return:
        """
        return cls(list_obj)





