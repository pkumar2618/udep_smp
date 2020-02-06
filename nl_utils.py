from ug_utils import *
from nltk.tokenize import word_tokenize
import string

class NLQuestion(object):
    """
    This class will take Natural Language Question and create an object.
    """
    def __init__(self, question):
        self.question = question

    # @classmethod
    # def from_file(cls, filename):
    #     "Take the question from questions in a file and create list of questions"
    #     data = open(filename, 'r').readlines()
    #     return cls(data)
    #
    # @classmethod
    # def from_list(cls, question_list):
    #     "Take questions from a dictionary of question and create a list "
    #     return cls(question_list.items())

    def tokenize(self):
        """
        tokenize the nl_question to create a NLQToken object
        :return:
        """
        # split sentence into token using nltk.word_tokenize(), this will have punctuations as separate tokens
        nlq_tokens = word_tokenize(self.question)

        # filter out punctuations from each word
        table = str.maketrans('', '', string.punctuation)
        nlq_tokens = [token.translate(table) for token in nlq_tokens]

        # remove token which are not alpha-numeric
        nlq_tokens = [token for token in nlq_tokens if token.isalpha()]

        # convert to lower case
        nlq_tokens = [token.lower() for token in nlq_tokens]

        return NLQTokens(nlq_tokens)

class NLQTokens(object):
    """
    Takes the Natural Language Question and processes using Standard NLP Tools.
    A wrapper class to wrap the output of PreProcessor into NLQTokens.
    """
    # pass
    def __init__(self, questions_tokens):
        self.nlq_tokens = questions_tokens

    def canonicalize(self):
        """
        Take the tokenized natural language question and parse it into un-grounded canonical form.
        :return:
        """
        canonical_form = self.nlq_tokens
        return NLCanonical(canonical_form)


class NLCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """
    def __init__(self, canonical_form):
        self.nl_canonical = canonical_form

    def formalize(self):
        """
        create a Query from the Canonical form
        :return: query
        """
        query_form = self.nl_canonical
        return Query(query_form)

class Query(object):
    """
    Wrapper for storing logical query
    """
    def __init__(self, query_form):
        """
        take the query_form obtained by formalizer and wrap it
        :param query_form:
        """
        self.sparql = query_form

class NLQDependencyTree(NLQuestion):
    """
    Take the Natural Langugage Question and return an dependy parsed tree.
    """
    pass
    # def __init__(self, *args, **kwargs):
    #     if kwargs.get(parser):
    #         self.dep_tree = standford_ud(args[0])
    #
    #     elif kwargs.get(parser == "stanford_parser"):
    #         self.dep_tree = stanford_parser(args[0])


