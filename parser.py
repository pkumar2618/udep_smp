from nl_utils import *
from ug_utils import *

class Parser(object):
    """
    Takes as input a list NLQuestion(one or more)
    and parse it.
    """

    def __init__(self, nlqs):
        """
        Take a list of questions (one or more)
        :param pp_nlqs: take as input a list of pre-processed NLQuestions
        """
        nlq_questions_list = [NLQuestion(nl_question) for nl_question in nlqs]
        self.nlq_tokens_list = [nl_question.tokenize() for nl_question in nlq_questions_list]
        self.nl_canonical_list = []
        self.query_list = []

    def canonicalize(self):
        self.nl_canonical_list = [nlq_tokens.canonicalize() for nlq_tokens in self.nlq_tokens_list]
        # return nl_canonical_list

    def formalize(self):
        """
        takes the nl_canonical form and formalize it into a query
        :return:
        """
        self.query_list = [nl_canonical.formalize() for nl_canonical in self.nl_canonical_list]
        # return query_list

    @staticmethod
    def nlq_to_ug_form(nlq):
        return nlq

    @staticmethod
    def ug_to_g_form(ug_form):
        return ug_form

    # @classmethod
    # def from_file(cls, file_obj):
    #     """
    #     should parse question in batch
    #     :param file_obj:
    #     :return:
    #     """
    #     return cls(file_obj.readlines())

    # @classmethod
    # def from_list(cls, question_list):
    #     """
    #     Should parse question in batch
    #     :param question_list:
    #     :return:
    #     """

# if __init__ == __main__:
