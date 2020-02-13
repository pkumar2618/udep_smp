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
        self.nlq_questions_list = [NLQuestion(nl_question) for nl_question in nlqs]
        self.nlq_tokens_list = []
        self.nl_canonical_list = []
        self.query_list = []
        self.entities_list = []

    def tokenize(self, dependency_parsing):
        """
        tokenize the natural language question question
        :return:
        """
        self.nlq_tokens_list = [nl_question.tokenize(dependency_parsing) for nl_question in self.nlq_questions_list]

    def canonicalize(self, dependency_parsing=False, canonical_form=False):
        self.nl_canonical_list = [nlq_tokens.canonicalize(dependency_parsing, canonical_form) for nlq_tokens in self.nlq_tokens_list]

    def disambiguate(self, linker=None):
        self.entities_list = [nl_canonical.entity_linker(linker) for nl_canonical in self.nl_canonical_list]

    def formalize(self):
        """
        takes the nl_canonical form and formalize it into a query
        :return:
        """
        self.query_list = [nl_canonical.formalize_into_sparql() for nl_canonical in self.nl_canonical_list]
        # return query_list

    @staticmethod
    def nlq_to_ug_form(nlq):
        return nlq

    @staticmethod
    def ug_to_g_form(ug_form):
        return ug_form

    @classmethod
    def from_file(cls, file_obj):
        """
        should parse question in batch
        :param file_obj:
        :return:
        """
        return cls(file_obj.readlines())

    # @classmethod
    # def from_list(cls, question_list):
    #     """
    #     Should parse question in batch
    #     :param question_list:
    #     :return:
    #     """

# if __init__ == __main__:
