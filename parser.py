from nl_utils import *
from ug_utils import *

class Parser(object):
    """
    Takes as input a list of pre_processed NLQuestion(one or more)
    questions and return list of
    """

    def __init__(self, pp_nlqs):
        """
        Take a list of questions (one or more)
        :param pp_nlqs: take as input a list of pre-processed NLQuestions
        """
        self.pp_nlqs = pp_nlqs
        self.ungrounded_form = []
        self.grounded_form = []

    def create_ungrounded_form(self):
        self.ungrounded_form= [self.nlq_to_ug_form(pp_question) for pp_question in self.pp_nlqs]
        return self.ungrounded_form

    def create_grounded_form(self):
        self.grounded_form = [self.ug_to_g_form(ug_form) for ug_form in self.ungrounded_form]
        return self.grounded_form

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
