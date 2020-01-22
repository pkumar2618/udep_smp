from nl_utils import *
from ug_utils import *

class Parser(object):
    """
    Takes input either a question or a list of question or a file containing
    question
    """
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

    def __init__(self, nlqs):
        for nl_question in nlqs:
            nl_question = NLQuestion(nl_question)
            ug_form = NLQuestion.nl_to_ug(nl_question)
            g_form = UGForm.ug_to_g(ug_form)






