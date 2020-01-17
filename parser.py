class Parser(object):
    """
    Takes input either a question or a list of question or a file containing
    question
    """
    @classmethod
    def from_file(cls, filename):
        """
        should parse question in batch
        :param filename:
        :return:
        """

    @classmethod
    def from_list(cls, question_list):
        """
        Should parse question in batch
        :param question_list:
        :return:
        """

    def __init__(self, nl_question):
        if isinstance(nl_question, NLQuestion):
            ug_form = nl_to_ug(nl_question)
            g_form = ug_to_g(ug_form)






