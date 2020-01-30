

class UGForm(object):
    """
    Object for Ungrounded Form
    """
    def __init__(self, ug_form):
        self.ug_form = ug_form

    # @classmethod
    # def from_file(cls, filename):
    #     """
    #     Create a list of ungrounded representation from file having ungrounded'
    #     representation of questions
    #     :param filename: name of the file
    #     :return: a list of ug_form
    #     """
    #     ug_list = open(filename, 'r').readlines()
    #     return cls(ug_list)
    #
    # @classmethod
    # def from_list(cls, ug_list):
    #     """
    #     Create a list of ug_form
    #     :param ug_list: input list
    #     :return: a list of ug_form
    #     """
    #     ug_list = ug_list.items()
    #     return cls(ug_list)

    @staticmethod
    def ug_to_g(ug_form):
        return ug_form
