class NLQuestion(object):
    """
    This class will take Natural Language Question and create an object.
    """
    def __init__(self, question):

        self.question = question

    @classmethod
    def from_file(cls, filename):
        "Take the question from questions in a file and create list of questions"
        data = open(filename, 'r').readlines()
        return cls(data)

    @classmethod
    def from_list(cls, question_list):
        "Take questions from a dictionary of question and create a list "
        return cls(question_list.items())


class NLQTokens(NLQuestion):
    """
    Takes the Natural Language Question and processes using Standard NLP Tools.

    """
    def __init__(self, nl_quesion):
        self.nlq_tokesns = tokenize(lammatise(nl_quesion))

class NLQDependencyTree(NLQuestion):
    """
    Take the Natural Langugage Question and return an dependy parsed tree.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get(parser):
            self.dep_tree = standford_ud(args[0])

        elif kwargs.get(parser == "stanford_parser"):
            self.dep_tree = stanford_parser(args[0])
