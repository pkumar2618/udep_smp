import string

import stanfordnlp
from nltk import word_tokenize

from udeplib.nlqtokens import NLQTokens, NLQTokensDepParsed

# Philosopy 1: Object may be created empty and later on modified using a bunch of operations on them.
# Philosopy 2: An object is modified by performing some operation on it from its own class and takes a completely
# different character that it has to be named a class of its own.


class NLQuestion(object):
    """
    This class will take Natural Language Question and create an object.
    """
    # class variable used to set up standfordnlp pipeline, so that every time loading could be avoided
    sd_nlp_loaded=False
    nlp = None


    def __init__(self, question, bypass_pre_processing=True):
        if bypass_pre_processing:
            self.question = question
        else:
            self.question = question
            if not NLQuestion.sd_nlp_loaded:
                NLQuestion.nlp = stanfordnlp.Pipeline(processors='tokenize,lemma,pos,depparse', use_gpu=True)
                NLQuestion.sd_nlp_loaded = True

    def tokenize(self, dependency_parsing=False, bypassing=True):
        """
        tokenize the nl_question to create a NLQToken object, when the dependency parser is True,
        it will return NLQTokenDepParsed
        :param dependency_parsing: when True, use stanfordnlp dependency parser.
        :return:
        """
        if bypassing:
            return NLQTokens(self.question)

        elif dependency_parsing:
            # nlp = stanfordnlp.Pipeline(processors='tokenize,lemma,pos,depparse')
            doc = NLQuestion.nlp(self.question)
            # doc.sentences[0].print_dependencies()
            return NLQTokensDepParsed(doc.sentences[0])

        else:
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