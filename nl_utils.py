from ug_utils import *
from nltk.tokenize import word_tokenize
import string
import spotlight
import stanfordnlp

class NLQuestion(object):
    """
    This class will take Natural Language Question and create an object.
    """
    # class variable used to set up standfordnlp pipeline, so that every time loading could be avoided
    nlp = stanfordnlp.Pipeline(processors='tokenize,lemma,pos,depparse')

    def __init__(self, question):
        self.question = question

    def tokenize(self, dependency_parsing=False):
        """
        tokenize the nl_question to create a NLQToken object, when the dependency parser is True,
        it will return NLQTokenDepParsed
        :param dependency_parsing: when True, use stanfordnlp dependency parser.
        :return:
        """
        if dependency_parsing:
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


class NLQTokens(object):
    """
    Takes the Natural Language Question and processes using Standard NLP Tools.
    A wrapper class to wrap the output of PreProcessor into NLQTokens.
    """
    # pass
    def __init__(self, questions_tokens):
        self.nlq_tokens = questions_tokens

    def canonicalize(self, enable=False):
        """
        Take the tokenized natural language question and parse it into un-grounded canonical form.
        :return:
        """
        if enable: #todo
            canonical_form = self.nlq_tokens
            return NLCanonical(canonical_form)
        else:
            canonical_form = self.nlq_tokens
            return NLCanonical(canonical_form)

class NLQTokensDepParsed(object):
    """
    Takes the Natural Language Question and processes using Standard NLP Tools.
    A wrapper class to wrap the output of PreProcessor into NLQTokens.
    """
    # pass
    def __init__(self, questions_tokens):
        self.nlq_tokens = questions_tokens

    def canonicalize(self, enable=False):
        """
        Take the tokenized natural language question and parse it into un-grounded canonical form.
        :return:
        """
        if enable:  # todo
            canonical_form = self.nlq_tokens
            return NLCanonical(canonical_form)
        else:
            canonical_form = self.nlq_tokens
            return NLCanonical(canonical_form)

class NLCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """
    def __init__(self, canonical_form):
        self.nl_canonical = canonical_form

    def entity_linker(self, linker=None):
        """
        entity linking using dbpedia Spotlight
        :return:
        """
        if linker == 'spotlight':
            # todo: creating filters based on POS tags.
            #linking entities using dbpedia sptolight
            entities =[]
            for token in self.nl_canonical:
                try:
                    entities.append(spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', token,
                                               confidence=0.1, support=5))
                except spotlight.SpotlightException as e:
                    pass

        elif linker == 'custom_linker':
            # todo: may be required to implement
            entities = self.nl_canonical
        else:
            # no entity linking
            entities = self.nl_canonical

        return entities

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


