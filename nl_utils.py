from ug_utils import *
from nltk.tokenize import word_tokenize
import string
import spotlight
import stanfordnlp
from SPARQLWrapper import SPARQLWrapper, JSON

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
        self.nlq_tokens_entity_dict = {}

    def entity_linker(self, linker=None, kg='dbpedia'):
        """
        entity linking using dbpedia Spotlight
        :return:
        """
        if linker == 'spotlight' or linker == 'dbpedia':
            # todo: creating filters based on POS tags.
            # linking entities using dbpedia sptolight
            entities = []
            for token in self.nlq_tokens:
                try:
                    entities.append(spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', token,
                                                       confidence=0.1, support=5))
                except spotlight.SpotlightException as e:
                    pass

        elif linker == 'custom_linker':
            # todo: may be required to implement
            self.nlq_token_entity_dict = {token: token for token in self.nlq_tokens}
        else:
            # no entity linking
            self.nlq_token_entity_dict = {token: token for token in self.nlq_tokens}


    def formalize_into_sparql(self, kg='dbpedia'):
        """
        when the canonicalization is disabled, we will not have the NLCanonical object, instead nl_Canonical_list
        is set with NLQTokens(subclass NLQTokenDepParsed) instead.
        Therefore this method will be used to conver the Tokens into query (formalization).
        :return: a Query object
        """

        # for word in
        query_form = self.nlq_tokens_entity_dict
        return Query(query_form)

    def canonicalize(self, dependency_parsing=False, canonical_form=False):
        """
        Take the tokenized natural language question and parse it into un-grounded canonical form.
        :return:
        """
        if canonical_form: #todo
            canonical_form = self.nlq_tokens
            return NLCanonical(canonical_form)
        elif dependency_parsing:
            return NLQTokensDepParsed(self.nlq_tokens)
        else: # both are false, no canonicalization, a vanilla tokenized form
            return NLQTokens(self.nlq_tokens)


class NLQTokensDepParsed(NLQTokens):
    """
    A wrapper class to wrap the output of stanfordnlp.pipeline(). This class inherit attributes and
     methods of it's base class NLQTokens which has most of the useful methods defined.
    """
    def __init__(self, questions_tokens):
        # self.nlq_tokens = questions_tokens
        super().__init__(questions_tokens)


    def __str__(self):
        """
        return representation for the dependency in the sentence.
        :return:
        """
        return "\n".join([f"({word.text}, {word.governor}, {word.dependency_relation})"
                         for word in self.nlq_tokens.words])


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
        self.results = []

    def run(self, kg='dbpedia'):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)
        try:
            sparql.setQuery(self.sparql)
            self.results = sparql.query().convert()
        except:
            self.results = None


class NLCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """
    def __init__(self, canonical_form):
        self.nl_canonical = canonical_form


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


