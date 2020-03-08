from ug_utils import *
from nltk.tokenize import word_tokenize
import string
import spotlight
import stanfordnlp
from sparql_builder import Query
from gensim.models import KeyedVectors
from threading import Semaphore

class NLQuestion(object):
    """
    This class will take Natural Language Question and create an object.
    """
    # class variable used to set up standfordnlp pipeline, so that every time loading could be avoided
    sd_nlp_loaded=False
    nlp = None


    def __init__(self, question):
        self.question = question
        if not NLQuestion.sd_nlp_loaded:
            NLQuestion.nlp = stanfordnlp.Pipeline(processors='tokenize,lemma,pos,depparse', use_gpu=False)
            NLQuestion.sd_nlp_loaded = True

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
        self.nlq_token_entity_dict = {}

    def canonicalize(self, dependency_parsing=False, canonical_form=False):
        """
        Take the tokenized natural language question and parse it into un-grounded canonical form.
        I am considering dependency_parsing as the canonical form here. However, there are works as in AskNow
        where the canonical form has altogether different syntax. Here we take either dependency parsing or the
        canonical form.
        :return:

        """
        if canonical_form: #todo
            # for now, consider canonicalization is disabled, and i pass on the nlq_token object over to the next stage,
            # through the nlq_canonical object.
            canonical_form = self.nlq_tokens
            return NLQCanonical(canonical_form)

        elif dependency_parsing:
            # in case dependency parsing is enabled, It will use the dependency tree as the canonical form
            # not that in that case the tokenizer in the NLQuestion object will return NLQTokensDepParsed object itself.
            return NLQCanonical(self.nlq_tokens)

        else:
            # both are false, no canonicalization, a vanilla tokenized form.
            # the NLQCanonical object will just be a list of tokens, as returned by NLQQuestion tokenizer with
            # dependency parsing disabled.
            return NLQCanonical(self.nlq_tokens)


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
        return "\n".join(["({ }, { }, { })".format(word.text, word.governor, word.dependency_relation) for word in self.nlq_tokens.words])


class NLQCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """

    # Using GloVe-6B trained on 6Billion tokens, contains 400k vocabulary
    # loading once for the object is created, and later to be used by entity linker
    # or predicate linker
    glove_loaded = False # flag to load the keyedvector from file once
    glove = None # the keyedvector stored using memory map for fast access



    def __init__(self, canonical_form):
        self.nlq_canonical = canonical_form
        if not NLQCanonical.glove_loaded:
            glove_loading_kv = KeyedVectors.load_word2vec_format("./pt_word_embedding/glove/glove.6B.50d.w2vformat.txt")
            glove_loading_kv.save('./glove_gensim_mmap')
            NLQCanonical.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
            NLQCanonical.glove.syn0norm = NLQCanonical.glove.syn0  # prevent recalc of normed vectors
            NLQCanonical.glove.most_similar('stuff')  # any word will do: just to page all in
            # Semaphore(0).acquire()  # just hang until process killed
            NLQCanonical.glove_loaded = True

    def entity_linker(self, linker=None, kg=None):
        """
        entity linking using dbpedia Spotlight
        Entity linker by definition must have reference to Knowledge Graph, whence it bring list of denotation for
        the token.
        :return:
        """
        if linker == 'spotlight' and kg == 'dbpedia':
            # todo: creating filters based on POS tags.
            # linking entities using dbpedia sptolight
            for token in self.nlq_canonical.words:
                try:
                    entities = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', token,
                                                       confidence=0.1, support=5)
                    self.nlq_token_entity_dict['token'] = entities

                except spotlight.SpotlightException as e:
                    self.nlq_token_entity_dict['token'] = None

        elif linker == 'custom_linker':
            # todo: may be required to implement
            self.nlq_token_entity_dict = {token.text: token.text for token in self.nlq_canonical.words}

        else:
            # no entity linking
            self.nlq_token_entity_dict = {k:v for (k, v) in zip([word.text for word in self.nlq_canonical.words],
                                                                [word.text for word in self.nlq_canonical.words])}

        return self.nlq_token_entity_dict

    def formalize_into_sparql(self, kg='dbpedia'):
        """
        when the canonicalization is disabled, we will not have the NLCanonical object, instead nlq_Canonical_list
        is set with NLQTokens(subclass NLQTokenDepParsed).
        Therefore this method will be used to conver the Tokens into query (formalization).
        Note: Fomalize into SPARQL will note require reference to a Knowledge Graph for the purpose of
        denotation as that is already done. The KG is required to provide list of namespace for creating query-string
        in during entity_linking stage of the parser.
        :return: a Query object
        """

        # for word in
        query_string = self.nlq_token_entity_dict
        return Query(query_string)


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


if __name__ == "__main__":
    nlq_canon = NLQCanonical("glove_testing")
    print(NLQCanonical.glove['the'])
    print(NLQCanonical.glove['lesson'])
    print(NLQCanonical.glove.most_similar('orange'))


