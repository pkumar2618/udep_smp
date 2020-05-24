# from ug_utils import *
# import spotlight
from udep_lib.nlqcanonical import NLQCanonical
# from gensim.models import KeyedVectors
import pickle
import logging

logger = logging.getLogger(__name__)
# Philosopy 1: Object may be created empty and later on modified using a bunch of operations on them.
# Philosopy 2: An object is modified by performing some operation on it from its own class and takes a completely
# different character that it has to be named a class of its own.

class NLQTokens(object):
    """
    Takes the Natural Language Question and processes using Standard NLP Tools.
    A wrapper class to wrap the output of PreProcessor into NLQTokens.
    """
    # pass
    def __init__(self, question_tokens):
        self.nlq_tokens = question_tokens
        logger.info(f'tokens: {question_tokens}')

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


if __name__ == "__main__":
    # NLQCanonical object should be created first to load and save the glove word2vec dataset_qald
    # nlq_canon = NLQCanonical("glove_testing")
    # print(NLQCanonical.glove['the'])
    # print(NLQCanonical.glove['lesson'])
    # print(NLQCanonical.glove.most_similar('orange'))
    # require to run the word with each predicate in the dbpedia.

    # testing entity_predicate_linker
    question = "Where is Fort Knox located ?"
    dump_name = "fort_knox"

    # question = "Where was Barack Obama Born?"
    # dump_name = "obama_born"

    try:
        # if the file exist load it
        with open(f'{dump_name}.pkl', 'rb') as f:
            nlq_tokens = pickle.load(f)

    except FileNotFoundError as e:
        from udep_lib.nlquestion import NLQuestion
        nlq = NLQuestion(question)
        nlq_tokens = nlq.tokenize(dependency_parsing=True)
        with open(f'{dump_name}.pkl', 'wb') as f:
            pickle.dump(nlq_tokens, f)

    nlq_canon = nlq_tokens.canonicalize(dependency_parsing=True)

    try:
        with open(f'{dump_name}_test.pkl', 'rb') as testing_f:
            nlq_canon = pickle.load(testing_f)

    except FileNotFoundError as e:
        nlq_canon.formalize_into_udeplambda()
        with open(f'{dump_name}_test.pkl', 'wb') as testing_f:
            pickle.dump(nlq_canon, testing_f)

    try:
        with open(f'{dump_name}_test1.pkl', 'rb') as testing_f:
            nlq_canon = pickle.load(testing_f)
    except FileNotFoundError as e:
        nlq_canon.entity_predicate_linker(linker='spotlight', kg='dbpedia')
        with open(f'{dump_name}_test1.pkl', 'wb') as testing_f:
            pickle.dump(nlq_canon, testing_f)

    query = nlq_canon.translate_to_sparql()

    print(query.get_query_string())

    # nlq_canon.entity_predicate_linker(linker='spotlight', kg='dbpedia')
