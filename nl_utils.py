# from ug_utils import *
from nltk.tokenize import word_tokenize
import string
import spotlight
import stanfordnlp
from sparql_builder import Query
from gensim.models import KeyedVectors
from threading import Semaphore
import pickle
import subprocess
import re
import json
from db_utils import get_property_using_cosine_similarity
import ast
from rdflib import URIRef, BNode


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
            NLQuestion.nlp = stanfordnlp.Pipeline(processors='tokenize,lemma,pos,depparse', use_gpu=True)
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
        self.udep_lambda = None
        self.nlq_phrase_kbentity_dict = {}
        self.nlq_word_kb_predicate_dict = {}
        if not NLQCanonical.glove_loaded:
            try:
                NLQCanonical.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
                NLQCanonical.glove_loaded = True
            except Exception as e:
                glove_loading_kv = KeyedVectors.load_word2vec_format("./pt_word_embedding/glove/glove.6B.50d.w2vformat.txt")
                glove_loading_kv.save('./glove_gensim_mmap')
                NLQCanonical.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
                # NLQCanonical.glove.syn0norm = NLQCanonical.glove.syn0  # prevent recalc of normed vectors
                # NLQCanonical.glove.most_similar('stuff')  # any word will do: just to page all in
                # Semaphore(0).acquire()  # just hang until process killed
                NLQCanonical.glove_loaded = True

    def entity_predicate_linker(self, linker=None, kg=None):
        """
        entity linking using dbpedia Spotlight
        Entity linker by definition must have reference to Knowledge Graph, whence it bring list of denotation for
        the token.
        :return:
        """
        if linker == 'spotlight' and kg == 'dbpedia':
            if self.udep_lambda: # if the formalization was true. start the linker
                # linking entities using dbpedia sptolight
                # identify entity phrase from the predicate arguments
                # and
                # linking dbpedia_predicate using cosine similarity on word2vec
                # find the event type tokens according to Neo-Davidsonia grammar
                for neod_lambda_term in self.udep_lambda['dependency_lambda'][0]:
                    word_udep, type_tuple = NLQCanonical.get_name_type_tuple(neod_lambda_term)
                    args = type_tuple.split(",")
                    word_modifier = word_udep.split(".")
                    word = word_modifier[0]
                    try:
                        modifier = word_modifier[1]
                    except IndexError as e:
                        modifier = None

                    try:
                        # linking resources
                        # if neod_term is a event use its name to get dbpedia predicate
                        if re.match(r'[\d]+:e', args[0]):
                            # if neod_term is a argument to a predicate the terms inside the bracket is an entity
                            if modifier is not None: # if modifier is not present, it's Question Term or Where term
                                if re.match(r'arg[\d+]', modifier): # if modifier is an arg[\d] term,
                                    # it contains entities inside the brackets
                                    entity_phrase  =  re.split(r'[.]', args[1])[1]
                                    try:
                                        entities = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', entity_phrase,
                                                                      confidence=0.0, support=0)
                                        self.nlq_phrase_kbentity_dict[entity_phrase] = entities
                                        # self.nlq_phrase_kbentity_dict[entity_phrase] = entity_phrase

                                    except spotlight.SpotlightException as e:
                                        self.nlq_phrase_kbentity_dict[entity_phrase] = None

                                else: # when the modifier is a grammar term it emplies the predicate
                                    # use lemma of the word
                                    word_index = ast.literal_eval(re.match(r'^[\d]+', args[0]).group())
                                    # word = self.nlq_canonical.words[word_index].lemma
                                    word_temp = self.nlq_canonical.words[word_index].text
                                    vector = NLQCanonical.glove[word_temp].reshape(1, -1)
                                    value_prefix = get_property_using_cosine_similarity(vector, recalculate_numpy_property_vector=False)
                                    self.nlq_word_kb_predicate_dict[word] = value_prefix['value']

                    except Exception as e:
                        pass

        elif linker == 'custom_linker':
            # todo: may be required to implement
            self.nlq_token_entity_dict = {token.text: token.text for token in self.nlq_canonical.words}

        else:
            # no entity linking
            self.nlq_token_entity_dict = {k:v for (k, v) in zip([word.text for word in self.nlq_canonical.words],
                                                                [word.text for word in self.nlq_canonical.words])}
    @staticmethod
    def get_name_type_tuple(neod_lambda_term):
        pattern = r'(\w[\w\d._]*)\((.*)\)$'
        match = re.match(pattern, neod_lambda_term)
        if match:
            return match.group(1), match.group(2)
        else:
            return None

    def join_fnln_based_on_dependency(self, start_index, next_nwords=1):
        n_words = len(self.nlq_canonical.words)
        # skip_nwords = 0
        # for word in self.nlq_canonical.words[start_index:start_index+next_nwords]:
        for i in range(start_index,start_index+next_nwords):
            # compare universal dependency.
            if self.nlq_canonical.words[i].upos == self.nlq_canonical.words[i+1].upos:
                # check if the two words also has dependency relations
                if (self.nlq_canonical.words[i].governor == self.nlq_canonical.words[i+1].index) or (self.nlq_canonical.words[i].index == self.nlq_canonical.words[i+1].governor):
                    return f"{self.nlq_canonical.words[i].text}_{self.nlq_canonical.words[i+1].text}"


    def formalize_into_udeplambda(self):
        # This is shortcut, note the we take help from UDepLambda to create lambda logical form
        # from the natural question itself. So all this pipeline from natural language uptil tokenization is
        # now taken care off by the UDepLambda.
        # the lambda form is stored in the self.udep_lambda object variable.
        nlq = " ".join([word.text for word in self.nlq_canonical.words])
        with open("udepl_nlq.txt", 'w') as f:
            f.write(f'{{"sentence":"{nlq}"}}')

        res = subprocess.check_output("./run_udep_lambda.sh")

        # convert the bytecode into dictionary.
        self.udep_lambda = json.loads(res.decode('utf-8'))



    def translate_to_sparql(self, kg='dbpedia'):
        """
        The logical form in udep_lambda is translated into SPARQL.
        Note: Fomalize into SPARQL will note require reference to a Knowledge Graph for the purpose of
        denotation as that is already done. The KG is required to provide list of namespace for creating query-string
        during entity_linking stage of the parser.
        :return: a Query object
        """
        query = Query()
        variables_list = []
        event_triple_dict = {}
        # tuple_list = [s, p, o]
        for neod_lambda_term in self.udep_lambda['dependency_lambda'][0]:
            word_udep, type_tuple = NLQCanonical.get_name_type_tuple(neod_lambda_term)
            args = type_tuple.split(",")
            word_modifier = word_udep.split(".")
            word = word_modifier[0]
            try:
                modifier = word_modifier[1]
            except IndexError as e:
                modifier = None

            # if neod_term is Question then use the variable in it to be used as select variables
            if re.match(r'^QUESTION', word):
                # Take the arguments inside it and change them into SPARQL Variable, prepend with '?'
                variables_list.append(f'?{args[0]}')

            if re.match(r'[\d\w]+:[e]\s$', args[0]):
                if args[0] in event_triple_dict.keys(): # we are going to use index:e as key for the triple dict
                    # if modifier is not args
                    if not re.match(r'arg[\d]+', modifier):
                        if event_triple_dict[args[0]][1] is 'p':
                            event_triple_dict[args[0]][1] = URIRef(self.nlq_word_kb_predicate_dict[word])

                    else: # the word is argument to the predicate
                        # extract out the argument and its kb-entity and append
                        if event_triple_dict[args[0]][0] is 's':
                            event_triple_dict[args[0]][0]= URIRef(self.nlq_phrase_kbentity_dict[re.split(r'[.]', args[1])[1]])

                        elif event_triple_dict[args[0]][2] is 'o':
                            event_triple_dict[args[0]][2] = URIRef(self.nlq_phrase_kbentity_dict[re.split(r'[.]', args[1])[1]])

                else:
                    # the tuple is not initiated yet for the event index:e
                    if not re.match(r'arg[\d]+', modifier):
                        # when the neod_term is not argument it's predicate
                        tuple_list = ['s', 'p', 'o']
                        tuple_list[1] = URIRef(self.nlq_word_kb_predicate_dict[word])
                        event_triple_dict[args[0]] = tuple_list

                    else: # the word is argument to the predicate
                        # extract out the argument and its kb-entity and append
                        tuple_list = ['s', 'p', 'o']
                        tuple_list[0] = URIRef(self.nlq_phrase_kbentity_dict[re.split(r'[.]', args[1])[1]])
                        event_triple_dict[args[0]] =tuple_list

        query.select(variables_list)
        query.distinct()

        for key, triple_spo in event_triple_dict.items():
            if triple_spo[0] is 's':
                triple_spo[0] = BNode(variables_list[0])
            elif triple_spo[1] is 'p':
                triple_spo[1] = BNode(variables_list[0])
            elif triple_spo[2] is 'o':
                triple_spo[2] = BNode(variables_list[0])

            query.where(triple_spo)

        return query



    @staticmethod
    def dbpedia_property_similarity():
        pass


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
    # NLQCanonical object should be created first to load and save the glove word2vec data
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
