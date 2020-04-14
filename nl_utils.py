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

# Object may be created empty and later on modified using a bunch of operations on them. Or it get's created in its
# final form by some other class.
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

    def __init__(self, canonical_form):
        self.nlq_canonical = canonical_form

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
        return UGLogicalForm(self.udep_lambda)

class UGLogicalForm():
    def __init__(self, udep_lambda):
        self.udep_lambda = udep_lambda

    def translate_to_sparql(self, kg='dbpedia'):
        """
        The logical form in udep_lambda is translated into SPARQL.
        Note: translate into SPARQL will note require reference to a Knowledge Graph for the purpose of
        denotation, as that is already done.

        :return: a Query object
        """
        query = Query()
        variables_list = []

        # # we will create a dictionary of event_id, which is dictionary of dictionary {key:predicate, value:{s_list, o_list}}
        event_triples_dict = {}
        for neod_lambda_term in self.udep_lambda['dependency_lambda'][0]:
            # neod_lambda term in the atomic form takes a function name predicate or predicate.dependency
            # or predicate.args
            try:
                pred_dependency, type_entity = UGLogicalForm.get_atomic_name_atomic_args(neod_lambda_term)
            except TypeError as e_type:
                continue # when the name of the neod_lambda_term could n't be split into atomic_name and
                # atomic_arguments, better skip that term.

            type_entity = type_entity.split(",")
            pred_dependency = pred_dependency.split(".")

            # if neod_term is Question then use the variable in it as select variables of the query object
            if re.match(r'^QUESTION', pred_dependency[0]):
                # Take the arguments inside it and change them into SPARQL Variable, prepend with '?'
                variables_list.append(f'?{type_entity[0]}')

            # # identify the event_id and the predicate
            elif re.match(r'[\d]+:e', type_entity[0]):  # when identified for the first time, the event will
                # create new dictionary entry into the event_triples_dict.
                event_id = type_entity[0]
                if len(pred_dependency) == 1: # the atomic name only contains predicate
                    predicate = pred_dependency[0]
                    UGLogicalForm.update_plist(event_triples_dict, event_id, predicate , rdf_type='URIRef')

                elif len(pred_dependency) == 2: # the atomic name coontains predicate and the dependency-relations
                    # or it contains predicate and arg0-1
                    predicate = pred_dependency[0]
                    dependency_relations = pred_dependency[1]
                    try:
                        entity = type_entity[1].split('.')[1]
                        rdf_type = 'URIRef'
                    except IndexError as e_index:  # index error
                        # the term is a varaible
                        entity = type_entity[1]
                        rdf_type = 'BNode'

                    if not re.match(r'arg[\d]+', dependency_relations): # the term is dependency-relation
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
                        else: # put the entity into slist, however, check for lenght of the two lists.
                            # the triple are formed when the list are balanced.
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)

                    else: # the second term in atomic_name is arg[\d]
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
                        else:  # put the entity into slist
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)

                elif len(pred_dependency) ==3: # the third term is preopositional modifier,
                    # second term the dependency relations
                    predicate = pred_dependency[0]
                    dependency_relations = pred_dependency[1]
                    prepositional_modifier = pred_dependency[2]
                    try:
                        entity = type_entity[1].split('.')[1]
                        rdf_type = 'URIRef'
                    except IndexError as e_index:  # index error
                        # the term is a varaible
                        entity = type_entity[1]
                        rdf_type = 'BNode'

                    if not re.match(r'arg[\d]+', dependency_relations):  # the term is dependency-relation
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity,
                                                       rdf_type=rdf_type)
                        else:  # put the entity into slist, however, check for length of the two lists.
                            # the triple are formed when the list are balanced.
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity,
                                                                rdf_type=rdf_type)

                    else:  # the second term in atomic_name is arg[\d]
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity,
                                                       rdf_type=rdf_type)
                        else:  # put the entity into slist
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity,
                                                                rdf_type=rdf_type)

            # if atomic expression is of type ':s' i.e. a type relations
            elif re.match(r'[\s]*[\d]+:s[\s]*$', type_entity[0]):
                # the atomic_expression is of type ':s'
                type_id = type_entity[0]
                variable_name = type_entity[1]
                type_label = pred_dependency[0]

                if len(pred_dependency) == 1: # only the type name is given, and the atomic_args contain variable
                    UGLogicalForm.update_plist(event_triples_dict, type_id, 'a', rdf_type='URIRef') # dbpedia uses a for type
                    UGLogicalForm.update_olist(event_triples_dict, type_id, 'a', type_label, rdf_type='URIRef')
                    UGLogicalForm.update_slist(event_triples_dict, type_id, 'a', variable_name, rdf_type='BNode')

                elif len(pred_dependency) == 2: # atomic expression for type ':s' don't have arg[\d]
                    pass

        query.select(variables_list)
        query.distinct()
        spo_triples = []
        for neod_type, kp_vso in event_triples_dict.items(): # kp_vso a dictionary of predicate as key and s_list, o_list as value
            # if neod_type is an event
            if re.match(r'[\d]+:e[\s]*', neod_type):
                # kp_vso: note that there may be many predicates and each predicate will have an associated s_list
                # and o_list, which in turn may form many subject-object pairs. There fore we create a different static
                # function to process it.
                spo_triples = spo_triples + UGLogicalForm.create_spo_triples(kp_vso)
            elif re.match(r'\s*[\d]+:s[\s]*', neod_type):
                spo_triples = spo_triples + UGLogicalForm.create_spo_triples(kp_vso)

        query.where(spo_triples)
        return UGSPARQLGraph(query), # query.get_query_string()

    @staticmethod
    def get_atomic_name_atomic_args(neod_lambda_term):
        # pattern = r'(\w[\w\d._]*)\((.*)\)$'
        pattern = r'(\w[\w\d._\']*)\((.*)\)$' # to include apostrophe 's
        match = re.match(pattern, neod_lambda_term)
        if match:
            return match.group(1), match.group(2)
        else:
            return None

    @staticmethod
    def update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef'):
        # we are going to form a dictionary of dictionary of dictionary : event_id:predicate:{s_list, o_list}
        try:
            if predicate in event_triples_dict[event_id].keys():
                pass
            else :
                event_triples_dict[event_id][predicate] = {}
                # if rdf_type is 'URIRef':
                #     event_triples_dict[event_id][URIRef(predicate)] = {}
                # elif rdf_type is 'BNode':
                #         event_triples_dict[event_id][BNode(predicate)] = {}
        except KeyError as event_error:
            event_triples_dict[event_id] = {}
            event_triples_dict[event_id][predicate] = {}
            # if rdf_type is 'URIRef':
            #     event_triples_dict[event_id][URIRef(predicate)] = {}
            # elif rdf_type is 'BNode':
            #     event_triples_dict[event_id][BNode(predicate)] = {}

    @staticmethod
    def update_bw_slist_olist(event_triples_dict, event_id, predicate, entity, rdf_type=URIRef):
        try:
            s_list_len = len(event_triples_dict[event_id][predicate]['s_list'])
            o_list_len = len(event_triples_dict[event_id][predicate]['o_list'])
            if s_list_len == o_list_len:
                UGLogicalForm.update_slist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
            elif s_list_len > o_list_len:
                UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
            else:
                UGLogicalForm.update_slist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
        except KeyError as e_key:
            if e_key.args[0] is 'o_list':
                UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
            elif e_key.args[0] is 's_list':
                UGLogicalForm.update_slist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)

    @staticmethod
    def update_slist(event_triples_dict, event_id, predicate, subject, rdf_type='URIRef'):
        try:
            sub_list_len = len(event_triples_dict[event_id][predicate]['s_list'])
            if sub_list_len >= 1:  # the subject_list already has a relation corresponding to the event.
                if rdf_type is 'URIRef':
                    event_triples_dict[event_id][predicate]['s_list'].append(URIRef(subject))
                elif rdf_type is 'BNode':
                    event_triples_dict[event_id][predicate]['s_list'].append(BNode(subject))
        except KeyError as ek:
            # create s_list
            if rdf_type is 'URIRef':
                event_triples_dict[event_id][predicate]['s_list'] = [URIRef(subject)]  # assign the subject
            elif rdf_type is 'BNode':
                event_triples_dict[event_id][predicate]['s_list'] = [BNode(subject)]  # assign the subject

    @staticmethod
    def exists_in_slist(event_triples_dict, event_id, predicate, entity):
        try:
            if entity in event_triples_dict[event_id][predicate]['s_list']:
                return True
        except KeyError as e_key:
            # if KeyError is encountered, that mean s_list doesn't exist
            return False

    @staticmethod
    def update_olist(event_triples_dict, event_id, predicate, object, rdf_type='URIRef'):
        try:
            object_list_len = len(event_triples_dict[event_id][predicate]['o_list'])
            if object_list_len >= 1:  # the object_list already has a entity corresponding to the event.
                if rdf_type is 'URIRef':
                    event_triples_dict[event_id][predicate]['s_list'].append(URIRef(object))
                elif rdf_type is 'BNode':
                    event_triples_dict[event_id][predicate]['s_list'].append(BNode(object))
        except KeyError as ek:
            # create the object_list with entity_id/type_id
            if rdf_type  is 'URIRef':
                event_triples_dict[event_id][predicate]['o_list'] = [URIRef(object)] # assign the object
            elif rdf_type is 'BNode':
                event_triples_dict[event_id][predicate]['o_list'] = [BNode(object)]  # assign the object

    @staticmethod
    def exists_in_olist(event_triples_dict, event_id, predicate, entity):
        try:
            if entity in event_triples_dict[event_id][predicate]['o_list']:
                return True
        except KeyError as e_key:
            # when KeyError encountered, that mean object_list doen't exists.
            return False

    @staticmethod
    def create_spo_triples(kp_vso):
    # kp_vso: note that there may be many predicates and each predicate will have an associated s_list
    # and o_list, which in turn may form many subject-object pairs.
        spo_triples = []
        for predicate, slist_olist_dict in kp_vso.items():
            try:
                assert len(slist_olist_dict['s_list']) == len(slist_olist_dict['o_list']), "hanging edge: missing subject or object node"
                predicate = URIRef(predicate)
                spo_triples = spo_triples + [(s, predicate, o) for s,o in zip(slist_olist_dict['s_list'], slist_olist_dict['o_list'])]
            except (AssertionError, KeyError) as e:
                pass

        return spo_triples


class UGSPARQLGraph:
    # UGSPARQLGraph will take in a sparql query object, which can also be treated as a graph object.
    # There are a bunch of operation that would be performed on the graph object to get it into a
    # final grounded logical form using predicate and entity linking.

    # Using GloVe-6B trained on 6Billion tokens, contains 400k vocabulary
    # loading once for the object is created, and later to be used by entity linker
    # or predicate linker
    glove_loaded = False  # flag to load the keyedvector from file once
    glove = None  # the keyedvector stored using memory map for fast access

    def __init__(self, ug_query):
        self.query_graph = ug_query
        self.nlq_phrase_kbentity_dict = {}
        self.nlq_word_kb_predicate_dict = {}
        if not UGSPARQLGraph.glove_loaded:
            try:
                UGSPARQLGraph.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
                UGSPARQLGraph.glove_loaded = True
            except Exception as e:
                glove_loading_kv = KeyedVectors.load_word2vec_format(
                    "./pt_word_embedding/glove/glove.6B.50d.w2vformat.txt")
                glove_loading_kv.save('./glove_gensim_mmap')
                UGSPARQLGraph.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
                # UGSPARQLGraph.glove.syn0norm = NLQCanonical.glove.syn0  # prevent recalc of normed vectors
                # UGSPARQLGraph.glove.most_similar('stuff')  # any word will do: just to page all in
                # Semaphore(0).acquire()  # just hang until process killed
                UGSPARQLGraph.glove_loaded = True

    def __str__(self):
        return self.query_graph.get_query_string()

    def ground_entity(self, linker=None, kg=None):
        """
        entity linking using dbpedia Spotlight
        Entity linker by definition must have reference to Knowledge Graph, whence it bring list of denotation for
        the token.
        :return:
        """
        if linker == 'spotlight' and kg == 'dbpedia':
            if self.udep_lambda:  # if the formalization was true. start the linker
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
                            if modifier is not None:  # if modifier is not present, it's Question Term or Where term
                                if re.match(r'arg[\d+]', modifier):  # if modifier is an arg[\d] term,
                                    # it contains entities inside the brackets
                                    entity_phrase = re.split(r'[.]', args[1])[1]
                                    try:
                                        entities = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate',
                                                                      entity_phrase,
                                                                      confidence=0.0, support=0)
                                        self.nlq_phrase_kbentity_dict[entity_phrase] = entities
                                        # self.nlq_phrase_kbentity_dict[entity_phrase] = entity_phrase

                                    except spotlight.SpotlightException as e:
                                        self.nlq_phrase_kbentity_dict[entity_phrase] = None

                                else:  # when the modifier is a grammar term it emplies the predicate
                                    # use lemma of the word
                                    word_index = ast.literal_eval(re.match(r'^[\d]+', args[0]).group())
                                    # word = self.nlq_canonical.words[word_index].lemma
                                    word_temp = self.nlq_canonical.words[word_index].text
                                    vector = NLQCanonical.glove[word_temp].reshape(1, -1)
                                    value_prefix = get_property_using_cosine_similarity(vector,
                                                                                        recalculate_numpy_property_vector=False)
                                    self.nlq_word_kb_predicate_dict[word] = value_prefix['value']

                    except Exception as e:
                        pass

        elif linker == 'custom_linker':
            # todo: may be required to implement
            self.nlq_token_entity_dict = {token.text: token.text for token in self.nlq_canonical.words}

        else:
            # no entity linking
            self.nlq_token_entity_dict = {k: v for (k, v) in zip([word.text for word in self.nlq_canonical.words],
                                                                 [word.text for word in self.nlq_canonical.words])}

    def ground_predicate(self, linker=None, kg=None): #todo
        pass

    def join_fnln_based_on_dependency(self, start_index, next_nwords=1):
        n_words = len(self.nlq_canonical.words)
        # skip_nwords = 0
        # for word in self.nlq_canonical.words[start_index:start_index+next_nwords]:
        for i in range(start_index, start_index + next_nwords):
            # compare universal dependency.
            if self.nlq_canonical.words[i].upos == self.nlq_canonical.words[i + 1].upos:
                # check if the two words also has dependency relations
                if (self.nlq_canonical.words[i].governor == self.nlq_canonical.words[i + 1].index) or (
                        self.nlq_canonical.words[i].index == self.nlq_canonical.words[i + 1].governor):
                    return f"{self.nlq_canonical.words[i].text}_{self.nlq_canonical.words[i + 1].text}"

    def get_g_sparql_graph(self): #todo
        # return GroundedSPARQLGraph(sparql_query)
        pass


class GroundedSPARQLGraph: #todo
    pass


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
