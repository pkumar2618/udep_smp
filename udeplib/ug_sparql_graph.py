import copy
import os

from rdflib import BNode, URIRef

from candidate_generation.searchIndex import entitySearch, propertySearch
from dbpedialib.db_utils import get_property_using_cosine_similarity
from dl_modules.spo_disambiguator import cross_emb_predictor
from udeplib.nlqtokens import question
from udeplib.g_sparql_graph import GroundedSPARQLGraph


class UGSPARQLGraph:
    # UGSPARQLGraph will take in a sparql query object, which can also be treated as a graph object.
    # There are a bunch of operation that would be performed on the graph object to get it into a
    # final grounded logical form using predicate and entity linking.

    # Using GloVe-6B trained on 6Billion tokens, contains 400k vocabulary
    # loading once for the object is created, and later to be used by entity linker
    # or predicate linker
    # glove_loaded = False  # flag to load the keyedvector from file once
    # glove = None  # the keyedvector stored using memory map for fast access

    def __init__(self, ug_query):
        self.query_graph = ug_query
        self.g_query = copy.deepcopy(ug_query) # this is just to get the various attribute copied.
        # and remove the basic graph pattern
        self.g_query.empty_bgp()
        # self.nlq_phrase_kbentity_dict = {}
        # self.nlq_word_kb_predicate_dict = {}
        # if not UGSPARQLGraph.glove_loaded:
        #     try:
        #         UGSPARQLGraph.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
        #         UGSPARQLGraph.glove_loaded = True
        #     except Exception as e:
                # glove_loading_kv = KeyedVectors.load_word2vec_format(
                #     "./pt_word_embedding/glove/glove.6B.50d.w2vformat.txt")
                # glove_loading_kv.save('./glove_gensim_mmap')
                # UGSPARQLGraph.glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
                # UGSPARQLGraph.glove.syn0norm = NLQCanonical.glove.syn0  # prevent recalc of normed vectors
                # UGSPARQLGraph.glove.most_similar('stuff')  # any word will do: just to page all in
                # Semaphore(0).acquire()  # just hang until process killed
                # UGSPARQLGraph.glove_loaded = True

    def __str__(self):
        return self.query_graph.get_query_string()

    def get_ug_sparql_query_string(self):
        return self.query_graph.get_query_string()

    def ground_spo(self, linker=None, kg=None): # this will generate a set of candidate spo,
        # which are lated passed throug a disambiguation stage to obtain a final spo-triple
        """
        entity linking using dbpedia Spotlight
        Entity linker by definition must have reference to Knowledge Graph, whence it bring list of denotation for
        the token.
        :return:
        """
        # if linker == 'spotlight' and kg == 'dbpedia':
        if self.query_graph:  # if the formalization was true. start the linker
            # linking entities using dbpedia sptolight
            # identify entity phrase from the predicate arguments
            # and
            # linking dbpedia_predicate using cosine similarity on word2vec
            # find the event type tokens according to Neo-Davidsonia grammar
            for sub, pred, obj in self.query_graph._data:
                if linker == 'spotlight':
                    # ground subject
                    subject_entities = UGSPARQLGraph.ground_so_spotlight(self.query_graph, sub)
                    # ground object, before passing them over strip off any white space.
                    object_entities = UGSPARQLGraph.ground_so_spotlight(self.query_graph, obj)
                    # ground predicate
                    predicate_property = UGSPARQLGraph.ground_predicate_w2v(pred)
                    # the triples obtained affter linking may need to be processed, or reranked.
                    self.g_query.add_triple((subject_entities, predicate_property, object_entities))

                elif linker == 'elasticsearch':
                    # ground subject
                    subject_entities_list = UGSPARQLGraph.ground_so_elasticsearch(self.query_graph, sub)
                    # ground object, before passing them over strip off any white space.
                    object_entities_list = UGSPARQLGraph.ground_so_elasticsearch(self.query_graph, obj)
                    # ground predicate
                    predicate_property_list = UGSPARQLGraph.ground_predicate_elasticsearch(pred, onto_hint=obj)

                    # we can create a combination of s, p, o such that, the high scoring element in the
                    # set of S, P, O are together.
                    disambiguated_spo = UGSPARQLGraph.disambiguate_using_cotext(question, subject_entities_list,
                                                                          predicate_property_list, object_entities_list)
                    #The set of candidate-spos we will get above would create for a given spo-triple
                    self.g_query.add_triple(disambiguated_spo)

    def get_g_sparql_graph(self):
        #todo from the list of candidates spo we need to disambiguate to obtain only one spo-triple

        return GroundedSPARQLGraph(self.g_query)

    @staticmethod
    def ground_so_elasticsearch(query_graph, so):
        if isinstance(so, BNode):
            rdf_type = 'BNode'
        elif isinstance(so, URIRef):
            rdf_type = 'URIRef'

        try:
            if query_graph.has_variable(so.strip()):
                so_entities = so
            else:
                so_entities = entitySearch(so)
                # so_entities = so

            return rdf_type, so_entities
        except:
            # if entity linking fails, a graounded query can't be obtained
            # however to stop the bert based cross embedder from failing we
            # will need to pass empty space.
            # so_entities = ' ',
            return ' ', ' '



    @staticmethod
    def ground_predicate_elasticsearch(predicate, onto_hint=None):
        onto_wh = {'where':['location', 'place', 'country', 'city'],
                   'when': ['date', 'year', 'time', 'hour'],
                   'who': ['person', 'man', 'woman'],
                   'what': ['thing', 'tool'],
                   'which': ['thing', 'type'],
                   'how': ['quantity', 'weight', 'distance']}

        predicates = []
        db_properties = []
        try:
            if predicate == 'a':
                if onto_hint in onto_wh.keys():
                    #todo Require ontology, may be expansion or contraction of the query_graph
                    predicates = onto_wh[onto_hint]
            else:
                predicates.append(predicate)

            for pred in predicates:
                db_properties.append(propertySearch(pred))

            return db_properties

        except:
            empty_list = []
            return empty_list

    @staticmethod
    def ground_so_spotlight(query_graph, so):
        if isinstance(so, BNode):
            rdf_type = 'BNode'
        elif isinstance(so, URIRef):
            rdf_type = 'URIRef'

        try:
            if query_graph.has_variable(so.strip()):
                so_entities = so
            else:
                # so_entities = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate',
                #                                       so,
                #                                       confidence=0.0, support=0)
                so_entities = so
        except spotlight.SpotlightException as e:
            so_entities = so

        if rdf_type == 'BNode':
            return BNode(so_entities)
        elif rdf_type == 'URIRef':
            return URIRef(so_entities)


    @staticmethod
    def ground_predicate_w2v(predicate):
        try:
            glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
        except Exception as e:
            glove_loading_kv = KeyedVectors.load_word2vec_format(
                "./glove/glove.840B.300d.w2vformat.txt")
            glove_loading_kv.save('./glove_gensim_mmap')
            glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')

        try:
            if predicate == 'a':
                predicate_property = predicate
            else:
                vector = glove[predicate].reshape(1, -1)
                value_prefix = get_property_using_cosine_similarity(vector, top_n=1)
                predicate_property = URIRef(value_prefix['value'])
        except Exception as e:
            predicate_property = predicate

        return predicate_property

    @staticmethod
    def disambiguate_using_cotext(question, subject_entities_list, predicate_property_list, object_entities_list):
        input_dict = {'question':question, 'spos':[], 'spos_label':[]}
        # todo we can do some innovative mixing to create spo triple from the separate list of s, o, p.
        # sort the list by thrid value of the sublist which is the score as returned by the elastic
        # search.
        subject_entities_list_sorted = sorted(subject_entities_list, key=lambda x: x[2])
        predicate_property_list_sorted = sorted(predicate_property_list, key=lambda x: x[2])
        object_entities_list_sorted = sorted(object_entities_list, key=lambda  x: x[2])
        for si in subject_entities_list_sorted:
            for pi in predicate_property_list_sorted:
                for oi in object_entities_list_sorted:
                    input_dict['spos'].append([subject_entities_list_sorted[si][1], predicate_property_list_sorted[pi][1],
                                              object_entities_list_sorted[oi][1]])

                    input_dict['spos_label'].append([subject_entities_list_sorted[si][0], predicate_property_list_sorted[pi][0],
                                              object_entities_list_sorted[oi][0]])
        model_file_path = os.path.join(os.getcwd(),"dl_modules", "model.th")
        reranked_spos = cross_emb_predictor(input_dict=input_dict, model_file=model_file_path, write_pred=False)
        reranked_spos_sorted = sorted(reranked_spos, key=lambda x: x['cross_emb_score'], reverse=True])
        return reranked_spos_sorted[0]['spo_triple_uri']