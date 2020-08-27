import copy
import os
import logging
from rdflib import BNode, URIRef

#from candidate_generation.searchIndex import entitySearch, propertySearch, ontologySearch
from dbpedia_lib.db_utils import get_property_using_cosine_similarity
from dl_lib.spo_disambiguator import cross_emb_predictor 
#from udep_lib.nlqtokens import question
from udep_lib.g_sparql_graph import GroundedSPARQLGraph
from udep_lib.sparql_builder import Query

logger = logging.getLogger(__name__)

class UGSPARQLGraph:
    # UGSPARQLGraph will take in a sparql query object, which can also be treated as a graph object.
    # There are a bunch of operation that would be performed on the graph object to get it into a
    # final grounded logical form using predicate and entity linking.

    # Using GloVe-6B trained on 6Billion tokens, contains 400k vocabulary
    # loading once for the object is created, and later to be used by entity linker
    # or predicate linker
    # glove_loaded = False  # flag to load the keyedvector from file once
    # glove = None  # the keyedvector stored using memory map for fast access

    def __init__(self, ug_query, grounded_topk=10):
        # with grounded_topk we are going to limit number of final sparql queries available for evaluation to us. 
        # these sparql-quereis can be run against sparql endpoint to provide MRR at say 5, 10. 
        self.query_graph = ug_query
        g_query_tmp = copy.deepcopy(ug_query) # this is just to get the various attribute copied.
        # and remove the basic graph pattern
        g_query_tmp.empty_bgp() 
        self.g_query_topk = [copy.deepcopy(g_query_tmp) for i in range(grounded_topk)]
        
        logger.info(f'ug_sparql: {self.query_graph.get_query_string()}')

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

    def ground_spo(self, question=None, annotation=None, linker=None, kg=None): # this will generate a set of candidate spo,
        # which are later passed throug a disambiguation stage to obtain a final spo-triple
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
            if linker == 'spotlight':
                for sub, pred, obj in self.query_graph._data:
                    # ground subject
                    subject_entities = UGSPARQLGraph.ground_so_spotlight(self.query_graph, sub)
                    # ground object, before passing them over strip off any white space.
                    object_entities = UGSPARQLGraph.ground_so_spotlight(self.query_graph, obj)
                    # ground predicate
                    predicate_property = UGSPARQLGraph.ground_predicate_w2v(pred)
                    # the triples obtained affter linking may need to be processed, or reranked.
                    self.g_query.add_triple((subject_entities, predicate_property, object_entities))

            elif linker == 'elasticsearch':
                for sub, pred, obj in self.query_graph._data:
                    logger.info(f'ug spo: {sub}, {pred}, {obj}')
                    # ground subject
                    rdf_type_s, subject_entities_list = UGSPARQLGraph.ground_so_elasticsearch(self.query_graph, sub, kg=kg)
                    # ground object, before passing them over strip off any white space.
                    # the object may belong to ontology whenever predicate is letter 'a'
                    rdf_type_o, object_entities_list = UGSPARQLGraph.ground_so_elasticsearch(self.query_graph, obj, onto_hint=pred, kg=kg)
                    # ground predicate, predicate is assumed given, never a blank node. 
                    # therefore not returning any type information.
                    # when predicate is letter 'a' specifying an ontology, this function will return 'a' withought going into searching 
                    # property-index in elasticsearch.
                    predicate_property_list = UGSPARQLGraph.ground_predicate_elasticsearch(pred, onto_hint=obj, kg=kg)
                    
                    # the returned items from elasticsearch are in the form of list of list-elements.
                    if kg == 'dbpedia':
                        # for dbpedia the entries only have lable uri score score while it changes for freebase
                        subject_entities_list_sorted = sorted(subject_entities_list, key=lambda x: x[2], reverse=True)
                        object_entities_list_sorted = sorted(object_entities_list, key=lambda  x: x[2], reverse=True)
                        predicate_property_list_sorted = sorted(predicate_property_list, key=lambda x: x[2], reverse=True)

                    if kg == 'freebase':
                        # for freebase the predicate also has description field besides label and uri, entity indes are same
                        subject_entities_list_sorted = sorted(subject_entities_list, key=lambda x: x[2], reverse=True)
                        object_entities_list_sorted = sorted(object_entities_list, key=lambda  x: x[2], reverse=True)
                        predicate_property_list_sorted = sorted(predicate_property_list, key=lambda x: x[3], reverse=True)

                    logger.info(f'top-es spo: ({subject_entities_list_sorted[0]}, {predicate_property_list_sorted[0]}, {object_entities_list_sorted[0]})')
                    # we can create a combination of s, p, o such that, the high scoring element in the
                    # set of S, P, O are together.
                    topk_es = 10 #use this to select topk search resutls from elastic search
                    # the choice os topk_es can slow down processing as it will create topk_es^3 spo triplets that need to be
                    # re-ranked by spo-disambiguator
                    disambiguated_spo_topk = UGSPARQLGraph.disambiguate_using_cotext(question, subject_entities_list_sorted[:topk_es],
                                                                                predicate_property_list_sorted[:topk_es], object_entities_list_sorted[:topk_es], rdf_type_s, rdf_type_o)
                    #The set of candidate-spos we will get above would create for a given spo-triple
                    for i, disambiguated_spo in enumerate(disambiguated_spo_topk):
                        disambiguated_spo_with_rdfterm = ['s', 'o', 'p']
                        if rdf_type_s == 'URIRef':
                            disambiguated_spo_with_rdfterm[0] = URIRef(disambiguated_spo[0])
                        elif rdf_type_s == 'BNode':
                            disambiguated_spo_with_rdfterm[0] = BNode(disambiguated_spo[0])
                        else: # in case Exception occurs, we will receive None 
                            disambiguated_spo_with_rdfterm[0] = BNode()
                        
                        if rdf_type_o == 'URIRef':
                            disambiguated_spo_with_rdfterm[2] = URIRef(disambiguated_spo[2])
                        elif rdf_type_o == 'BNode':
                            disambiguated_spo_with_rdfterm[2] = BNode(disambiguated_spo[2])
                        else: # in case Exception occurs, we will receive None 
                            disambiguated_spo_with_rdfterm[2] = BNode()

                        #predicate will carry URIRef
                        disambiguated_spo_with_rdfterm[1] = URIRef(disambiguated_spo[1])
                        self.g_query_topk[i].add_triple(tuple(disambiguated_spo_with_rdfterm))

            elif linker == 'queryKB':
                entity_name = annotation['name']
                entity_mid = annotation['annotation']
                # storing triplet_original as list of ungrounded tuples
                # storing triplet_grounded as list of list of tuples, 
                triplets = {'triplets_ug':[], 'triplet_grounded':[]}
                for sub, pred, obj in self.query_graph._data:
                    triplet = [sub, pred, obj]
                    triplets['triplets_ug'].append(tuple(triplet))

                # once the triplets are obtained, we need to ground them by querying the KB for all possible relation that 
                # emanates from the annotated entities. 
                # the entity_name might directly appear in a triplet or it could be mentioned in the query.nodes_type
                # for the first case
                for triplet in triplets['triplets_ug']:
                    triplet_candidates = []
                    for so_position, surface_name in enumerate([triplet[0], triplet[2]]):
                        if re.search(surface_name, entity_name):
                            # then go for querying the KB for all the triplets emanating from or coming to entity
                            # the ground_spo_query_KB may return only topk triplets, based on how the BERT-Softmax
                            # re-ranker finds their similarity in it's embedding space
                            # ground_triplet_queryKB will return a list corresponding to the ungrounded triplet.
                            # letter we may as well accumulate re-ranker score to find do beam search when there are multiple
                            # ungrounded triplets
                            triplet_candidates = triplet_candidates + ground_triplet_queryKB(triplet, entity_mid, so_position=so_position, kg=kg) 
                            
                        # check if surface_name has a correponding common_name/type_name in the query.nodes_type
                        elif surface_name in self.query_graph.nodes_type.keys():
                            surface_name = self.query_graph.nodes_type[surface_name]
                            if re.search(surface_name, entity_name):
                                triplet_candidates = triplet_candidates + ground_triplet_queryKB(triplet, entity_mid, so_position=so_position, kg=kg) 

                    triplets['triplet_grounded'].append(triplet_candidates)

    def get_g_sparql_graph(self):
        #todo from the list of candidates spo we need to disambiguate to obtain only one spo-triple
        return GroundedSPARQLGraph(self.g_query_topk)

    def get_g_sparql_query(self):
        sparql_query = [g_query.get_query_string() for g_query in self.g_query_topk]
        logger.info(f'g-SPARQL: {sparql_query}') 
        return sparql_query

    @staticmethod
    def ground_so_elasticsearch(query_graph, so, onto_hint=None, kg='dbpedia'):
        if kg=='dbpedia':
            from candidate_generation.searchIndex import entitySearch, propertySearch, ontologySearch
        if kg=='freebase':
            from candidate_generation_fb.searchIndex import entitySearch, propertySearch, ontologySearch

        if isinstance(so, BNode):
            rdf_type = 'BNode'
        elif isinstance(so, URIRef):
            rdf_type = 'URIRef'

        try:
            if query_graph.has_variable(so.strip()):
                so_entities = [[f'{so}', ' ', 0, 0]] # return as a list of list confirming to the output of the
                # elasticsearch which has label, uri, score and another elemen 0
            elif rdf_type == 'BNode':
                so_entities = [[f'{so}', ' ', 0, 0]]
            elif onto_hint == 'a':
                # do a search in ontology
                so_entities = ontologySearch(f'{so}')
            else:
                so_entities = entitySearch(f'{so}')
                # so_entities = so

            return rdf_type, so_entities
        except:
            # if entity linking fails, a graounded query can't be obtained
            # however to stop the bert based cross embedder from failing we
            # will need to pass empty space.
            # so_entities = ' ',
            return None, [[' ', ' ', 0, 0]]



    @staticmethod
    def ground_predicate_elasticsearch(predicate, onto_hint=None, kg='dbpedia'):
        if kg=='dbpedia':
            from candidate_generation.searchIndex import entitySearch, propertySearch, ontologySearch
        if kg=='freebase':
            from candidate_generation_fb.searchIndex import entitySearch, propertySearch, ontologySearch
        onto_wh = {'where':['location', 'place', 'country', 'city'],
                   'when': ['date', 'year', 'time', 'hour'],
                   'who': ['person', 'man', 'woman'],
                   'what': ['thing', 'tool'],
                   'which': ['thing', 'type'],
                   'how': ['quantity', 'weight', 'distance']}

        try:
            if f'{predicate}' == 'a':
                # when predicate is 'a' we return it as is. 
                # also make the uri as 'a'
                return [[f'a', 'a', 0, 0]]
            else:
                db_properties = propertySearch(f'{predicate}')
                return db_properties

        except:
            empty_list = []
            return empty_list

    @staticmethod
    def ground_triplet_queryKB(triplet, entity_mid, so_position=so_position, kg=kg): 

        # if hops==1: # will do one hop first
        if so_position ==0:
            query_string = f'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?obj WHERE{{ns:{entity_mid} ?rel ?obj}}' 
            Query.run(query_string, kg=kg)

        elif so_position ==2
            query_string = f'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?sub ?rel WHERE{{ ?sub ?rel ns:{entity_mid}}}' 
            Query.run(query_string, kg=kg)


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
    def disambiguate_using_cotext(question, subject_entities_list_sorted, predicate_property_list_sorted, object_entities_list_sorted, rdf_type_s=None, rdf_type_o=None):
        input_dict = {'question':question, 'spos':[], 'spos_label':[]}
        # todo we can do some innovative mixing to create spo triple from the separate list of s, o, p.
        # sort the list by thrid value of the sublist which is the score as returned by the elastic
        # search.

        for si in subject_entities_list_sorted:
            for pi in predicate_property_list_sorted:
                for oi in object_entities_list_sorted:
                    tmp_list_spos = ['','','']
                    tmp_list_spos_label = ['','','']
                    # don't want blank-node interfere with the cross-embedding score therefore passing it as empty-string
                    if rdf_type_s == 'BNode':
                        # would like to pass on the label of the blank node as uri,
                        # so that it could be passed through the uri, but we keep the label an empty-string 
                        tmp_list_spos[0] = si[0] # uri is name of the variable 
                        tmp_list_spos_label[0] = '' # label is empty list
                    else:
                        tmp_list_spos[0] = si[1]
                        tmp_list_spos_label[0] = si[0]

                    if rdf_type_o == 'BNode':
                        tmp_list_spos[2] = oi[0] # uri is name of the variable 
                        tmp_list_spos_label[2] = '' # label is empty list
                    else:
                        tmp_list_spos[2] = oi[1]
                        tmp_list_spos_label[2] = oi[0]

                    tmp_list_spos[1] = pi[1]
                    tmp_list_spos_label[1] = pi[0]
                    
                    input_dict['spos'].append(tmp_list_spos)
                    input_dict['spos_label'].append(tmp_list_spos_label)

        reranked_spos = cross_emb_predictor(input_dict=input_dict, write_pred=False)
        reranked_spos_sorted = sorted(reranked_spos[0], key=lambda x: x['cross_emb_score'], reverse=True)
        only_uri_list_topk = [only_uri['spo_triple_uri'] for only_uri in reranked_spos_sorted[:10]]
        logger.info(f"cross-emb spo: {only_uri_list_topk}")
        logger.debug(f"cross-emb spo: {reranked_spos_sorted}")
        return only_uri_list_topk

    @staticmethod
    def flatten_search_result(nested_list):
        result = []
        if not nested_list:
            return
        for element in nested_list:
            if isinstance(element, list):
                result.append(element)
            elif not element:  # element is NoneType
                continue
            else:
                result = result + flatten_search_result(element)
        return result
