import re
import json
import pandas as pd
import rdflib

from rdflib.term import Node, BNode, URIRef
from rdflib.namespace import XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.algebra import pprintAlgebra, translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.namespace import FOAF
from rdflib.graph import Graph, ConjunctiveGraph, plugin
from rdflib.compare import isomorphic, similar
from rdflib.tools.rdf2dot import rdf2dot
from graphviz import Source
from rdflib.tools.graphisomorphism import IsomorphicTestableGraph
from rdflib.serializer import Serializer


class QueryGraph:
    def __init__(self, sparql_query, prefix_bindings):
        """sparql_query is query string, which will be converted into sparqlAlgebra
        during construction.
        """

        ## using direct algebra
        query_algebra = prepareQuery(sparql_query, initNs=prefix_bindings)
        # pprintAlgebra(query_algebra)

        self.algebra = query_algebra.algebra

        self.query_type = query_algebra.algebra.name
        # print(query_type)

        self.parameter_variables = query_algebra.algebra['PV']
        # print(*parameter_variables, sep=', ')

        bgp_expression = self.find_p_name_list('BGP')
        # self.bgp_variables = query_algebra.algebra['p']['p']['p']['_vars']

        try:
            # Create queryGraph
            # self.graph = Graph()
            # for triple in triples_list:
            #     self.graph.add(triple)

            # Create processed queryGraph
            self.processed_graph = Graph()

            self.bgp_variables = bgp_expression['_vars']
            self.triples_list = bgp_expression['triples']
            self.processed_triples = []

            bgp_variables_dict = {}
            var_count = 0
            param_variables_dict = {}
            param_var_count = 0

            for variable in self.bgp_variables:
                if variable not in self.parameter_variables:
                    var_count += 1
                    bgp_variables_dict[variable] = "var%d" % var_count

            for pv in self.parameter_variables:
                param_var_count += 1
                param_variables_dict[pv] = "param_var%d" % param_var_count

            # anonymizing uri's
            uri_seen_dict = {}
            uri_seen_count = 0
            for tuple_spo in self.triples_list:
                processed_tuple_spo = []
                for item in tuple_spo:
                    if item in param_variables_dict.keys():
                        processed_tuple_spo.append(URIRef(param_variables_dict[item]))

                    elif item in bgp_variables_dict.keys():
                        processed_tuple_spo.append(URIRef(bgp_variables_dict[item]))

                    elif isinstance(item, URIRef):
                        # processed_tuple_spo.append(item)
                        if item not in uri_seen_dict.keys():
                            uri_seen_count += 1
                            uri_seen_dict[item] = "uri_ref_%d" %uri_seen_count
                            processed_tuple_spo.append(URIRef(uri_seen_dict[item]))
                        elif item in uri_seen_dict.keys():
                            processed_tuple_spo.append(URIRef(uri_seen_dict[item]))

                processed_tuple_spo = tuple(processed_tuple_spo)
                self.processed_triples.append(processed_tuple_spo)
                self.processed_graph.add(processed_tuple_spo)

        except Exception as e:
            pass


    def find_p_name_list(self, p_name):

        def find_p_name(algebra_expression, p_name):
            if not isinstance(algebra_expression, CompValue):
                raise Exception('query algebra format not found')
            elif isinstance(algebra_expression, CompValue):
                if algebra_expression.name == p_name:
                    return algebra_expression
                else:
                    return find_p_name(algebra_expression['p'], p_name)

        algebra_expresison = self.algebra

        try:
             return find_p_name(algebra_expresison, p_name)

        except Exception:
            pass

    def get_graph(self):
        return self.graph

    def compare_for_isomorphism(self, template_graph, ignore_labels_of=["BNode"]):
        # BNode label is ignored IsomorphicTestableGraph method is used

        if "BNode" in ignore_labels_of:
            return self.graph == template_graph

        # elif "URIRef" in ignore_labels_of:
        #     return

    # def compare_for_equality(self, template_graph):
    #     # Uses an algorithm to compute unique hashes which takes bnodes into account.
    #     return isomorphic(self.graph, template_graph.graph)
    def compare_for_equality(self, template_graph):
        # Uses an algorithm to compute unique hashes which takes bnodes into account.
        return isomorphic(self.processed_graph, template_graph.processed_graph)


    # def compare_for_similarity(self, template_graph):
    #     # Checks if the two graphs are “similar”, by comparing sorted triples where all bnodes have been replaced
    #     # by a singular mock bnode (the _MOCK_BNODE).
    #     return similar(self.graph, template_graph.graph)
    def compare_for_similarity(self, template_graph):
        # Checks if the two graphs are “similar”, by comparing sorted triples where all bnodes have been replaced
        # by a singular mock bnode (the _MOCK_BNODE).
        return similar(self.processed_graph, template_graph.processed_graph)

    def write_image(self, display=False):
        """" to do """

        with open("%s.dot" % str(self.graph), 'w') as f:
            rdf2dot(Graph(self.graph), f)
            if display == "True":
                Source.main(f)
            # print(done)
def triples_list_to_graph(triples_list):
    q_graph = Graph()
    for tuple in triples_list:
        q_graph.add(tuple)
    return q_graph