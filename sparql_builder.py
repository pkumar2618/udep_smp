from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef
from rdflib.namespace import Namespace, NamespaceManager
from surf.query import a, select
from surf.rdf import BNode, Graph, ConjunctiveGraph, Literal, Namespace
from surf.rdf import RDF, URIRef
import surf.namespace as namespace

class Query(object):
    """
    Wrapper for storing logical query
    """
    namespace_manager = namespace
    namespace_manager.register(dbo="http://dbpedia.org/ontology/")
    namespace_manager.register(dbp="http://dbpedia.org/property/")
    namespace_manager.register(dbr="http://dbpedia.org/resource/")
    namespace_manager.register(dbt="http://dbpedia.org/resource/Template:")
    namespace_manager.register(dbc="http://dbpedia.org/resource/Category:")
    # namespace.register(dbpedia_commons="http://commons.dbpedia.org/resource/")
    # namespace.register(dbpedia_wikidata='http://wikidata.dbpedia.org/resource/')

    def __init__(self):
        """
        take the query_form obtained by formalizer and wrap it
        :param query_form:
        """
        self.sparql = None
        # self.namespace_manager = NamespaceManager(Graph())
        # self.triples = Graph()
        # self.results = []

    def select(self, var):
        """create a SELECT query"""
        self.sparql = select(var)

    def distinct(self):
        """
        add DISTINCT modifier to select clause
        :return:
        """
        self.sparql.distinct()

    def where(self, tripple_pattern):
        """
        Where clause for the select query
        :param tripple_pattern:
        :return:
        """
        self.sparql.where(tripple_pattern)

    def optional_group(self, optional_tripple):
        """
        add optional triple to where clause
        :param optional_tripple:
        :return:
        """
        self.sparql.optional_group(optional_tripple)

    def filter(self, filter):
        """
        add filter to where clause
        :param filter:
        :return:
        """

    def get_query_string(self):
        return unicode(self.sparql)

    def add_namespace(self, namespace_dict=[]):
        """
        Default add dbpedia prefixes.
        :return:
        """
        for key in namespace_dict.keys():
            Query.namespace_manager.register(key=namespace_dict[key])

    def run(self, kg='dbpedia'):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)
        try:
            sparql.setQuery(self.sparql)
            self.results = sparql.query().convert()

        except:
            self.results = None


if __name__ == "__main__":
    print("testing class Query")
    sparql = Query()
    sparql.select("?date")
    sparql.distinct()
    sparql.where((URIRef("http://dbpedia.org/resource/Michael_Jackson"), URIRef("http://dbpedia.org/ontology/deathDate"), "?date"))
    print(unicode(sparql.sparql))
    # sparql.run(kg="dbpedia")
    # result_list_dict  = sparql.results["results"]["bindings"]
    # print(result["label"]["value"])
    # for result_dict in result_list_dict:
    #     print("\n".join(["label: { }\t value: { }".format(key, result_dict[key]) for key in result_dict.keys()]))

    # Test Namespace hold across various call to the QUERY Object
    sparql.add_namespace({'dbpedia_wikidata':'http://wikidata.dbpedia.org/resource/'})


    print Query.namespace_manager.get_namespace_url('dbr')
    print namespace.get_namespace_url('DBP')
    print namespace.get_namespace_url('DBR')
    print namespace.get_namespace_url('DBC')
    # print Query.namespace_manger.get_namespace_url('dbpedia_wikidata')