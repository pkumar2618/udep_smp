from SPARQLWrapper import SPARQLWrapper, JSON
# from surf.query import a, select
# from surf.rdf import BNode, Graph, ConjunctiveGraph, Literal, Namespace
# from surf.rdf import RDF, URIRef
# import surf.namespace as namespace
import rdflib
from rdflib import URIRef, Graph
from rdflib.namespace import Namespace, NamespaceManager

class Query(object):
    """
    Wrapper for storing logical query
    """
    # Graph to keep all the namespace
    sparql_group = Graph()

    # namespace manager, to be attached to the sparql_group
    sparql_group.namespace_manager = NamespaceManager(Graph())

    # namespace.register(dbo="http://dbpedia.org/ontology/")
    # namespace.register(dbp="http://dbpedia.org/property/")
    # namespace.register(dbr="http://dbpedia.org/resource/")
    # namespace.register(dbt="http://dbpedia.org/resource/Template:")
    # namespace.register(dbc="http://dbpedia.org/resource/Category:")
    # namespace.register(dbpedia_commons="http://commons.dbpedia.org/resource/")
    # namespace.register(dbpedia_wikidata='http://wikidata.dbpedia.org/resource/')

    def __init__(self):
        """
        take the query_form obtained by formalizer and wrap it
        :param query_form:
        """
        self.sparql = None
        self.results = None
        # self.namespace_manager = NamespaceManager(Graph())
        # self.triples = Graph()


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
        pass

    def get_query_string(self):
        return unicode(self.sparql)

    def add_namespace(self, prefix, url_string):
        """
        #todo not working
        :return:
        """
        # create the Namespace using url_string
        namespace = Namespace(url_string)
        Query.sparql_group.namespace_manager.bind(prefix, namespace, override=False)

    def get_uri_for_prefix(self, prefix):
        """
        given the prefix, return its uri if it exist in the namespace
        :param prefix:
        :return:
        """
        for p, u in Query.sparql_group.namespaces():
            if p == prefix:
                return u

    def run(self, kg='dbpedia'):
        sparql_endpoint = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql_endpoint.setReturnFormat(JSON)
        try:
            sparql_endpoint.setQuery(self.sparql)
            self.results = sparql_endpoint.query().convert()

        except:
            print("error quering endpoint")
            self.results = None


if __name__ == "__main__":
    print("testing class Query")
    query = Query()
    # test 1
    query.add_namespace('dbr', "http://dbpedia.org/resource/")
    # test 2
    print(query.get_uri_for_prefix('dbr'))
    # print([(p, u) for p, u in Query.sparql_group.namespaces()])
    # test 3

    # # query example
    # sparql = Query()
    # sparql.select("?date")
    # sparql.distinct()
    # sparql.where((URIRef("http://dbpedia.org/resource/Michael_Jackson"), URIRef("http://dbpedia.org/ontology/deathDate"), "?date"))
    # print(unicode(sparql.sparql))

    # another query example
    # query = Query()
    # print(namespace.get_namespace_url('xsd'))
    # query.select('?d')
    # query.distinct()
    # query.where((URIRef("http://dbpedia.org/resource/Diana,_Princess_of_Wales"), URIRef("http://dbpedia.org/ontology/deathDate"), "?d"))
    # print(unicode(query.sparql))
    #
    # query.run(kg="dbpedia")
    # result_list_dict  = query.results["results"]["bindings"]
    # print(result["label"]["value"])
    # for result_dict in result_list_dict:
    #     print("\n".join(["label: { }\t value: { }".format(key, result_dict[key]) for key in result_dict.keys()]))
    #
