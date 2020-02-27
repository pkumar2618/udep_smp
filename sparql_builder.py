from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef
from rdflib.namespace import Namespace, NamespaceManager
from rdflib import Literal, XSD
class Query(object):
    """
    Wrapper for storing logical query
    """

    def __init__(self, query_string):
        """
        take the query_form obtained by formalizer and wrap it
        :param query_form:
        """
        self.sparql = query_string
        self.namespace_manager = NamespaceManager(Graph())
        self.triples = Graph()
        self.results = []

    def select(self, name, distinct=False):
        self.query_type = "SELECT"
        self.param_name = name
        if distinct:
            self.distinct = "DISTINCT"

    def where(self):


    def get_query_string(self):
        return f'{self.query_type} \n'

    def PREFIX(self):
        """
        Default add dbpedia prefixes.
        :return:
        """
        dbo = Namespace("http://dbpedia.org/ontology/")
        dbp = Namespace("http://dbpedia.org/property/")
        dbr = Namespace("http://dbpedia.org/resource/")
        dbt = Namespace("http://dbpedia.org/resource/Template:")
        dbc = Namespace("http://dbpedia.org/resource/Category:")
        dbpedia_commons = Namespace("http://commons.dbpedia.org/resource/")
        dbpedia_wikidata = Namespace("http://wikidata.dbpedia.org/resource/")
        dbpedia_namespace_dict = {"dbo": dbo, "dbp":dbp, "dbr": dbr,"dbt": dbt,"dbc": dbc,
                             "dbpedia-commons": dbpedia_commons, "dbpedia-wikidata": dbpedia_wikidata}

        for db_prefix in dbpedia_namespace_dict.keys():
            self.namespace_manager.bind(db_prefix, dbpedia_namespace_dict[db_prefix], override=False)

    def run(self, kg='dbpedia'):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)
        try:
            sparql.setQuery(self.sparql)
            self.results = sparql.query().convert()

        except:
            self.results = None
