from SPARQLWrapper import SPARQLWrapper, JSON
# from surf.query import a, select
# from surf.rdf import BNode, Graph, ConjunctiveGraph, Literal, Namespace
# from surf.rdf import RDF, URIRef
# import surf.namespace as namespace
import rdflib
from rdflib import URIRef, Graph, BNode
from rdflib.namespace import Namespace, NamespaceManager

class Query(object):
    """
    Wrapper for storing logical query
    """
    # Graph to keep all the namespace
    sparql_group = Graph()
    TYPES = ['SELECT']
    # namespace manager, to be attached to the sparql_group
    sparql_group.namespace_manager = NamespaceManager(Graph())

    # namespace.register(dbo="http://dbpedia.org/ontology/")
    # namespace.register(dbp="http://dbpedia.org/property/")
    # namespace.register(dbr="http://dbpedia.org/resource/")
    # namespace.register(dbt="http://dbpedia.org/resource/Template:")
    # namespace.register(dbc="http://dbpedia.org/resource/Category:")
    # namespace.register(dbpedia_commons="http://commons.dbpedia.org/resource/")
    # namespace.register(dbpedia_wikidata='http://wikidata.dbpedia.org/resource/')

    def __init__(self, *vars):
        """
        take the query_form obtained by formalizer and wrap it
        :param query_form:
        """
        self.sparql = None
        self._where = None
        self.results = None
        self._type = None
        self._data = Graph()
        self._modifier = None
        self._vars = []
        self._limit = None
        self._order_by = []
        self.nodes_type = {}


    def _validate_variable(self, var):
        if isinstance(var, str):
            if var.startswith('?'):
                return True
            else:
                raise ValueError(f"Not a Variable:{var} should start with ?")

    def has_variable(self, variable):
        #variable = f'?{variable}'
        if variable in self._vars:
            return True
        else:
            return False

    def _validate_triple_pattern(self, spo_tuple):
        if type(spo_tuple) in [list, tuple]:
            try:
                s, p, o = spo_tuple
            except:
                raise ValueError('t_pattern requires 3 terms')

            if isinstance(s, [BNode, URIRef]):
                pass
            else:
                raise ValueError('Subject is not a valid rdf term')

            if isinstance(p, [BNode, URIRef]):
                pass
            else:
                raise ValueError('Predicate is not a valid rdf term')

            if isinstance(o, [BNode, URIRef]):
                pass
            else:
                raise ValueError('Object is not a valid rdf term')

    def select(self, vars: object) -> object:
        """create a SELECT query"""
        self._type = "SELECT"
        self._vars = [var for var in vars if self._validate_variable(var)]


    def distinct(self):
        """
        add DISTINCT modifier to select clause
        :return:
        """
        self._modifier = 'DISTINCT'

    def add_triple(self, spo_tuple):
        """
        will take the tuple of s p o and add it to the _data to crete the graph-pattern
        :return:
        """
        self._data.add(spo_tuple)

    def empty_bgp(self):
        """
        empty the basic graph pattern
        :return:
        """
        self._data = Graph()


    def where(self, spo_triples):
        """
        Where clause for the select query
        :param tripple_pattern:
        :return:
        """
        # take spo_tuple and add to the graph-patern
        # if self._validate_triple_pattern(spo_tuple):
        self._where = "WHERE"
        for spo_tuple in spo_triples:
            self._data.add(spo_tuple)

    def optional_group(self, optional_tripple):
        """
        add optional triple to where clause
        :param optional_tripple:
        :return:
        """
        uelf.sparql.optional_group(optional_tripple)

    def filter(self, filter):
        """
        add filter to where clause
        :param filter:
        :return:
        """
        pass

    def _angular_braces(self, term):
        if isinstance(term, URIRef):
            if f'{term}'=='a':
                return f'{term}'
            else:
                return f'<{term}>'
        elif isinstance(term, BNode):
            return term
        else:
            raise ValueError("can't compose sparql query")

    def get_query_string(self, logical_form = 'sparql'):
        if logical_form == 'sparql':
            query_type = f"{self._type} {self._modifier} {' '.join(self._vars)} "

            bgp_string = ' . '.join([f"{self._angular_braces(s)} {self._angular_braces(p)} {self._angular_braces(o)}" for s, p, o in self._data])

            self.sparql= f"{query_type} {self._where} {{{bgp_string}}}"

        return str(self.sparql)

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

    def get_namespace(self):
        all_ns = '\n'.join([f"PREFIX: {p} \t URL: {u}" for p, u in Query.sparql_group.namespaces()])
        print(all_ns)

    def run(kg='dbpedia'):
        #sparql_endpoint = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql_endpoint = SPARQLWrapper("http://10.208.20.61:8890/sparql/")
        if kg=='freebase':
            sparql_endpoint = SPARQLWrapper("http://10.208.20.61:8895/sparql/")

        sparql_endpoint.setReturnFormat(JSON)
        try:
            sparql_endpoint.setQuery(self.sparql)
            self.results = sparql_endpoint.query().convert()

        except:
            #print("error quering endpoint")
            self.results = []

    @staticmethod
    def run(query_string, kg='dbpedia'):
        #sparql_endpoint = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql_endpoint = SPARQLWrapper("http://10.208.20.61:8890/sparql/")
        if kg=='freebase':
            sparql_endpoint = SPARQLWrapper("http://10.208.20.61:8895/sparql/")

        sparql_endpoint.setReturnFormat(JSON)
        try:
            sparql_endpoint.setQuery(query_string)
            results = sparql_endpoint.query().convert()
            result_list_dict  = results["results"]["bindings"]

        except:
            #print("error quering endpoint")
            results = []


if __name__ == "__main__":
    print("testing class Query")
    query = Query()

    # # test 1
    query.add_namespace('dbr', "http://dbpedia.org/resource/")
    # # test 1.1
    print(query.get_uri_for_prefix('dbr'))

    # test 2
    # compose sparql query
    # test 2.1
    query.select('?d')
    query.distinct()
    query.where((URIRef("http://dbpedia.org/resource/Diana,_Princess_of_Wales"), URIRef("http://dbpedia.org/ontology/deathDate"), BNode("?d")))
    # query.add_triple((URIRef("http://dbpedia.org/resource/_Princess_of_Wales"), URIRef("http://dbpedia.org/ontology/deathDate"), BNode("?d")))
    print(query.get_query_string())

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
    query.run(kg="dbpedia")
    result_list_dict  = query.results["results"]["bindings"]
    # print(result_list_dict["label"]["value"])
    for result_dict in result_list_dict:
        print("\n".join([f"label:{key}\t value:{result_dict[key]}" for key in result_dict.keys()]))
