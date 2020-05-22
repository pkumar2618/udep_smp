from udep_lib.nlquestion import NLQuestion


class Parser(object):
    """
    Takes as input a list NLQuestion(one or more)
    and parse it.
    """
    # ug_sparql_graph_list: List[Any]

    def __init__(self, nlqs):
        """
        Take a list of questions (one or more)
        :param pp_nlqs: take as input a list of pre-processed NLQuestions
        """
        self.nlq_questions_list = [NLQuestion(nl_question) for nl_question in nlqs]
        self.nlq_tokens_list = []
        self.nlq_canonical_list = []
        self.ug_logical_form_list = []
        self.ug_sparql_graph_list = []
        self.g_sparql_graph_list = []
        self.results_list = []

    def tokenize(self, dependency_parsing, bypass=True):
        """
        tokenize the natural language question question
        :return:
        """
        self.nlq_tokens_list = [nl_question.tokenize(dependency_parsing) for nl_question in self.nlq_questions_list]

    def canonicalize(self, dependency_parsing=False, canonical_form=False):
        self.nlq_canonical_list = [nlq_tokens.canonicalize(dependency_parsing, canonical_form) for nlq_tokens in self.nlq_tokens_list]

    def ungrounded_logical_form(self):
        # self.ug_logical_form_list = [nlq_canonical.formalize_into_udeplambda()
        #                              for nlq_canonical in self.nlq_canonical_list]
        # for nlq_canon, sentence in zip(self.nlq_canonical_list, self.nlq_questions_list):
        #     self.ug_logical_form_list.append(nlq_canon.direct_to_udeplambda(sentence))
        self.ug_logical_form_list = [nlq_canonical.direct_to_udeplambda()
                                     for nlq_canonical in self.nlq_canonical_list]

    def ungrounded_sparql_graph(self, kg='dbpedia'):
        """
        takes the the sparql query object obtained from the ug_logical_form.
        :return:
        """
        for ug_logical_form in self.ug_logical_form_list:
            self.ug_sparql_graph_list.append(ug_logical_form.translate_to_sparql(kg))

    def grounded_sparql_graph(self, linker=None, kg=None):
        for ug_sparql_graph, nlquestion in zip(self.ug_sparql_graph_list, self.nlq_questions_list):
            ug_sparql_graph.ground_spo(question=nlquestion.question, linker=linker, kg=kg)
            self.g_sparql_graph_list.append(ug_sparql_graph.get_g_sparql_graph())

    def query_executor(self, kg='dbpedia'):
        # self.results_list = [query.run(kg) for query in self.query_list]
        # query_string = """SELECT DISTINCT xsd:date(?d) WHERE { <http://dbpedia.org/resource/Diana,_Princess_of_Wales>
        # <http://dbpedia.org/ontology/deathDate> ?d}
        # """
        # query_string="""
        # SELECT
        # DISTINCT ?date
        # WHERE
        # { <http://dbpedia.org/resource/Michael_Jackson> < http://dbpedia.org/ontology/deathDate> ?date}"""
        # query = Query(query_string)
        for query in self.g_sparql_graph_list:
            query.run(kg)
            result_list_dict  = query.results["results"]["bindings"]

            # print(result["label"]["value"])
            for result_dict in result_list_dict:
                print("\t".join(["label: { } \t value: { }".format(key, result_dict[key]) for key in result_dict.keys()]))


    @staticmethod
    def nlq_to_ug_form(nlq):
        return nlq


    @staticmethod
    def ug_to_g_form(ug_form):
        return ug_form


    @classmethod
    def from_file(cls, file_obj):
        """
        should parse question in batch
        :param file_obj:
        :return:
        """
        return cls(file_obj.readlines())


    # @classmethod
    # def from_list(cls, question_list):
    #     """
    #     Should parse question in batch
    #     :param question_list:
    #     :return:
    #     """

if __name__ == "__main__":
    parser = Parser("When did Michael Jackson die?")
    parser.query_executor()
