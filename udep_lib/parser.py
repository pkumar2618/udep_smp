from udep_lib.nlquestion import NLQuestion
import logging
import json
logger = logging.getLogger(__name__)

class Parser(object):
    """
    Takes as input a list NLQuestion(one or more)
    and parse it.
    """
    def __init__(self, nlqs):
        """
        Take a list of questions (one or more)
        :param pp_nlqs: take as input a list of pre-processed NLQuestions
        """
        self.nlq_questions_list = [NLQuestion(nl_question) for nl_question in nlqs]
        self.nlq_tokens_list = []
        self.nlq_canonical_list = []
        self.ug_logical_form_list = []
        self.ug_gpgraphs_list = []
        self.ug_sparql_graphs_list = []
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
        # self.ug_logical_form_list = [nlq_canonical.direct_to_udeplambda()
                                     # for nlq_canonical in self.nlq_canonical_list]
        self.ug_gpgraphs_list = [nlq_canonical.direct_to_gpgraph()
                                     for nlq_canonical in self.nlq_canonical_list]

    def ungrounded_sparql_graph(self, kg='dbpedia'):
        """
        takes the the sparql query object obtained from the ug_logical_form.
        :return:
        """
        #for ug_logical_form in self.ug_logical_form_list:
        #    self.ug_sparql_graphs_list.append(ug_logical_form.translate_to_sparql(kg))
        for ug_graphs in self.ug_gpgraphs_list:
            self.ug_sparql_graphs_list.append(ug_graphs.gpgraph_to_sparql(kg))

    # this is where all the magic happens, linking using elasticsearch, as well as reranking using BERT
    def grounded_sparql_graph(self, linker=None, kg=None):
        for ug_sparql_graphs, nlquestion in zip(self.ug_sparql_graphs_list, self.nlq_questions_list):
            # there might me many gp_graphs obtained, we are using the first one for now, which is usually 
            # the case with UDepLambda, it generates only on logical form therefore on ungrounded gp-graph
            ug_sparql_graphs[0].ground_spo(question=nlquestion.question, linker=linker, kg=kg)
            graph_query = {'sparql_graph': ug_sparql_graphs[0].get_g_sparql_graph(), 'sparql_query': ug_sparql_graphs[0].get_g_sparql_query()}
            self.g_sparql_graph_list.append(graph_query)

    def query_executor(self, kg='dbpedia'):
        with open('execution_results.json', 'a') as f_handle:
            for query, nlq in zip(self.g_sparql_graph_list, self.nlq_questions_list):
                json_item = {}
                topk_sparql_graphs = query['sparql_graph']
                topk_sparql_queries = query['sparql_query']
                #topk = 1 #
                json_item['question'] = nlq.question.strip()
                json_item['topk_queries'] =  []
                for cand_query, q_string in zip(topk_sparql_graphs.g_query_topk, topk_sparql_queries):
                    temp_store = {'query_output': None, 'query_string': None}
                    try:
                        cand_query.run(kg)
                        result_list_dict  = cand_query.results["results"]["bindings"]
                        temp_store['query_output'] = result_list_dict
                        temp_store['query_string'] = q_string
                        json_item['topk_queries'].append(temp_store)
                        # for result_dict in result_list_dict:
                        # output_values = "\n".join([f"label: {key} \t value: { result_dict[key]}") for key in result_dict.keys()])
                        # f_handle.writeline(output_values)
                        # print("\n".join([f"label: {key} \t value: { result_dict[key]}") for key in result_dict.keys()]))
                    except TypeError as e:
                        temp_store['query_output'] =  f'{e}'
                        temp_store['query_string'] = q_string
                        json_item['topk_queries'].append(temp_store)

                json_item_string = json.dumps(json_item)
                f_handle.write(json_item_string + '\n')


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


    @classmethod
    def from_list(cls, question_list):
        """
        Should parse question in batch
        :param question_list:
        :return:
        """
        return cls(question_list)

if __name__ == "__main__":
    parser = Parser("When did Michael Jackson die?")
    parser.query_executor()
