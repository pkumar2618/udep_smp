import json
import subprocess
import logging
import os

from udep_lib.ug_logicalform import UGLogicalForm
logger = logging.getLogger(__name__)

class NLQCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """

    def __init__(self, canonical_form):
        self.nlq_canonical = canonical_form
        self.udep_lambda = None
        self.ug_gp_graphs = None
        logger.info(f'canonical-form: {canonical_form}')

    def formalize_into_udeplambda(self):
        # This is shortcut, note that we take help from UDepLambda to create lambda logical form
        # from the natural question itself. So all this pipeline from natural language uptil tokenization is
        # now taken care off by the UDepLambda.
        # the lambda form is stored in the self.udep_lambda object variable.
        nlq = " ".join([word.text for word in self.nlq_canonical.words])
        with open("./udepl_nlq.txt", 'w') as f:
            f.write(f'{{"sentence":"{nlq}"}}')

        res = subprocess.check_output("./run_udep_lambda.sh")

        # convert the bytecode into dictionary.
        self.udep_lambda = json.loads(res.decode('utf-8'))
        return UGLogicalForm(self.udep_lambda)

    def direct_to_udeplambda(self, sentence=None):
        """
        here instead we will directly provide sentences to udeplambda
        :return: UGLogicalForm
        """
        if sentence is None:
            with open("./udepl_nlq.txt", 'w') as f:
                f.write(f'{{"sentence":"{self.nlq_canonical.strip()}"}}')
        else:
            with open("./udepl_nlq.txt", 'w') as f:
                f.write(f'{{"sentence":"{sentence}"}}')

        res = subprocess.check_output("./run_udep_lambda.sh")
        self.udep_lambda = json.loads(res.decode('utf-8'))
        # the output json_object will be stored in a file dep_parse.txt, that will be used by 
        # another script from UDepLambda to get to the ungrounded graph, implemented in ug_logical_form.translate_to_sparql.
        with open("./dep_parse.txt", 'w') as f:
            json.dump(self.udep_lambda, f)

        # the elements in entities will be provided key label entity instead of phrase.  
        semantic_parse = {} 
        with open("./dep_parse.txt", 'r') as f:
            semantic_parse = json.load(f)
            if 'entities' in semantic_parse.keys():
                temp_entities = [] # a list of dictionary items.
                semantic_parse_entity_list_dict = semantic_parse['entities'].copy()
                del semantic_parse['entities']
                for entity_dict in semantic_parse_entity_list_dict:
                   temp_entity = entity_dict.copy() 
                   del entity_dict['phrase']
                   entity_dict['entity'] = temp_entity['phrase']
                   temp_entities.append(entity_dict)

                semantic_parse['entities']= temp_entities
                self.udep_lambda = semantic_parse

        with open("./dep_parse.txt", 'w') as f:
            json.dump(semantic_parse, f)

        # convert the bytecode into dictionary.
        return UGLogicalForm(self.udep_lambda)

    def lambda_to_sqg(self):
        res = subprocess.check_output("./run_lambda_ug_graph.sh")
        #self.ug_graph = json.loads(res.decode('utf-8'))
        #the graph from gp is written in a file ug_graph.txt
        # it is for the next_module in the pipeline ug_logicalform to read the file and convert the gp-graph into a
        # sparql graph.
    
    def direct_to_gpgraph(self):
        self.direct_to_udeplambda()
        self.lambda_to_sqg()
        #this function will return the gp-graph as dictionary of keys: {Semantic Parse, Words, Edgesi, Types, Properties, EventTypes, EventEventModifiers} 
        #curr_path = os.getcwd()
        with open('./ug_graph.txt') as f_read:
           lines = f_read.readlines()
           line_iter = iter(lines)
           graphs_count = 0
           line = next(line_iter) 
           line_split = line.split(":", 1)
           json_gp_graphs = []
           if line_split[0] == "UG Graphs":
               graphs_count = int(line_split[1].strip())
           for g in range(graphs_count):
               json_graph = {}
               while True:
                   line = next(line_iter)
                   line_split = line.split(":", 1)
                   if line_split[0] == "Semantic Parse":
                       json_graph["Semantic Parse"]=line_split[1]
                   elif line_split[0] == "Words":
                       line = next(line_iter)
                       json_graph["nodes"] = []
                       while(line.split(":", 1)[0] != "Edges"):
                           json_graph['nodes'].append(line.replace("LexicalItem", ""))
                           line = next(line_iter)
                       json_graph["Edges"] = []
                       line = next(line_iter)
                       while(line.split(":", 1)[0] != "Types"):
                           json_graph['Edges'].append(line.split("\t", 1))
                           line = next(line_iter)
                       json_graph["Types"] = []
                       line = next(line_iter)
                       while(line.split(":", 1)[0] != "Properties"):
                           json_graph['Types'].append(line.split("\t", 1))
                           line = next(line_iter)
                       json_graph["Properties"] = []
                       line = next(line_iter)
                       while(line.split(":", 1)[0] != "EventTypes"):
                           json_graph["Properties"].append(line.split("\t", 1))
                           line = next(line_iter)
                       json_graph["EventTypes"] = []
                       line = next(line_iter)
                       while(line.split(":", 1)[0] != "EventEventModifiers"):
                           json_graph["EventTypes"].append(line.split("\t", 1))
                           line = next(line_iter)
                       json_graph["EventEventModifiers"] = []
                       try:
                           line = next(line_iter, None)
                           while(line.split(":", 1)[0] != "Semantic Parse"):
                               json_graph["EventEventModifiers"].append(line.split("\t", 1))
                               line = next(line_iter)
                           json_gp_graphs.append(json_graph)
                           break
                       except AttributeError as e:
                           json_gp_graphs.append(json_graph)
                           break

        self.ug_gp_graphs = json_gp_graphs
        return UGLogicalForm(None, self.ug_gp_graphs)

if __name__=='__main__':
    parser = NLQCanonical('Who was the doctoral supervisor of Albert Einstein?')
    parser.direct_to_udeplambda()
    parser.lambda_to_sqg()

