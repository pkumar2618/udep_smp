import json
import subprocess
import logging

# from udep_lib.ug_logicalform import UGLogicalForm
logger = logging.getLogger(__name__)

class NLQCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """

    def __init__(self, canonical_form):
        self.nlq_canonical = canonical_form
        self.ug_graph = None
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
        # return UGLogicalForm(self.udep_lambda)
        return self.udep_lambda

    def lambda_to_sqg(self):
        res = subprocess.check_output("./run_lambda_ug_graph.sh")
        self.ug_graph = json.loads(res.decode('utf-8'))


if __name__=='__main__':
    parser = NLQCanonical('Who was the doctoral supervisor of Albert Einstein?')
    parser.direct_to_udeplambda()
    parser.lambda_to_sqg()
