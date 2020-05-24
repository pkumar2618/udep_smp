import json
import subprocess
import logging

from udep_lib.ug_logicalform import UGLogicalForm
logger = logging.getLogger(__name__)

class NLQCanonical(object):
    """
    Wrapper Class for Canonical form of the Natural Language Questions
    """

    def __init__(self, canonical_form):
        self.nlq_canonical = canonical_form
        logger.info(f'canonical-form: {canonical_form}')
    def formalize_into_udeplambda(self):
        # This is shortcut, note the we take help from UDepLambda to create lambda logical form
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

        # convert the bytecode into dictionary.
        self.udep_lambda = json.loads(res.decode('utf-8'))
        return UGLogicalForm(self.udep_lambda)
