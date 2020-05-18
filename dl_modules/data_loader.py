from allennlp.data.vocabulary import Vocabulary
from typing import *
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ListField, Field, ArrayField
from allennlp.data import Instance, token_indexers
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import json
import numpy as np
import logging
from dl_utilities import ConfigJSON

config = ConfigJSON('configuration.json')
config.update(section_name="dataset_settings",
                       data={"testing": False, "testing_samples":4, "max_seq_len":100,
                             "max_vocab_size": 100000}
                       )

logger = logging.getLogger(__name__)

"""
The basic unit here is a block of sentence_positive-spos and sentence_negative-spos. This block is batched by the 
BucketIterator. Note that the block size is not finite, it may vary.
So is the number of tokens in the sentence_spo_tokens. 
"""
class QuestionSPOReader(DatasetReader):
    def __init__(
            self,
            my_tokenizer,
            my_token_indexers: Dict[str, TokenIndexer] = None,

    ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = my_tokenizer
        self.token_indexers = my_token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, sentence_spo_list: List[List[Token]],
                         sentence_spo_label_list: np.ndarray=None):
        fields: Dict[str, Field] = {}
        sentence_spo = ListField([TextField(sentence_spo_tokens, self.token_indexers) for sentence_spo_tokens in sentence_spo_list])
        fields['sentence_spo']= sentence_spo
        if sentence_spo_label_list is None:
            labels = np.zeros(len(sentence_spo_list)[0])
        else:
            labels = sentence_spo_label_list
        label_field = ArrayField(array=labels)
        fields["labels"] = label_field

        # if sentence_spo_label_list is None:
        #     labels = np.zeros(len(sentence_spo))
        #     fields['labels'] = ListField([LabelField(label) for label in sentence_spo_label_list])
        return Instance(fields)

    # @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        logger.info("Reading file at %s", file_path)
        with open(file_path, 'r') as f_read:
            json_dict = json.load(f_read)
            if config.config["dataset_settings"]["testing"]: # when testing only load a few examples
                json_dict = json_dict[:config.config["dataset_settings"]["testing_samples"]]

            # consider taking spo-triples from next 4 queries in the training set to form negative samples
            # of the current question's spo-triples
            len_dataset = len(json_dict)
            for i, question_spos in enumerate(json_dict):
                question = question_spos['question']
                question_tokens = self.tokenizer(question)
                question_tokens = [Token(x) for x in question_tokens]
                question_spo_list = list()
                question_spo_label_list = list()
                for spo_list in question_spos['spos_label']:
                    # correct spo: positive sample
                    spo_label_joined = ' '.join(spo_list)
                    spo_tokens = self.tokenizer(spo_label_joined)
                    spo_tokens = [Token(x) for x in spo_tokens]
                    question_spo_tokens = [Token("[CLS]")] + question_tokens + [Token("[SEP]")] + spo_tokens
                    question_spo_list.append(question_spo_tokens)
                    question_spo_label_list.append(1) # label is 1 for positive sample
                    # incorrect spo: the negative sample, not that this is not really
                    # a batch negative sample, where we pick up spo from the instances
                    # in the batch. Which kind of help in speed up, the least we can say.

                    for neg_sample in range(4):
                        if i < len_dataset - 4: # take next 4 queries
                            neg_question_spos = json_dict[i+neg_sample+1]
                        else: # take previous examples
                            neg_question_spos = json_dict[i -1- neg_sample]
                #         note that we will only take up spo-triples to form the negative examples for training.
                        for neg_spo_list in neg_question_spos['spos_label']:
                            neg_spo_label_joined = ' '.join(neg_spo_list)
                            neg_spo_tokens = self.tokenizer(neg_spo_label_joined)
                            neg_spo_tokens = [Token(x) for x in neg_spo_tokens]
                            # the question will stay from the positive sample,
                            # only the spo-triple will be taken up for negative example
                            question_neg_spo_tokens = [Token("[CLS]")] + question_tokens + [Token("[SEP]")] + neg_spo_tokens
                            question_spo_list.append(question_neg_spo_tokens)
                            question_spo_label_list.append("0") # zero label for negative sample

                    yield self.text_to_instance(question_spo_list, np.array(question_spo_label_list))


# The sentence need to be converted into a tensor(embedding space),
# but before that the tokens are mapped to an integer using the token_indexer.
# For BERT pre-trained we will use indexer, vocab and embedder all from the BERT pre-trained.
# Note that the BERT's wordpiece tokenizer need to be used

from allennlp.data.token_indexers import PretrainedBertIndexer
bert_token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-large-uncased",
    max_pieces=100,
    do_lowercase=True,
 )
# The vocabulary reuiqred for indexing is taken from the BERT pre-trained.
# vocab = Vocabulary()

# Tokenizer is obtained from Bert using PretrainedBertIOndexr
def bert_tokenizer(s: str):
    return bert_token_indexer.wordpiece_tokenizer(s)[:config.config["dataset_settings"]["max_seq_len"] - 2]
