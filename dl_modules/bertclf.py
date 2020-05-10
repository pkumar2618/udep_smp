import torch
from pathlib import Path
from configparser import ConfigParser


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    testing=True,
    testing_sample=100,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
)

# ------
# from allennlp.data.vocabulary import Vocabulary
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

import json
import overrides
import logging

logger = logging.getLogger(__name__)

class QuestionSPOReader(DatasetReader):
    def __init__(
            self,
            my_tokenizer: Tokenizer = None,
            my_token_indexers: Dict[str, TokenIndexer] = None,
            passage_length_limit: int = None,
            question_length_limit: int = None,
            skip_invalid_examples: bool = False,
    ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = my_tokenizer
        self.token_indexers = my_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples

    # @overrides
    def text_to_instance(self, sentence_tokens: List[Token], spo_tokens: List[Token]):
        sentence_field = TextField(sentence_tokens, self.token_indexers)
        spo_field = TextField(spo_tokens, self.token_indexers)
        fields = {"sentence_tokens": sentence_field, "spo_tokens": spo_field}
        return Instance(fields)

    # @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        logger.info("Reading file at %s", file_path)
        with open(file_path, 'r') as f_read:
            json_dict = json.load(f_read)
            if config.testing: # when testing only load a few examples
                json_dict = json_dict[:config.testing_sample]

            for question_spos in json_dict:
                question = question_spos['question']
                # question_tokens = self.tokenizer.tokenize(question)
                question_tokens = [Token(word) for word in question]
                try:
                    for spo_list in question_spos['spos_label']:
                        spo_label_joined = ' '.join(spo_list)
                        # spo_tokens = self.tokenizer.tokenize(spo_label_joined)
                        spo_tokens = [Token(word) for word in spo_label_joined]
                        yield self.text_to_instance(question_tokens, spo_tokens)
                except:
                    pass


# The sentence need to be converted into tensor, the indexer is used to do that.
# For BERT pre-trained we will use indexer from BERT, which will get us the BERT Vocabulary as well as
# the BERT's wordpiece tokenizer.

from allennlp.data.token_indexers import PretrainedBertIndexer
bert_token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-large-uncased",
    max_pieces=100,
    do_lowercase=True,
 )


# Tokenizer is obtained from Bert using PretrainedBertIOndexr
def bert_tokenizer(s: str):
    return bert_token_indexer.wordpiece_tokenizer(s)[:eval(config.max_seq_len) - 2]
#------


# creating the dataset reader
# from dl_modules.data_loader import QuestionSPOReader, tokenizer, token_indexer
# from dl_modules.data_loader import *
reader = QuestionSPOReader(my_tokenizer=bert_tokenizer,
                           my_token_indexers={"sentence_tokens": bert_token_indexer, "spo_tokens": bert_token_indexer})

# config['dataset_settings'] = {'test_file': Path("../dataset_qald") / "qald_input.json"}
# loading test_set data
train_ds = reader.read(Path("../dataset_qald") / "qald_input.json")
val_ds = None

print(len(train_ds))

## running the trainier
