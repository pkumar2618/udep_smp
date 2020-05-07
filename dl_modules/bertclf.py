import torch
from pathlib import Path
from configparser import ConfigParser

config = ConfigParser()
config['default_settings'] = {
    'testing': True,
    'seed': 1,
    'batch_size': 64,
    'lr': 3e-4,
    'epochs': 2,
    'hidden_sz': 64,
    'max_seq_len': 100,  # necessary to limit memory usage
    'max_vocab_size': 100000,
}
config['dataset_settings'] = {'test_file': Path("../dataset_qald") / "qald_input.json"}
config['gpu_settings'] = {'available': torch.cuda.is_available()}
config['torch_settings'] = {'seed': config['default_settings']['seed']}

with open('bert_clf_config.ini', 'w') as f_write:
    config.write(f_write)
# ------
from allennlp.data.vocabulary import Vocabulary
from typing import *
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data import Instance, token_indexers
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
import json
import overrides
import logging

logger = logging.getLogger(__name__)

class QuestionSPOReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            passage_length_limit: int = None,
            question_length_limit: int = None,
            skip_invalid_examples: bool = False,
    ) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples


    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        logger.info("Reading file at %s", file_path)
        with open(file_path, 'r') as f_read:
            json_dict = json.load(f_read)
            if config['default_settings']['testing']:
                json_dict = json_dict[:config['default_settings']['testing_load']]

            for question_spos in json_dict:
                question = question_spos['question']
                question_tokens = self._tokenizer.tokenize(question)

                for spo_list in question_spos['spos_label']:
                    spo_label_joined = ' '.join(spo_list)
                    spo_tokens = self._tokenizer.tokenize(spo_label_joined)
                    yield self.text_to_instance(question_tokens, spo_tokens)


    @overrides
    def text_to_instance(
            self,
            sentence_tokens: List[Token]=None,
            spo_tokens: List[Token]=None
    ) -> Optional[Instance]:
        sentence_field = TextField(sentence_tokens, self.token_indexers)
        spo_field = TextField(spo_tokens, self.token_indexers)
        fields = {"sentence_tokens": sentence_field, "spo_tokens": spo_field}
        return Instance(fields)

# The sentence need to be converted into tensor, the indexer is used to do that.
# For BERT pre-trained we will use indexer from BERT, which will get us the BERT Vocabulary as well as
# the BERT's wordpiece tokenizer.

from allennlp.data.token_indexers import PretrainedBertIndexer
token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-large-uncased",
    max_pieces=config['default_settings']['max_seq_len'],
    do_lowercase=True,
 )
# Tokenizer is obtained from Bert using PretrainedBertIOndexr
def tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)[:config['default_settings']['max_seq_len'] - 2]
#------

# creating the dataset reader
# from dl_modules.data_loader import QuestionSPOReader, tokenizer, token_indexer
# from dl_modules.data_loader import *
reader = QuestionSPOReader(tokenizer=tokenizer, token_indexers={"sentence_tokens": token_indexer})#, "spo_tokens": token_indexer})

# loading test_set data
train_ds = reader.read(config['default_settings']['test_file'])
val_ds = None

len(train_ds)

## running the trainier
