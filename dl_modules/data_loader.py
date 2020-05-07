# Loading config_settings
from configparser import ConfigParser
config = ConfigParser()
config.read('bert_clf_config.ini')
# print(config['DEFAULT']['testing'])

from allennlp.data.vocabulary import Vocabulary
from typing import *
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data import Instance, token_indexers
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import json
import overrides


class QuestionSPOReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config['default_settings']['max_seq_len']) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, sentence_tokens: List[Token]) -> Instance:
        sentence_field = TextField(sentence_tokens, self.token_indexers)
        # spo_field = TextField(spo_tokens, self.token_indexers)
        fields = {"sentence_tokens": sentence_field} #, "spo_tokens": spo_field}
        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as f_read:
            json_dict = json.load(f_read)
            if config['default_settings']['testing']:
                json_dict = json_dict[:config['default_settings']['testing_load']]

            for question_spos in json_dict:
                question = question_spos['question']
                question_tokens = [Token(x) for x in self.tokenizer(question)]
                for spo_list in question_spos['spos_label']:
                    spo_label_joined = ' '.join(spo_list)
                    spo_tokens = [Token(x) for x in self.tokenizer(spo_label_joined)]
                    yield self.text_to_instance(question_tokens, spo_tokens)

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



if __main__ == __name__:
    """Testing"""