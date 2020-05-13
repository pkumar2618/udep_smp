import torch
from pathlib import Path
from configparser import ConfigParser
import pickle

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
    batch_size=4,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
)
from allennlp.common.checks import ConfigurationError
USE_GPU = torch.cuda.is_available()
DATA_ROOT = Path("../data") / "jigsaw"
torch.manual_seed(config.seed)

# ------
from allennlp.data.vocabulary import Vocabulary
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
            my_tokenizer,
            my_token_indexers: Dict[str, TokenIndexer] = None,
            # passage_length_limit: int = None,
            # question_length_limit: int = None,
            # skip_invalid_examples: bool = False,
    ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = my_tokenizer
        self.token_indexers = my_token_indexers or {"tokens": SingleIdTokenIndexer()}
        # self.passage_length_limit = passage_length_limit
        # self.question_length_limit = question_length_limit
        # self.skip_invalid_examples = skip_invalid_examples

    # @overrides
    def text_to_instance(self, sentence_spo_tokens: List[Token]):
        sentence_spo_field = TextField(sentence_spo_tokens, self.token_indexers)
        fields = {"sentence_spo_tokens": sentence_spo_field}
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
                question_tokens = self.tokenizer(question)
                # question_tokens = [Token(word) for word in question]
                for spo_list in question_spos['spos_label']:
                    spo_label_joined = ' '.join(spo_list)
                    spo_tokens = self.tokenizer(spo_label_joined)
                    # spo_tokens = [Token(word) for word in spo_label_joined]
                    question_spo_tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + spo_tokens
                    yield self.text_to_instance([Token(x) for x in question_spo_tokens])



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
vocab = Vocabulary()

# Tokenizer is obtained from Bert using PretrainedBertIOndexr
def bert_tokenizer(s: str):
    return bert_token_indexer.wordpiece_tokenizer(s)[:config.max_seq_len - 2]
#------


# creating the dataset reader
# from dl_modules.data_loader import QuestionSPOReader, tokenizer, token_indexer
# from dl_modules.data_loader import *
# load the text to instance if already done
# try:
#     with open('train_ds.pkl', 'rb') as f_read:
#         train_ds = pickle.load(f_read)
#         print(len(train_ds))
# except FileNotFoundError as err_file:
reader = QuestionSPOReader(my_tokenizer=bert_tokenizer,
                       my_token_indexers={"sentence_spo_tokens": bert_token_indexer})
# loading test_set data
train_ds = reader.read("../dataset_qald/qald_input.json")
val_ds = None
# with open('train_ds.pkl', 'wb') as f_write:
#     pickle.dump(train_ds, f_write)

### the iterator is used to batch the data and prepare it for input to the model
from allennlp.data.iterators import BucketIterator
iterator = BucketIterator(batch_size=config.batch_size,
                          sorting_keys=[('sentence_spo_tokens', 'num_tokens')])
# # the iterator has to be informed about how the indexing has to be done, indexing require the vocabulary
# # of all the tokens (may be trimmed if the vocab size is huge).
iterator.index_with(vocab)
# # example batch
# batch = next(iter(iterator(train_ds)))
# print(batch)
# print(batch['sentence_spo_tokens']['sentence_spo_tokens'].shape)
## running the trainier

## creating the Model now
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder


class CrossEncoderModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 cls_token_rep: Seq2VecEncoder,
                 out_sz: int = 1):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.cls_token_rep = cls_token_rep
        self.cross_emb_score = nn.Linear(self.cls_token_rep.get_output_dim(), out_sz)
        # self.loss = nn.BCEWithLogitsLoss()

    def forward(self, sentence_spo_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = get_text_field_mask(sentence_spo_tokens)
        bert_embedding = self.word_embeddings(sentence_spo_tokens)
        cls_token_rep = self.cls_token_rep(bert_embedding, mask)
        cross_emb_score_logit = self.cross_emb_score(cls_token_rep)
        output = {"cross_emb_score": cross_emb_score_logit}
        output["loss"] = cross_emb_score_logit  # todo how to add logits for negative samples
        # output["loss"] = self.loss(cross_emb_score_logit) # todo how to add logits for negative samples
        return output

# The embedder for TextField is not changed, it's basic and Text Field embedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# We need to use BERT pre-trained model to create the embedding for the input tokens
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-large-uncased",
        top_layer_only=True, # conserve memory
)
# The embedder gets us an embedding, here a werd embedding.
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"sentence_spo_tokens": bert_embedder},
                                                            # we'll be ignoring masks so we'll need to set this to True
                                                           allow_unmatched_keys = True)

BERT_DIM = word_embeddings.get_output_dim()
# After we have the embedding vectors for all the tokens in the sentence, we need to reduce
# this sequence of vectors into a single vector (Humeau 2019 poly-encoder).
# we are going to use the representation of [CLS] token
class ClsTokenRep(Seq2VecEncoder):
    def forward(self, emb_seq: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        return emb_seq[:, 0]

    def get_output_dim(self) -> int:
        return BERT_DIM


cls_token_emb = ClsTokenRep(vocab)
model = CrossEncoderModel(word_embeddings, cls_token_emb)

if USE_GPU: model.cuda()
else: model

# Sanity Check
# # example batch
batch = next(iter(iterator(train_ds)))
sentence_spo_tokens = batch["sentence_spo_tokens"]
mask = get_text_field_mask(sentence_spo_tokens)
embeddings = model.word_embeddings(sentence_spo_tokens)
cls_rep = model.cls_token_rep(embeddings, mask)
class_logits = model.cross_emb_score(cls_rep)
loss = model(**batch)["loss"]
print(loss)
# Do we need training? with representation of [cls] tokens.
# Batch negative samples.



# training
optimizer = optim.Adam(model.parameters(), lr=config.lr)
from allennlp.training.trainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    cuda_device=0 if USE_GPU else -1,
    num_epochs=config.epochs,
)
metrics = trainer.train()
print(metrics)