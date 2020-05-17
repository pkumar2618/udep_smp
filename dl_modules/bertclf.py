from pathlib import Path
import pickle
import numpy as np
import json
import overrides
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from typing import Iterator, Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING
from allennlp.data import Instance
from allennlp.data.fields import TextField, Field, ListField, LabelField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.trainer import Trainer
from dl_modules.data_loader import QuestionSPOReader, bert_token_indexer, bert_tokenizer
from dl_modules.dl_utilities import ConfigJSON

# we are goint to use a single configuration file for the entire deep learning module.
config = ConfigJSON('configuration.json')
config.update(section_name = "training_settings",
                       data={"seed": 1, "batch_size":2, "learning_rate":3e-4,
                             "epochs": 3, "USE_GPU": torch.cuda.is_available()}
                       )

logging.basicConfig(filename='bertclf.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)


torch.manual_seed(config.config["training_settings"]["seed"])

# The sentence need to be converted into a tensor(embedding space),
# but before that the tokens are mapped to an integer using the token_indexer.
# For BERT pre-trained we will use indexer, vocab and embedder all from the BERT pre-trained.
# Note that the BERT's wordpiece tokenizer need to be used


# creating the dataset reader
reader = QuestionSPOReader(my_tokenizer=bert_tokenizer,
                       my_token_indexers={"sentence_spo": bert_token_indexer})

train_ds = reader.read("../dataset_qald/qald_train.json")
val_ds = reader.read("../dataset_qald/qald_val.json")

# ### the iterator is used to batch the data and prepare it for input to the model
# from allennlp.data.iterators import BucketIterator
# iterator = BucketIterator(batch_size=config.batch_size,
#                           sorting_keys=[('sentence_spo', 'num_fields'), ('sentence_spo', 'list_num_tokens')])
# # # the iterator has to be informed about how the indexing has to be done, indexing require the vocabulary
# # # of all the tokens (may be trimmed if the vocab size is huge).
# iterator.index_with(vocab)
# # # example batch
# # batch = next(iter(iterator(train_ds)))
# # print(batch)
# # print(batch['sentence_spo']['sentence_spo'].shape)


## running the trainier
## creating the Model now

#
# class CrossEncoderModel(Model):
#     def __init__(self, word_embeddings: TextFieldEmbedder,
#                  cls_token_rep: Seq2VecEncoder,
#                  out_sz: int = 1):
#         super().__init__(vocab)
#         self.word_embeddings = word_embeddings
#         self.cls_token_rep = cls_token_rep
#         self.cross_emb_score = nn.Linear(self.cls_token_rep.get_output_dim(), out_sz)
#         self.loss = nn.BCEWithLogitsLoss()
#
#     def forward(self, sentence_spo: Dict[str, torch.Tensor],
#                 labels: List[torch.Tensor] = None) -> torch.Tensor:
#         all_sample_logits = list()
#         # all_sample_labels = list()
#         for sentence_spo_block in sentence_spo['sentence_spo']:
#             for sentence_spo_tokens in sentence_spo_block:
#                 # mask = get_text_field_mask(sentence_spo_tokens)
#                 bert_embedding = self.word_embeddings({'sentence_spo':sentence_spo_tokens.unsqueeze(dim=0)})
#                 cls_token_rep = self.cls_token_rep(bert_embedding.squeeze(0))
#                 cross_emb_score_logit = self.cross_emb_score(cls_token_rep)
#                 all_sample_logits.append(cross_emb_score_logit)
#
#         output = {'sentence_spo_logits': all_sample_logits,
#                   'sentence_spo_labels': labels}
#         if labels is not None:
#             loss = self.loss(torch.cat(all_sample_logits), labels.view(-1))
#             output["loss"] = loss
#
#         return output
#
# # The embedder for TextField is not changed, it's basic and Text Field embedder
# from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# # We need to use BERT pre-trained model to create the embedding for the input tokens
# from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
#
# bert_embedder = PretrainedBertEmbedder(
#         pretrained_model="bert-large-uncased",
#         top_layer_only=True, # conserve memory
#         # pretrained_model="bert-base-uncased",
#         # top_layer_only=True, # conserve memory
# )
# # The embedder gets us an embedding, here a werd embedding.
# word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"sentence_spo": bert_embedder},
#                                                             # we'll be ignoring masks so we'll need to set this to True
#                                                            allow_unmatched_keys = True)
#
# BERT_DIM = word_embeddings.get_output_dim()
# # After we have the embedding vectors for all the tokens in the sentence, we need to reduce
# # this sequence of vectors into a single vector (Humeau 2019 poly-encoder).
# # we are going to use the representation of [CLS] token
# class ClsTokenRep(Seq2VecEncoder):
#     def forward(self, emb_seq: torch.tensor) -> torch.tensor:
#         # extract first token tensor
#         return emb_seq[0, :]
#
#     def get_output_dim(self) -> int:
#         return BERT_DIM
#
# cls_token_encoder = ClsTokenRep(vocab)
#
# #if the model exist already don't train again load the trained model
# # instantiate the model with the cls_token_encoder which just takes out the embedding for the [CLS]
# # token
#
# model = CrossEncoderModel(word_embeddings, cls_token_encoder)
# if USE_GPU:
#     model.cuda()
# else:
#     model
#
# # Sanity Check
# # # example batch
# # batch = next(iter(iterator(train_ds)))
# # sentence_spo = batch["sentence_spo"]
# # # mask = get_text_field_mask(sentence_spo)
# # embeddings = model.word_embeddings(sentence_spo)
# # cls_rep = model.cls_token_rep(embeddings)
# # class_logits = model.cross_emb_score(cls_rep)
# # loss = model(**batch)["loss"]
# # print(loss)
# # Do we need training? with representation of [cls] tokens.
# # Batch negative samples.
#
# # training
# optimizer = optim.Adam(model.parameters(), lr=config.lr)
# trainer = Trainer(
#     model=model,
#     optimizer=optimizer,
#     iterator=iterator,
#     train_dataset=train_ds,
#     validation_dataset=val_ds,
#     cuda_device=0 if USE_GPU else -1,
#     num_epochs=config.epochs,
# )
#
# metrics = trainer.train()
# print(metrics)
# # Save the model after training is complete
# # Here's how to save the model.
# with open('model_bup.th', 'wb') as f:
#     torch.save(model.state_dict(), f)
# # save the vocabulary
# vocab.save_to_files("./vocabulary")
