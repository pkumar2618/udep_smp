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
    testing_sample=10,
    seed=1,
    batch_size=5,
    lr=3e-4,
    epochs=5,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
)

logging.basicConfig(filename='bertclf.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)

USE_GPU = torch.cuda.is_available()
torch.manual_seed(config.seed)


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
            if config.testing: # when testing only load a few examples
                json_dict = json_dict[:config.testing_sample]

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
vocab = Vocabulary()

# Tokenizer is obtained from Bert using PretrainedBertIOndexr
def bert_tokenizer(s: str):
    return bert_token_indexer.wordpiece_tokenizer(s)[:config.max_seq_len - 2]


# creating the dataset reader
reader = QuestionSPOReader(my_tokenizer=bert_tokenizer,
                       my_token_indexers={"sentence_spo": bert_token_indexer})
train_ds = reader.read("../dataset_qald/qald_input.json")
val_ds = None

### the iterator is used to batch the data and prepare it for input to the model
from allennlp.data.iterators import BucketIterator
iterator = BucketIterator(batch_size=config.batch_size,
                          sorting_keys=[('sentence_spo', 'num_fields'), ('sentence_spo', 'list_num_tokens')])
# # the iterator has to be informed about how the indexing has to be done, indexing require the vocabulary
# # of all the tokens (may be trimmed if the vocab size is huge).
iterator.index_with(vocab)
# # example batch
# batch = next(iter(iterator(train_ds)))
# print(batch)
# print(batch['sentence_spo']['sentence_spo'].shape)


## running the trainier
## creating the Model now


class CrossEncoderModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 cls_token_rep: Seq2VecEncoder,
                 out_sz: int = 1):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.cls_token_rep = cls_token_rep
        self.cross_emb_score = nn.Linear(self.cls_token_rep.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, sentence_spo: Dict[str, torch.Tensor],
                labels: List[torch.Tensor] = None) -> torch.Tensor:
        all_sample_logits = list()
        # all_sample_labels = list()
        for sentence_spo_block in sentence_spo['sentence_spo']:
            for sentence_spo_tokens in sentence_spo_block:
                # mask = get_text_field_mask(sentence_spo_tokens)
                bert_embedding = self.word_embeddings({'sentence_spo':sentence_spo_tokens.unsqueeze(dim=0)})
                cls_token_rep = self.cls_token_rep(bert_embedding.squeeze(0))
                cross_emb_score_logit = self.cross_emb_score(cls_token_rep)
                all_sample_logits.append(cross_emb_score_logit)

        output = {'sentence_spo_logits': all_sample_logits,
                  'sentence_spo_labels': labels}
        if labels is not None:
            loss = self.loss(torch.cat(all_sample_logits), labels.view(-1))
            output["loss"] = loss

        return output

# The embedder for TextField is not changed, it's basic and Text Field embedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# We need to use BERT pre-trained model to create the embedding for the input tokens
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-large-uncased",
        top_layer_only=True, # conserve memory
        # pretrained_model="bert-base-uncased",
        # top_layer_only=True, # conserve memory
)
# The embedder gets us an embedding, here a werd embedding.
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"sentence_spo": bert_embedder},
                                                            # we'll be ignoring masks so we'll need to set this to True
                                                           allow_unmatched_keys = True)

BERT_DIM = word_embeddings.get_output_dim()
# After we have the embedding vectors for all the tokens in the sentence, we need to reduce
# this sequence of vectors into a single vector (Humeau 2019 poly-encoder).
# we are going to use the representation of [CLS] token
class ClsTokenRep(Seq2VecEncoder):
    def forward(self, emb_seq: torch.tensor) -> torch.tensor:
        # extract first token tensor
        return emb_seq[0, :]

    def get_output_dim(self) -> int:
        return BERT_DIM


# instantiate the model with the cls_token_encoder which just takes out the embedding for the [CLS]
# token
cls_token_encoder = ClsTokenRep(vocab)
model = CrossEncoderModel(word_embeddings, cls_token_encoder)

if USE_GPU: model.cuda()
else: model

# Sanity Check
# # example batch
# batch = next(iter(iterator(train_ds)))
# sentence_spo = batch["sentence_spo"]
# # mask = get_text_field_mask(sentence_spo)
# embeddings = model.word_embeddings(sentence_spo)
# cls_rep = model.cls_token_rep(embeddings)
# class_logits = model.cross_emb_score(cls_rep)
# loss = model(**batch)["loss"]
# print(loss)
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
