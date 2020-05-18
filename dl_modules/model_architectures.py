import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.models import Model
from typing import Dict, List, Any
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# We need to use BERT pre-trained model to create the embedding for the input tokens
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.vocabulary import Vocabulary

class CrossEncoderModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 cls_token_rep: Seq2VecEncoder,
                 vocab: Vocabulary,
                 out_sz: int = 1
                 ):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.cls_token_rep = cls_token_rep
        self.cross_emb_score = nn.Linear(self.cls_token_rep.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, sentence_spo: Dict[str, torch.Tensor],
                sentence_spo_raw: Any,
                labels: List[torch.Tensor] = None) -> torch.Tensor:
        all_sample_logits = list()
        # all_sample_labels = list()
        for sentence_spo_block in sentence_spo['sentence_spo']:
            for sentence_spo_tokens in sentence_spo_block:
                # mask = get_text_field_mask(sentence_spo_tokens)
                bert_embedding = self.word_embeddings({'sentence_spo': sentence_spo_tokens.unsqueeze(dim=0)})
                cls_token_rep = self.cls_token_rep(bert_embedding.squeeze(0))
                cross_emb_score_logit = self.cross_emb_score(cls_token_rep)
                all_sample_logits.append(cross_emb_score_logit)

        output = {'sentence_spo_logits': all_sample_logits,
                  'sentence_spo_labels': labels}
        if labels is not None:
            loss = self.loss(torch.cat(all_sample_logits), labels.view(-1))
            output["loss"] = loss

        return output


bert_embedder = PretrainedBertEmbedder(
    pretrained_model="bert-large-uncased",
    top_layer_only=True,  # conserve memory
    # pretrained_model="bert-base-uncased",
    # top_layer_only=True, # conserve memory
)

# The embedder gets us an embedding, here a werd embedding.
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"sentence_spo": bert_embedder},
                                                            # we'll be ignoring masks so we'll need to set this to True
                                                            allow_unmatched_keys=True)

BERT_DIM = word_embeddings.get_output_dim()


# After we have the embedding vectors for all the tokens in the sentence, we need to reduce
# this sequence of vectors into a single vector (Humeau 2019 poly-encoder).
# we are going to use the representation of [CLS] token
class Encoder(Seq2VecEncoder):
    """
    Any model can be considered to be formed of an ecoder and a decoder. We here define the ecoder just something which
    takes out the first token ([CLS]) representation out of the bert_embedding obtained.
    """

    def forward(self, emb_seq: torch.tensor) -> torch.tensor:
        # extract first token tensor
        return emb_seq[0, :]

    def get_output_dim(self) -> int:
        return BERT_DIM
