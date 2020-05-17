import logging
import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary


from allennlp.training.trainer import Trainer
from dl_modules.data_loader import QuestionSPOReader, bert_token_indexer, bert_tokenizer
from dl_modules.dl_utilities import ConfigJSON

from udeplib.parser import Parser
# from pre_processor import PreProcessor
import pickle
import argparse
import sys


arguments_parser = argparse.ArgumentParser(
    prog='Entity Disambiguation',
    description="Will take candidates entity(relation) and re-rank them to get the top candidate for final use in the"
                "query.")

arguments_parser.add_argument("--training", help="Train the Model, the training-settings are at configuration.json"
                                                 " file", action="store_true")
arguments_parser.add_argument("--prediction", help="pass the list of candidate <S,P,O> to find out their score")

args = arguments_parser.parse_args()

if args.training:
    # we are goint to use a single configuration file for the entire deep learning module.
    config = ConfigJSON('configuration.json')
    config.update(section_name = "training_settings",
                           data={"seed": 1, "batch_size":2, "learning_rate":3e-4,
                                 "epochs": 3, "USE_GPU": torch.cuda.is_available(),
                                 "training_dataset": "../dataset_qald/qald_train.json",
                                "validation_dataset": "../dataset_qald/qald_val.json"}
                           )

    logging.basicConfig(filename='spo_disambiguator.log',level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    torch.manual_seed(config.config["training_settings"]["seed"])

    # The sentence need to be converted into a tensor(embedding space),
    # but before that the tokens are mapped to an integer using the token_indexer.
    # For BERT pre-trained we will use indexer, vocab and embedder all from the BERT pre-trained.
    # Note that the BERT's wordpiece tokenizer need to be used


    # creating the dataset reader
    reader = QuestionSPOReader(my_tokenizer=bert_tokenizer,
                           my_token_indexers={"sentence_spo": bert_token_indexer})

    train_ds = reader.read(config.config["training_settings"]["training_dataset"])
    val_ds = reader.read(config.config["training_settings"]["validation_dataset"])

    ### the iterator is used to batch the data and prepare it for input to the model
    from allennlp.data.iterators import BucketIterator
    iterator = BucketIterator(batch_size=config.config["training_settings"]["batch_size"],
                              sorting_keys=[('sentence_spo', 'num_fields'), ('sentence_spo', 'list_num_tokens')])

    # # the iterator has to be informed about how the indexing has to be done, indexing require the vocabulary
    # # of all the tokens (may be trimmed if the vocab size is huge).
    vocab = Vocabulary()
    iterator.index_with(vocab)
    # # example batch
    # batch = next(iter(iterator(train_ds)))
    # print(batch)
    # print(batch['sentence_spo']['sentence_spo'].shape)


    # instantiating the model
    from dl_modules.model_architectures import CrossEncoderModel, word_embeddings, Encoder
    cls_token_encoder = Encoder(vocab)
    model = CrossEncoderModel(word_embeddings, cls_token_encoder, vocab)
    if config.config["training_settings"]["USE_GPU"]:
        model.cuda()
    else:
        model

    # # Sanity Check
    # # example batch
    # batch = next(iter(iterator(train_ds)))
    # sentence_spo = batch["sentence_spo"]
    # # mask = get_text_field_mask(sentence_spo)
    # embeddings = model.word_embeddings(sentence_spo)
    # cls_rep = model.cls_token_rep(embeddings)
    # class_logits = model.cross_emb_score(cls_rep)
    # loss = model(**batch)["loss"]
    # print(loss)

    # training
    optimizer = optim.Adam(model.parameters(), lr=config.config["training_settings"]["learning_rate"])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        cuda_device=0 if config.config["training_settings"]["USE_GPU"] else -1,
        num_epochs=config.config["training.settings"]["epochs"],
    )

    metrics = trainer.train()
    # print(metrics)

    # Save the model after training is complete
    # Here's how to save the model.
    with open('model_bup.th', 'wb') as f:
        torch.save(model.state_dict(), f)
    # save the vocabulary
    vocab.save_to_files("./vocabulary")


if args.prediction:
    pass
