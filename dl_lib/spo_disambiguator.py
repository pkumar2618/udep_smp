import logging
import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BasicIterator
import sys
sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/dl_lib')
from data_loader import QuestionSPOReader, bert_token_indexer, bert_tokenizer
from dl_utilities import ConfigJSON
from model_architectures import CrossEncoderModel, word_embeddings, Encoder
from predictor import Predictor
import os
import pickle
import json
import argparse

def cross_emb_predictor(input_file_str=None, input_dict=None, write_pred=False, model_file=None):
    """
    The input file is json with item as dict =
    {
        "question": "Give me all types of eating disorders.",
        "spos": [[]],
        "spos_label": [[]]
    },
    :param input_file_str:
    :return:
    """
    #logging.basicConfig(filename='predictor.log',level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    USE_GPU = torch.cuda.is_available()
    # vocab = Vocabulary.from_files("./vocabulary")
    reader = QuestionSPOReader(my_tokenizer=bert_tokenizer,
                               my_token_indexers={"sentence_spo": bert_token_indexer})

    # test_ds = reader.read("../dataset_qald/qald_test.json")
    # test_ds = test_ds[:3]
    test_ds = reader.read(file_path=input_file_str, input_dict=input_dict) #one of them must be None
    vocab = Vocabulary()
    seq_iterator = BasicIterator(batch_size=1)
    seq_iterator.index_with(vocab)
    # instantiating the model
    cls_token_encoder = Encoder(vocab)
    model = CrossEncoderModel(word_embeddings, cls_token_encoder, vocab)
    if USE_GPU:
        model.cuda()
    else:
        model
    # loading model_state from the saved model
    device = torch.device("cuda")
    if not model_file:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_file_abs = os.path.join(dir_path, 'model.th')
        with open(model_file_abs, 'rb') as f_model:
            model.load_state_dict(torch.load(f_model, map_location='cuda:0'))
            model.to(device)

    elif model_file:
        try:
            with open(model_file, 'rb') as f_model:
                model.load_state_dict(torch.load(f_model, map_location='cuda:0'))
                model.to(device)
        except:
            raise

    predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
    test_preds = predictor.predict(test_ds, write_pred=write_pred)
    return test_preds
    #print(test_preds)
    #print("prediction done, see the output_prediction.json")


def cross_emb_trainer():
    # loading the configuration data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = ConfigJSON(os.path.join(dir_path,'configuration.json'))

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
        num_epochs=config.config["training_settings"]["epochs"],
    )

    metrics = trainer.train()
    # print(metrics)

    # Save the model after training is complete
    # Here's how to save the model.
    with open('model.th', 'wb') as f:
        torch.save(model.state_dict(), f)
    # save the vocabulary
    # vocab.save_to_files("./vocabulary")
