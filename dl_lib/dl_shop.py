import logging
import torch
import sys
sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/dl_lib')
from dl_utilities import ConfigJSON
import os
import pickle
import json
import argparse

if __name__ =="__main__":
    input_dict = {
        "question": "Give me all types of eating disorders.",
        "spos": [
            [
                "uri",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://dbpedia.org/class/yago/EatingDisorders"
            ],
            [
                "http://dbpedia.org/resource/Washington_(state)",
                "http://dbpedia.org/property/largestmetro",
                "uri"
            ]

        ],
        "spos_label": [
            [
                "uri",
                "type",
                "Eating Disorders"
            ],
            [
                "Washington state",
                "largestmetro",
                "uri"
            ]
        ]
    }

    arguments_parser = argparse.ArgumentParser(
       prog='Entity Disambiguation',
       description="Will take candidates entity(relation) and re-rank them to get the top candidate for final use in the"
                   "query.")
    arguments_parser.add_argument("--training", help="Train the Model, the training-settings are at configuration.json"
                                                    " file", action="store_true")
    arguments_parser.add_argument("--prediction", help="pass the block of candidate <S,P,O> to find out their score.",
                                 action="store_true")
    arguments_parser.add_argument("--test_file", type = str, help="test file to be scored by corss_emb_predictor")
    arguments_parser.add_argument("--new_experiment", help="Start a new training experiment.", action="store_true")
    arguments_parser.add_argument("--iteration_info", help="Provide info on what is new about this iteration.", action="store_true" )
    arguments_parser.add_argument("--iteration_data", type=json.loads, help="Provide the data as string, which will be loaded with json.load")
    arguments_parser.add_argument("--model_file", type = str, help="name of the saved model_file to be used for prediction")
    args = arguments_parser.parse_args()

    if args.prediction:
        from spo_disambiguator import cross_emb_predictor 
        cross_emb_predictor(input_file_str=args.test_file, input_dict=input_dict, write_pred=True, model_file=args.model_file)

    if args.training:
        config = ConfigJSON('configuration.json')
        logging.basicConfig(filename='training.log',level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        if args.new_experiment:
            # we are goint to use a single configuration file for the entire deep learning module.
            config.update(section_name = "training_settings",
                                   data={"seed": 1, "batch_size":8, "learning_rate":3e-4,
                                         "epochs": 8, "USE_GPU": torch.cuda.is_available(),
                                         "training_dataset": "../dataset_WebQSP/webqsp_train.json",
                                         "validation_dataset": "../dataset_WebQSP/webqsp_val.json",
                                        "iteration_number": 0
                                        }
                                )
            config.update(section_name="dataset_settings",
                          data={"testing": False, "testing_samples": 4, "max_seq_len": 100,
                                "max_vocab_size": 100000}
                          )

            config.experiment_info("New Experiment")
            # config.run_cycle_reset() #when called it will reset the experiment run cycle,
            # the training_run in the configuration file will be set to zero. and further iteration will update the value. 
        elif args.experiment_iter:
            # if continuing with the same experimens and only running its further iterations.
            config.iteration_info(iteration_info)
            if iteration_data: #iteration data may be optional.
                config.update(section_name="training_settings", data=iteration_data)

            config.iter_cycle_update()

        from spo_disambiguator import cross_emb_trainer, cross_emb_predictor 
        cross_emb_trainer()
