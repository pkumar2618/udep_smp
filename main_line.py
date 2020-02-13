from nl_utils import *
from ug_utils import *
from nl_to_ug import *
from ug_to_g import *
from parser import Parser
from pre_processor import PreProcessor
import argparse
import sys
sys.path.insert(0, './dl_modules')

arguments_parser = argparse.ArgumentParser(
    prog= 'NLQA',
    description="Take a Natural Language Question and provide it's answer.")

arguments_parser.add_argument("--questions_file", help="Path to the file containing Natural Langugae Questions" )
arguments_parser.add_argument("--questions_list", help="List all the questions one after another, separated by new line")
arguments_parser.add_argument("-question", help="Ask your questions!")
arguments_parser.add_argument("--canonical_form", help="enable the intermediate canonical form, default is disable",
                              action="store_true")
arguments_parser.add_argument("--dependency_parsing", help="Dependency parsing with universal dependency parser, "
                                                           "default is disable",
                              action="store_true")
arguments_parser.add_argument("--disambiguator", help="select linker for disambiguation among spotlight and "
                                                       "custom", choices=['spotlight', 'custom'])

arguments_parser.add_argument("--one_hot", help="Take the questions and create a one-hot representation based on "
                                                "dbpedia vocabulary.")
arguments_parser.add_argument("--input_embedding", help="Enter type of word_embedding to be used.",
                              choices=['Word2Vec', 'GloVe'])

args = arguments_parser.parse_args()

while True:
    """
    Continuously run, unless interrupted. 
    """
    try:
        """
        Continuously look for input question and spit out Answer to the question,
        unless interrupted. 
        """
        # print("Enter name of the file containing the questions")
        if args.questions_file:
            # filename = input()
            filename = args.questions_file
            # with open(filename, 'r') as file_obj:
            #     pre_processor = PreProcessor.from_file(file_obj)
            #
            # pp_questions = pre_processor.get_pp_questions()

            with open(filename, 'r') as file_obj:
                parser = Parser.from_file(file_obj)

            # current pipeline,
            # tokenization -> canonicalization(default bypassed)->disambiguation->formalization
            # Tokenize the list of questions.
            parser.tokenize(args.dependency_parsing)

            # for nlq_tokens in parser.nlq_tokens_list:
            #     print(nlq_tokens)

            # canonicalize based on canonical_form flag and dependency_parsing flag. when canonical_form flag is
            # disabled the parser sets it's attribute self.canonical_list as copy of nlq_token_list
            parser.canonicalize(args.dependency_parsing, args.canonical_form)

            # entity linking or disambiguation is an required for the tokens in the questions.
            # if args.disambiguator:
            #     parser.disambiguate(args.disambiguator)
            # else:
            #     parser.disambiguate(None)

            # convert the question into a Query
            parser.formalize()

            # Reuslt of Querying the Knowledge Graph
            print("done")

        elif args.questions_list:
            pass

        elif args.question:
            pass

        break

    except FileNotFoundError as f_error:
        print("File named %s not found" % filename, f_error)