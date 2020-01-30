from nl_utils import *
from ug_utils import *
from nl_to_ug import *
from ug_to_g import *
from parser import Parser
from pre_processor import PreProcessor
import argparse

arguments_parser = argparse.ArgumentParser(
    prog= 'NLQA',
    description="Take a Natural Language Question and provide it's answer.")

arguments_parser.add_argument("--questions_file", help="Path to the file containing Natural Langugae Questions" )
arguments_parser.add_argument("--questions_list", help = "List all the questions one after another, separated by new line")
arguments_parser.add_argument("-question", help = "Ask your questions!")

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
        print("Enter name of the file containing the questions")
        if args.questions_file:
            # filename = input()
            filename = args.questions_file
            file_obj = open(filename, 'r')
            with open(filename, 'r') as file_obj:
                PreProcessor.from_file(file_obj)



        elif args.questions_list:
            pass

        elif args.question:
            pass

        break

    except FileNotFoundError as f_error:
        print("File named %s not found" % filename, f_error)