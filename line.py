from nl_utils import *
from ug_utils import *
from nl_to_ug import *
from ug_to_g import *
from parser import Parser
from pre_processor import PreProcessor

while True:
    """
    Continuously run, unless interrupted. 
    """
    # try:
    #     """
    #     Continuously look for input question and spit out Answer to the question,
    #     unless interrupted.
    #     """
        # print("Enter your question")
        # nl_question = input()
        # Parser(filename)
    # except Exception:


    # try:
    #     """
    #     Continuously look for input question and spit out Answer to the question,
    #     unless interrupted.
    #     """
    # except:
    #     print("Enter list of questions:")
    #         nlq_list = input()
    #         Parser.from_file(nlq_list)

    try:
        """
        Continuously look for input question and spit out Answer to the question,
        unless interrupted. 
        """
        print("Enter name of the file containing the questions")
        # filename = input()
        filename = "nlqs.txt"
        # file_obj = open(filename, 'r')
        with open(filename, 'r') as file_obj:
            PreProcessor.from_file(file_obj)

        break

    except FileNotFoundError as f_error:
        print("File named %s not found" % filename, f_error)