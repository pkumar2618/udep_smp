from nl_utils import *
from ug_utils import *
from nl_to_ug_api import *
from ug_to_g_api import *

while True:
    """
    Continuously run, unless interrupted. 
    """
    try:
        """
        Continuously look for input question and spit out Answer to the question,
        unless interrupted. 
        """
        print("Enter your question or questions of file containing the questions")
        filename = input()
        parser(filename)

    except FileNotFoundError f_error:
        print("File named %s" % filename, f_error)




