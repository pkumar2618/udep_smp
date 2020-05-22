from udep_lib.parser import Parser
# from pre_processor import PreProcessor
import pickle
import argparse
import sys

sys.path.insert(0, './dl_lib')
sys.path.insert(0, './udep_lib')
sys.path.insert(0, './candidate_generation')

arguments_parser = argparse.ArgumentParser(
    prog='NLQA',
    description="Take a Natural Language Question and provide it's answer.")

arguments_parser.add_argument("--questions_file", help="Path to the file containing Natural Language Questions")
arguments_parser.add_argument("--questions_list", help="List all the questions one after another,"
                                                       " separated by new line")
arguments_parser.add_argument("-question", help="Ask your questions!")

arguments_parser.add_argument("--canonical_form", help="enable the intermediate canonical form, default is disable",
                              action="store_true")

arguments_parser.add_argument("--dependency_parsing", help="Dependency parsing with universal dependency parser, "
                                                           "default is disable",
                              action="store_true")

arguments_parser.add_argument("--disambiguator", help="select linker for disambiguation among spotlight and "
                                                      "custom", choices=['spotlight', 'custom'])

arguments_parser.add_argument("--knowledge_graph", help="select the KG to be used for querying",
                              choices=['dbpedia', 'wikidata', 'freebase'])

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
        if args.questions_file or args.question: # per question need to work out
            # # # # current pipeline,
            # # # # nl_question -> tokenization -> canonicalization(default bypassed)->disambiguation->formalization
            # # # # Tokenize the list of questions.
            # save the name of the question_sets under consideration.
            qset = "select_2spo"
            try:
                with open(f'{qset}_log0_parser.pkl', 'rb') as f:
                    parser = pickle.load(f)
            except FileNotFoundError as e:
                # # # provide the file name containing the questions
                with open(args.questions_file, 'r') as file_obj:
                    parser = Parser.from_file(file_obj)

                parser.tokenize(args.dependency_parsing)
                # for nlq_tokens in parser.nlq_tokens_list:
                #     print(nlq_tokens)

                # # # saving the parser state using pickle
                with open(f'{qset}_log0_parser.pkl', 'wb') as f:
                    pickle.dump(parser, f)


            # # canonicalize based on canonical_form flag and dependency_parsing flag. when canonical_form flag is
            # # disabled the parser sets it's attribute self.canonical_list as copy of self.nlq_token_list
            try:
                with open(f'{qset}_log1_parser.pkl', 'rb') as f:
                    parser = pickle.load(f)
            except FileNotFoundError as e:
                parser.canonicalize(args.dependency_parsing, args.canonical_form)
                with open(f'{qset}_log1_parser.pkl', 'wb') as f:
                    pickle.dump(parser, f)


            # # convert the question into a Query, the reference to knowledge graph is rquired to provide list of namespace
            # # prefixes used during creating a query-string
            try:
                with open(f'{qset}_log2_parser.pkl', 'rb') as f:
                    parser = pickle.load(f)
            except FileNotFoundError as e:
                parser.ungrounded_logical_form()
                with open(f'{qset}_log2_parser.pkl', 'wb') as f:
                    pickle.dump(parser, f)


            # translate logical form into graphical representation using SPARQL basic graph pattern (BGP)
            try:
                with open(f'{qset}_log3_parser.pkl', 'rb') as f:
                    parser = pickle.load(f)
            except FileNotFoundError as e:
                parser.ungrounded_sparql_graph(kg=args.knowledge_graph)
                with open(f'{qset}_log3_parser.pkl', 'wb') as f:
                    pickle.dump(parser, f)


            # entity linking or disambiguation is an required for the tokens in the questions. The disambiguator
            # provides denotation (entity or resources) for each token,
            # the parser stores a dictionary of token-denotation pairs
            try:
                with open(f'{qset}_log4_parser.pkl', 'rb') as f:
                    parser = pickle.load(f)
            except FileNotFoundError as e:
                parser.grounded_sparql_graph(linker=args.disambiguator, kg=args.knowledge_graph)
                with open(f'{qset}_log4_parser.pkl', 'wb') as f:
                    pickle.dump(parser, f)

            # parser.query_executor(args.knowledge_graph)

            # Result of Querying the Knowledge Graph
            print("done")

        elif args.questions_list:
            pass

        elif args.question:
            pass

        break

    except FileNotFoundError as f_error:
        print("File named %s not found" % args.questions_file, f_error)
