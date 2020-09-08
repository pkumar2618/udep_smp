from udep_lib.parser import Parser
import pickle
import pandas as pd
import argparse
import sys
import json

sys.path.insert(0, './udep_lib')

arguments_parser = argparse.ArgumentParser(
    prog='pick the pickle for some value x')

arguments_parser.add_argument("--output_file", help="Name of the file where the extract from the log has to be dumped")
arguments_parser.add_argument("--logname", help="Name of the file where the extract from the log has to be dumped")

args = arguments_parser.parse_args()


#try:
#    with open(f'{args.logname}_log0_parser.pkl', 'rb') as f:
#        parser = pickle.load(f)
##except FileNotFoundError as e:
#    # # # provide the file name containing the questions
#    with open(args.questions_file, 'r') as file_obj:
#        parser = Parser.from_file(file_obj, annotation = args.annotation)
#
#    parser.tokenize(args.dependency_parsing)
#    # for nlq_tokens in parser.nlq_tokens_list:
#    #     print(nlq_tokens)
#
#    # # # saving the parser state using pickle
#    with open(f'{args.logname}_log0_parser.pkl', 'wb') as f:
#        pickle.dump(parser, f)


# # canonicalize based on canonical_form flag and dependency_parsing flag. when canonical_form flag is
# # disabled the parser sets it's attribute self.canonical_list as copy of self.nlq_token_list
#try:
#    with open(f'{args.logname}_log1_parser.pkl', 'rb') as f:
#        parser = pickle.load(f)
#except FileNotFoundError as e:
#    parser.canonicalize(args.dependency_parsing, args.canonical_form)
#    with open(f'{args.logname}_log1_parser.pkl', 'wb') as f:
#        pickle.dump(parser, f)


# # convert the question into a Query, the reference to knowledge graph is rquired to provide list of namespace
# # prefixes used during creating a query-string
#try:
#    with open(f'{args.logname}_log2_parser.pkl', 'rb') as f:
#        parser = pickle.load(f)
#except FileNotFoundError as e:
#    parser.ungrounded_logical_form()
#    with open(f'{args.logname}_log2_parser.pkl', 'wb') as f:
#        pickle.dump(parser, f)


# translate logical form into graphical representation using SPARQL basic graph pattern (BGP)
try:
    with open(f'{args.logname}_log3_parser.pkl', 'rb') as f:
        parser = pickle.load(f)

    columns_dict = {'question':[], 'ug_gpgraphs':[], 'Error-Type': [], 'Description': []}

    for question, ug_gpgraphs in zip(parser.nlq_questions_list, parser.ug_gpgraphs_list):
        question = question.question
        columns_dict['question'].append(question)
        ug_gpgraph = ug_gpgraphs.ug_gpgraphs[0]
        columns_dict['ug_gpgraphs'].append(json.dumps(ug_gpgraph))
        # logic for classification
        edges = ug_gpgraph['Edges']
        if len(edges) == 0:
            columns_dict['Error-Type'].append('No Edge')
            columns_dict['Description'].append('UdepLambda has no edge')
        else: #there are one or more edges
            for edge in edges:
                indices = edge[0].strip('()').split(',')
                if indices[0]==indices[1]  or indices[1]==indices[2] or indices[2]==indices[0]:
                    columns_dict['Error-Type'].append('Half-Edge')
                    columns_dict['Description'].append(f'same two indices: {indices}')
                    break
                else: #all three indices are different
                    columns_dict['Error-Type'].append('System Failure')
                    columns_dict['Description'].append(f'System failed to create SPARQL Query: {indices}')
                    break

    df_table = pd.DataFrame(columns_dict)
    df_table.to_csv(args.output_file, sep=',', index=False)
except FileNotFoundError as e:
    raise
    #parser.ungrounded_sparql_graph(kg=args.knowledge_graph)
    #with open(f'{args.logname}_log3_parser.pkl', 'wb') as f:
    #    pickle.dump(parser, f)


# entity linking or disambiguation is an required for the tokens in the questions. The disambiguator
# provides denotation (entity or resources) for each token,
# the parser stores a dictionary of token-denotation pairs
#start_qn = args.start_qn 
#end_qn = args.end_qn 
#try:
#    with open(f'{args.logname}_log4_{start_qn}_{end_qn}_parser.pkl', 'rb') as f:
#        parser = pickle.load(f)
#except FileNotFoundError as e:
#    parser.grounded_sparql_graph(start_qn = start_qn, end_qn=end_qn, linker=args.disambiguator, kg=args.knowledge_graph)
#    with open(f'{args.logname}_log4_{start_qn}_{end_qn}_parser.pkl', 'wb') as f:
#        pickle.dump(parser, f)
#parser.query_executor(args.knowledge_graph, result_file=f'{args.logname}_execution_result_{start_qn}_{end_qn}.json', start_qn=args.start_qn, end_qn=args.end_qn)

# Result of Querying the Knowledge Graph
print("done")
