import json
import pandas as pd
import sys
import argparse
from rdflib import URIRef
sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/query_lib')
from query_graph import QueryGraph

#sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/candidate_generation_fb')

arguments_parser = argparse.ArgumentParser(
    prog='classifier for failure case',
    description="the failing cases may be classified into category based on what caused them to fail")

arguments_parser.add_argument("--output_file", help="file with query output and query string")
arguments_parser.add_argument("--output_file_dir", help="directory of output file, where the extension are json.")
arguments_parser.add_argument("--benchmark", help="benchmark dataset for question answering", choices=['qald-6', 'WebQSP'])
arguments_parser.add_argument("--printflag", help="enable print when a matched answer is found", choices=['match', 'both'])

args = arguments_parser.parse_args()

if __name__=='__main__':
    result_file_full_path = []
    if args.output_file:
        result_file_full_path = [args.output_file]

    for result_file in result_file_full_path:
        # will go through the question in the output file one by one 
        f_read = open(result_file, 'r')
        lines = f_read.readlines()
        for question_num, line in enumerate(lines):
            json_item = json.loads(line)
            question = json_item["question"].strip()
            # for every question we start with answer found flag as False, 
            # as we go over all the possible queries for this question we will change this flag True
            clf_noEdge =True # assume there is no Edge, if we found an edge for any one query, we will turn this flag
            # clf_noEdge =False
            # iterating through the list of generated queries for a question
            # find out if there was any correct answer,
            
            for i, query in enumerate(json_item["topk_queries"]):
                query_string = query["query_string"]
                query_out_dict = query["query_output"]
                if not query_out_dict:# if there is no output and it is an empty dict 
                    # check if the query string has empty bgp:
                    try:
                        q_graph = QueryGraph(query, {})
                        triplets =  q_graph.triples_list
                        q_type = q_graph.query_type
                        if len(triplets) >= 1:
                            clf_noEdge = False
                            break
                        else:
                            continue
                    except TypeError as e:
                        # the triplets is None
                        continue

            if clf_noEdge:
                with open(f'{args.benchmark}_clf_noEdge.txt', 'a') as f_write1:
                    f_write1.write(question+'\n')
            elif not clf_noEdge:
                with open(f'{args.benchmark}_clf_bad_query.txt', 'a') as f_write2:
                    f_write2.write(question)
