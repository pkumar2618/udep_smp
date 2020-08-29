import json
import pandas as pd
import sys
from rdflib import URIRef
sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/query_lib')
from query_graph import QueryGraph

sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/candidate_generation_fb')
from searchIndex import entitySearch

import re
from urllib.parse import urlparse, urlunparse
def create_list_nlqs_with_annotation(file_in, file_out):
    with open(file_in, 'r') as f_read:
        json_dict = json.load(f_read)

    f_write = open(file_out, 'w')
    for question_query_dict in json_dict["Questions"]:
        json_item = {'question':[], 'annotation':[], 'gold-query':[]}
        question = question_query_dict['RawQuestion']
        parses = question_query_dict['Parses']
        parse = parses[0] 
        query = parse['Sparql']
        mid_name = parse["TopicEntityName"]
        mid = parse["TopicEntityMid"]
        json_item['question'] = question
        json_item['annotation'] = {"name": mid_name, "annotation": mid}
        json_item['gold-query'] = query
        json_string = json.dumps(json_item)
        f_write.write(json_string+'\n')
    f_write.close()

if __name__=='__main__':
    try:
        file_out ='nlqs_annot_webqsp.json'
        with open(file_out, 'r') as f_read:
            lines = f_read.readlines()
            for line in lines:
                json_item = json.loads(line)
                query = json_item['gold-query']
                if (len(query) == 0 or query == '{}'):
                    continue
                elif re.match(r'[\s*]OUT OF SCOPE[\s*]', query):
                    # json_qspo.append(json_item)
                    continue
                else: 
                    # SPO triples are in bgp. We need to use rdflib to parse and get us the spo triples.
                    try:
                        q_graph = QueryGraph(query, {})
                        triplets =  q_graph.triples_list
                        q_type = q_graph.query_type
                        if q_type == 'SelectQuery':
                            if len(triplets) == 1:
                                with open('nlqs_webqsp_select_t1.json', 'a') as f_write:
                                    f_write.write(line)
                            elif len(triplets) == 2:
                                with open('nlqs_webqsp_select_t2.json', 'a') as f_write:
                                    f_write.write(line)
                            else:
                                with open('nlqs_webqsp_select_t3p.json', 'a') as f_write:
                                    f_write.write(line)

                        elif q_type == 'AskQuery':
                            if len(triplets) == 1:
                                with open('nlqs_webqsp_ask_t1.json', 'a') as f_write:
                                    f_write.write(line)
                            elif len(triplets) == 2:
                                with open('nlqs_webqsp_ask_t2.json', 'a') as f_write:
                                    f_write.write(line)
                            else:
                                with open('nlqs_webqsp_ask_t3p.json', 'a') as f_write:
                                    f_write.write(line)
                    except Exception as e:
                        pass

    except FileNotFoundError as e:
        file_in ='./data/WebQSP.test.json'
        file_out ='nlqs_annot_webqsp.json'
        create_list_nlqs_with_annotation(file_in, file_out)

