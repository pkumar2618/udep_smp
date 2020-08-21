import json
import sys
from rdflib import URIRef
sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/query_lib')
from query_graph import QueryGraph
import re
from urllib.parse import urlparse, urlunparse


with open('qald6_test_combined_with_query.csv', 'r') as f_read:
    lines = f_read.readlines()
    for line in lines:
        try:
            line_split = line.split(',')
            query_str = line_split[4]
            question = line_split[3]
            query = query_str.strip().strip('{}').split(':',1)[1].strip().strip("''")
            if (len(query) == 0 or query == '{}'):
                continue
            elif re.match(r'[\s*]OUT OF SCOPE[\s*]', query):
                # json_qspo.append(json_item)
                continue
            else: 
                # SPO triples are in bgp. We need to use rdflib to parse and get us the spo triples.
                q_graph = QueryGraph(query, {})
                triplets =  q_graph.triples_list
                q_type = q_graph.query_type
                if q_type == 'SelectQuery':
                    if len(triplets) == 1:
                        with open('select_t1.csv', 'a') as f_write:
                            f_write.write(line)
                        with open('nlqs_select_t1.txt', 'a') as f_write:
                            f_write.write(question)
                    elif len(triplets) == 2:
                        with open('select_t2.csv', 'a') as f_write:
                            f_write.write(line)
                        with open('nlqs_select_t2.txt', 'a') as f_write:
                            f_write.write(question)
                    else:
                        with open('select_t3p.csv', 'a') as f_write:
                            f_write.write(line)
                        with open('nlqs_select_t3p.txt', 'a') as f_write:
                            f_write.write(question)

                elif q_type == 'AskQuery':
                    if len(triplets) == 1:
                        with open('ask_t1.csv', 'a') as f_write:
                            f_write.write(line)
                        with open('nlqs_ask_t1.txt', 'a') as f_write:
                            f_write.write(question)
                    elif len(triplets) == 2:
                        with open('ask_t2.csv', 'a') as f_write:
                            f_write.write(line)
                        with open('nlqs_ask_t2.txt', 'a') as f_write:
                            f_write.write(question)
                    else:
                        with open('ask_t3p.csv', 'a') as f_write:
                            f_write.write(line)
                        with open('nlqs_ask_t3p.txt', 'a') as f_write:
                            f_write.write(question)

        except Exception as e:
            pass
