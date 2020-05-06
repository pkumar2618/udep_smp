import json
import rdflib
from my_libs.query_graph import QueryGraph
import re
# query = eval(query)['sparql']
# q_graph = QueryGraph(query, {})

with open('qald_combined.json', 'r') as f_read:
    json_dict = json.load(f_read)
    json_qspo = []
    json_item = {}
    for question_query_dict in json_dict:
        query = question_query_dict['query']
        try:
            if (len(query) == 0 or query == '{}'):
                json_item = {'question': question_query_dict['question'], 'spos': None}
                json_qspo.append(json_item)
                continue
            elif re.match(r'[\s*]OUT OF SCOPE[\s*]', query):
                json_item = {'question': question_query_dict['question'], 'spos': None}
                json_qspo.append(json_item)
                continue
        except Exception as may_be_dict:
            try:
                query = query['sparql']
            except Exception as not_a_dict:
                query = query.strip()

        # SPO triples are in bgp. We need to use rdflib to parse and get us the spo triples.
        q_graph = QueryGraph(query, {})
        json_item = {'question': question_query_dict['question'], 'spos': q_graph.triples_list}
        json_qspo.append(json_item)

with open('qald_input.json', 'w') as f_write:
    json.dump(json_qspo, f_write, indent=4)

