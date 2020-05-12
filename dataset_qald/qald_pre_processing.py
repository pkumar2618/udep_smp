import json
from rdflib import URIRef
from my_libs.query_graph import QueryGraph
import re
from urllib.parse import urlparse, urlunparse


def get_label_from_uri(uri):
    if isinstance(uri, URIRef):
        url_parsed = urlparse(uri)
        if url_parsed.fragment is not '':
            value_label = url_parsed.fragment
            url_base = urlunparse((url_parsed.scheme, url_parsed.netloc, url_parsed.path, "", "", ""))
        else:
            url_path_split = url_parsed.path.split('/')
            value_label = url_path_split[-1]
            url_base = urlunparse((url_parsed.scheme, url_parsed.netloc, '/'.join(url_path_split[:-1]), "", "", ""))
        return split_camelcase_predicates(value_label)
    else:
        return uri

def split_camelcase_predicates(cc_predicate):
    list_words = re.findall(r'[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', cc_predicate)
    return ' '.join(list_words)

with open('qald_combined.json', 'r') as f_read:
    json_dict = json.load(f_read)
    json_qspo = []
    json_item = {}
    for question_query_dict in json_dict:
        query = question_query_dict['query']
        try:
            if (len(query) == 0 or query == '{}'):
                json_item = {'question': question_query_dict['question'], 'spos': None}
                # json_qspo.append(json_item)
                continue
            elif re.match(r'[\s*]OUT OF SCOPE[\s*]', query):
                json_item = {'question': question_query_dict['question'], 'spos': None}
                # json_qspo.append(json_item)
                continue
        except Exception as may_be_dict:
            try:
                query = query['sparql']
            except Exception as not_a_dict:
                query = query.strip()

        # SPO triples are in bgp. We need to use rdflib to parse and get us the spo triples.
        q_graph = QueryGraph(query, {})
        json_item = {'question': question_query_dict['question'], 'spos': q_graph.triples_list}

        # Get label for all the spos
        spos_label = []
        try:
            for uri_tuple in q_graph.triples_list:
                spos_label.append(tuple([get_label_from_uri(uri) for uri in uri_tuple]))

            # append all the tuples of spo_labels to json_item
            json_item['spos_label'] = spos_label
            json_qspo.append(json_item)
        except TypeError as e_uri_label:
            pass
            #todo need to look out why there are zero tuples.
            # json_item['spos_label'] = None

        # don't create json_item with None label in spos_label
        # if json_item['spos'] is not None: # non zero length of triple
        #     if len(spos_label): # non zero length
        #         json_qspo.append(json_item)
        # else: #zero triples in the label
        #     continue

with open('qald_input.json', 'w') as f_write:
    json.dump(json_qspo, f_write, indent=4)