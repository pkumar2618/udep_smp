# -*- coding: utf-8 -*-
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


import json
import sys
from rdflib import URIRef

sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/query_lib')
from query_graph import QueryGraph

from SPARQLWrapper import SPARQLWrapper, JSON
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
            if re.match(r'm.[a-zA-Z0-9]+', value_label): #it is a mid in freebase
                #todo: later using elastic_search we should get the names of
                #the mid and use it as the spo label
                query = f'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?y WHERE {{ns:{value_label} ns:type.object.name ?y.}}'
                sparql_endpoint = SPARQLWrapper("http://10.208.20.61:8895/sparql/")
                sparql_endpoint.setReturnFormat(JSON)
                results = None 
                try:
                    sparql_endpoint.setQuery(query)
                    results = sparql_endpoint.query().convert()

                except:
                    #print("error quering endpoint")
                    results = None

                # extract only english name
                try:
                    y_values  = results["results"]["bindings"]
                    # print(result_list_dict["label"]["value"])
                    name_bup = y_values[0]['y']['value']
                    value_label = None
                    for y in y_values:
                        name = y['y']['value']
                        lang = y['y']['xml:lang']
                        if isEnglish(name):
                            value_label = name
                            break
                        else:
                            continue
                    if value_label is None:
                        value_label = name_bup
                    return value_label
                except Exception as e:
                    return ''
            else: #not a mid but a predicate of the form type.object.type
                return value_label.split('.')[-1]
                url_base = urlunparse((url_parsed.scheme, url_parsed.netloc, '/'.join(url_path_split[:-1]), "", "", ""))
                    #return split_camelcase_predicates(value_label)
    else:
        return 'uri'

def split_camelcase_predicates(cc_predicate):
    list_words = re.findall(r'[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', cc_predicate)
    return ' '.join(list_words)

try:
    with open('WebQSP_processed.train.json', 'r') as f_read:
        json_qspo = json.load(f_read)

    total_len = len(json_qspo)
    train_len = int(0.9*total_len)
    val_len = total_len - train_len

    with open('webqsp_train.json', 'w') as f_write:
        json.dump(json_qspo[:train_len], f_write, indent=4)

    with open('webqsp_val.json', 'w') as f_write:
        json.dump(json_qspo[train_len:total_len], f_write, indent=4)

except Exception as e:
    #with open('./data/WebQSP.train.json', 'r') as f_read:
    with open('./data/WebQSP.test.json', 'r') as f_read:
        json_dict = json.load(f_read)
        json_qspo = []
        json_item = {}
        count_train = 0
        count_processed =0
        for question_query_dict in json_dict["Questions"]:
            question = question_query_dict['RawQuestion']
            parses = question_query_dict['Parses']
            parse = parses[0] 
            query = parse['Sparql']
            if query is not "null":
                count_train += 1

            # SPO triples are in bgp. We need to use rdflib to parse and get us the spo triples.
            q_graph = QueryGraph(query, {})
            json_item = {'question': question, 'spos': q_graph.triples_list}

            # Get label for all the spos
            spos_label = []
            try:
                for uri_tuple in q_graph.triples_list:
                    spos_label.append(tuple([get_label_from_uri(uri) for uri in uri_tuple]))

                # append all the tuples of spo_labels to json_item
                json_item['spos_label'] = spos_label
                json_qspo.append(json_item)
                count_processed +=1
            except TypeError as e_uri_label:
                print(e_uri_label)
                #todo need to look out why there are zero tuples.
                # json_item['spos_label'] = None

    print(f'count_traing {count_train}')
    print(f'count_processed {count_processed}')
    #with open('WebQSP_processed.train.json', 'w') as f_write:
    with open('WebQSP_processed.test.json', 'w') as f_write:
        json.dump(json_qspo, f_write, indent=4)
