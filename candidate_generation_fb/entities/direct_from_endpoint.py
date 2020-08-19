import json
from SPARQLWrapper import SPARQLWrapper, JSON
import subprocess
import multiprocessing as mp
import time
import re
import os

def name_direct_from_endpoint(mid):
    mid = mid.strip().strip('<>').lstrip('\/')
    query = f'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?y WHERE {{ns:{mid} ns:type.object.name ?y.}}'
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
    json_item = {"_index": "fbentityindex", "_type":"common.topic", "_source":{'uri':'', 'label': ''}}
    mid_uri = "http://rdf.freebase.com/ns/"
    mid = mid_uri + mid
    json_item['_source']['uri'] = mid
    try:
        y_values  = results["results"]["bindings"]
        # print(result_list_dict["label"]["value"])
        name_bup = y_values[0]['y']['value']
        for y in y_values:
            name = y['y']['value']
            lang = y['y']['xml:lang']
            if lang=='@en':
                json_item['_source']['label'] = name
                break
            else:
                continue
        if json_item['_source']['label'] == '':
            json_item['_source']['label'] = name_bup

        json_object = json.dumps(json_item)
        return json_object
    except Exception as e:
        return '\n'
        #print(f'No label found: {e}')

if __name__ == '__main__':
    file_with_mids ="fb-rdf-topic-s02-c01"  
    with open(file_with_mids, 'r') as f_read_topic:
        lines = f_read_topic.readlines()

    # Preserve the triples with the ids, and get label from label-file 
    for line_id in lines:
        # find the mid in the line_id
        line_id_split = line_id.split('\t')
        mid = line_id_split[0].strip()

        if re.match(r'</m.[a-zA-Z0-9_]+>', mid):
            json_item = name_direct_from_endpoint(mid)
            with open("direct_entity_dump.json", 'a') as f_write:
                f_write.write(json_item +'\n')
