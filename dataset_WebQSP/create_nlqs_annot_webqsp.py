import json
import pandas as pd
import sys
from rdflib import URIRef

sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/candidate_generation_fb')
from searchIndex import entitySearch

import re
from urllib.parse import urlparse, urlunparse

if __name__=='__main__':
    file_name ='nlqs_annot_webqsp.json'
    with open('./data/WebQSP.test.json', 'r') as f_read:
        json_dict = json.load(f_read)

    f_write = open('nlqs_annot_webqsp.json', 'w')
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
