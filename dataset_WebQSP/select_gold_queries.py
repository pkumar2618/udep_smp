import json
import pandas as pd
import sys
from rdflib import URIRef

sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/candidate_generation_fb')
from searchIndex import entitySearch

import re
from urllib.parse import urlparse, urlunparse

if __name__=='__main__':
    file_name ='gold_webqsp.csv'
    with open('./data/WebQSP.test.json', 'r') as f_read:
        json_dict = json.load(f_read)

    columns_dict = {'question':[], 'gold-query':[]}
    count_test = 0
    count_processed =0
    for question_query_dict in json_dict["Questions"]:
        question = question_query_dict['RawQuestion']
        parses = question_query_dict['Parses']
        parse = parses[0] 
        query = parse['Sparql']
        columns_dict['question'].append(question)
        columns_dict['gold-query'].append(json.dumps(query))

    df_table = pd.DataFrame(columns_dict)
    df_table.to_csv(file_name, sep=',', index=False)
