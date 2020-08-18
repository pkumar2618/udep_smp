import json
import sys
from rdflib import URIRef

sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/candidate_generation_fb')
from searchIndex import entitySearch

import re
from urllib.parse import urlparse, urlunparse

if __name__=='__main__':
    with open('./data/WebQSP.test.json', 'r') as f_read:
        json_dict = json.load(f_read)

    count_test = 0
    count_processed =0
    with open('select_few_with_mid_in_es.json', 'w') as f_write:
        for question_query_dict in json_dict["Questions"]:
            question = question_query_dict['RawQuestion']
            parses = question_query_dict['Parses']
            parse = parses[0] 
            query = parse['Sparql']
            mid_name = parse["TopicEntityName"]
            mid = parse["TopicEntityMid"]
            if query is not "null":
                count_test += 1

            result_es = entitySearch(mid_name)
            for result in result_es:
                if re.search(mid, result[1]):
                    f_write.write(question+'\n')
                    count_processed +=1
                    break
    print(f'count_test {count_test}')
    print(f'count_processed {count_processed}')
