import json
import pandas as pd
import sys
from rdflib import URIRef
sys.path.append('/home/pawan/projects/aihn_qa/udep_smp/query_lib')

import argparse
import os

arguments_parser = argparse.ArgumentParser(
    prog='given the csv file with classes of failing cases create separate files nlqs',
    description="find the corresponding question class.")

arguments_parser.add_argument("--nlqs_dataset_file", help="file containing original questions")
arguments_parser.add_argument("--nlqs_cls_csv", help="file containing classe of failing questions")

args = arguments_parser.parse_args()

if __name__=='__main__':
    #load the csv file into a panda dataframe
    question_cls = pd.read_csv(f'{args.nlqs_cls}')
    dataset_file ='nlqs_annot_webqsp.json'
    with open(dataset_file, 'r') as f_read:
        lines = f_read.readlines()
        for line in lines:
            json_item = json.loads(line)
            question = json_item['question']
            if question in question_cls['question']:
                idx = question_cls['question'].index()
                error_cls = question_cls['Error-Type'][idx]
                with open(f'{error_cls}_nlqs.json', 'w+') as f_write:
                    f_write.write(line)
