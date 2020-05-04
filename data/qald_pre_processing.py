import json
import rdflib

with open('qald_combined.json', 'r') as f_read:
    json_dict = json.load(f_read)
    json_qspo = []
    for question_query_dict in json_dict:
        question = question_query_dict['question']
        query = question_query_dict['query']
        # SPO triples are in bgp. We need to use rdflib to parse and get us the spo triples.

