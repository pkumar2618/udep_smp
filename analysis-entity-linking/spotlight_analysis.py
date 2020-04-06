import spotlight
import re
import json
import ast
json_dict = {}
json_dict['questions'] = []
with open('../scope_questions/q1_entities.txt', 'r') as f:
    for line in f:
        splits = re.split(r'[,]+?', line)
        entities =splits[1:]
        question_dict = {}
        question_dict['question'] = splits[0]
        for entity_phrase in entities:
            try:
                db_resources = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', entity_phrase,
                                                                      confidence=0.0, support=0)
                question_dict[entity_phrase] = db_resources[0]['URI']
            except Exception as e:
                question_dict[entity_phrase] = 'NA'
                print('nothing returned etc')

        json_dict['questions'].append(question_dict)

with open('q1_entities_with_db_resource.json', 'w') as f_json:
    json.dump(json_dict, f_json, indent=4)