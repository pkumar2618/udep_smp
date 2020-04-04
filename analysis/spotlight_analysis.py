import spotlight
import re
import ast

with open('q1_entities.txt', 'r') as f:
    for line in f:
        splits = re.split(r'[,]+?', line)
        entities =splits[1:]

        for entity_phrase in entities:
            try:
                db_resources = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', entity_phrase,
                                                                      confidence=0.0, support=0)
            except Exception as e:
                print('nothing returned etc')