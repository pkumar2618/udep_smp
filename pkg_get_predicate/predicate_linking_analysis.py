import json
import re
from gensim.models import KeyedVectors
import numpy as np
from db_utils import get_property_using_cosine_similarity, split_camelcase_predicates, dbpedia_property_vectorizer


try:
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
except FileNotFoundError as ef:
    # vectorize the dbpedia properties using glove embedding, to be run if the embedding has to be changed.
    dbpedia_property_vectorizer(w2v_embedding='../glove/glove.840B.300d.w2vformat.txt')
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')

json_dict = {}
json_dict['questions'] = []
# test cosine_smilarity the function will return top-n property based on the cosine similarity

with open('../scope_questions/q1_predicates.txt', 'r') as f:
    for line in f:
        splits = re.split(r'[,]+?', line)
        predicates =splits[1:]
        question_dict = {}
        question_dict['question'] = splits[0]

        for predicate in predicates:
            for word in split_camelcase_predicates(predicate):
                glove_hit = False
                word_count = 0
                try:
                    if word_count == 0:
                        word_vector = glove[word].reshape(1, -1)
                        vector = np.array(word_vector)
                        word_count += 1
                        glove_hit = True # consequent hit, in else condition will keep this true
                    else:
                        word_vector = glove[word].reshape(1, -1)
                        vector = np.append(vector, word_vector, axis=0)
                        word_count += 1
                except KeyError as e:
                    pass

            if glove_hit:
                predicate_emb = np.average(vector, axis=0)
                db_property = get_property_using_cosine_similarity(predicate_emb)
                question_dict[predicate.rstrip()] = db_property['value_label']
                question_dict[f'{predicate.rstrip()}_uri'] = db_property['value']

            else:
                # there is no need to calculate the glove similarity
                question_dict[predicate.rstrip()] = 'NA'
                question_dict[f'{predicate.rstrip()}_uri'] = "NA"

        json_dict['questions'].append(question_dict)

with open('q1_predicate_dbproperty-glove.json', 'w') as f_json:
    json.dump(json_dict, f_json, indent=4)