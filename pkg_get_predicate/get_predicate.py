import pandas as pd
import numpy as np

from ast import literal_eval
from gensim.models import KeyedVectors
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from db_utils import get_dbpedia_predicates, dbpedia_property_vectorizer
from db_utils import get_property_using_cosine_similarity


if __name__ == "__main__":

    # # get all the dbpedia_predicates
    # get_dbpedia_predicates(refresh=True)

    # covert the predicates into prefix and label
    # get_dbpedia_predicates(refresh=False)

    # vectorize the dbpedia properties using glove embedding.
    # dbpedia_property_vectorizer()

    # test cosine_smilarity the function will return top-n property based on the cosine similarity
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')

    # # if embedding has changed, you may need to delete the numpy_property_vector.pkl file
    # vector = glove['palace'].reshape((1,-1))
    # print(get_property_using_cosine_similarity(vector))

    # Analysing Glove embedding for feq predicates taken from simple questions q1
    with open("scope_questions/q1_predicates.txt", 'r') as f:
        for predicate in f:
            word_count = 0
            for word in predicate.split():
                try:
                    if word_count == 0:
                        word_vector = glove[word].reshape(1, -1)
                        vector = np.array(word_vector)
                    else:
                        word_vector = glove[word].reshape(1, -1)
                        vector = np.append(vector, word_vector, axis=0)
                except KeyError as e:
                    pass
            predicate_emb = np.average(vector, axis=0)
            db_property = get_property_using_cosine_similarity(predicate_emb)
            # db_property['predicate'] = predicate
            with open("pkg_get_predicate/q1_predicate_dbproperty.txt", 'a') as f:
                f.write(f'{predicate.rstrip()}\t{db_property}\n')