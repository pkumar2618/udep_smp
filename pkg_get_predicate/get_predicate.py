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
    vector = glove['palace'].reshape((1,-1))
    print(get_property_using_cosine_similarity(vector))