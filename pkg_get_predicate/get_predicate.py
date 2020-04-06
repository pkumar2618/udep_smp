import pandas as pd
import numpy as np

from ast import literal_eval
from gensim.models import KeyedVectors
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def get_property_using_cosine_similarity(vector, pd_property_vector="dbpedia_property_avg_vector_prefix_uri.pkl",
                                         recalculate_numpy_property_vector=False, top_n=1):
    # gives similarity between a vector and list of other vectors
    if recalculate_numpy_property_vector:
        with open(pd_property_vector, 'rb') as f:
            property_vector_df = pickle.load(f)
        # emb_dim = vector.shape[1]
        property_vector_df['avg_vector'] = property_vector_df['avg_vector'].fillna(
            np.nan)  # pd.Series(np.zeros(vector.shape[1])))

        np_property_vector = np.array([])
        count = 1
        for row in property_vector_df.itertuples():
            try:
                if row[2].any() == np.nan:
                    row_2 = np.zeros(vector.shape)
                    if count == 1:
                        np_property_vector = row_2
                        count += 1
                    else:
                        np_property_vector = np.concatenate((np_property_vector, row_2), axis=0)
                        count += 1
                else:
                    if count == 1:
                        np_property_vector = row[2].reshape(1, -1)
                        count += 1
                    else:
                        np_property_vector = np.concatenate((np_property_vector, row[2].reshape(1, -1)), axis=0)
                        count += 1

            except AttributeError as e:
                print(e)
                if np.isnan(row[2]):
                    row_2 = np.zeros(vector.shape)
                    if count == 1:
                        np_property_vector = row_2
                        count += 1
                    else:
                        np_property_vector = np.concatenate((np_property_vector, row_2), axis=0)
                        count += 1

        # save the np_property_vector matrix for repeated use
        with open("numpy_property_vector.pkl", 'wb') as f:
            pickle.dump(np_property_vector, f)

    elif not recalculate_numpy_property_vector:
        # loading the numpy_property_vector
        with open("numpy_property_vector.pkl", 'rb') as f:
            np_property_vector = pickle.load(f)

        # load dataframe containing uri's as well
        with open(pd_property_vector, 'rb') as f:
            property_vector_df = pickle.load(f)

    # calculate cosine similarity and return top-n properties
    res = cosine_similarity(np.array(vector).reshape((1, -1)), np_property_vector)
    index_topn = np.argmax(res)
    res_property_prefix_uri = property_vector_df.loc[index_topn, ['value_label', 'prefix', 'value']]
    return res_property_prefix_uri.to_dict()


if __name__ == "__main__":

    # # test cosine_smilarity the function will return top-n property based on the cosine similarity
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')

    # # you may need to run this once to get numpy array of property_vector
    vector = glove['stuff'].reshape((1,-1))
    # get_property_using_cosine_similarity(vector, recalculate_numpy_property_vector=True)

    # # further run don't require refresh, as the numpy 2d array of property_value is loaded from the
    # # first run with recaclulate_numpy_property_vector=True
    # vector = glove['location'].reshape(1,-1)
    print(get_property_using_cosine_similarity(vector, recalculate_numpy_property_vector=False))
