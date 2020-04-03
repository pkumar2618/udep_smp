import pandas as pd
import numpy as np
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import re
from urllib.parse import urlparse, urlunparse
import sys
from ast import literal_eval
from gensim.models import KeyedVectors
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def get_dbpedia_predicates(filename_raw="dbpedia_predicates.csv", filename_pretty="dbpedia_predicates_pretty.csv", namespaces="dbp_namespaces_prefix.csv", refresh=False):
    # pull up all the predicates from db_pedia when refresh is True, when refresh is False,
    # use the existing file dbpedia_predicate.csv

    if refresh:
        # pulling all the db-pedia properties
        query = """ SELECT * { ?x a rdf:Property }"""
        # dbpedia sparql endpoint
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # creating dataframe with obtained property values
        predicate_df = pd.DataFrame([result['x'] for result in results['results']['bindings']],
                                    columns=['type', 'value'])
        with open(filename_raw, 'w') as f_handle:
            # for result in results["results"]["bindings"]:
            #   print(result["type"]["value"])
            predicate_df.to_csv(f_handle, index=False)
    else:
        # create dictionary of url as key and prefix as value
        dict_ns_prefix_url = {}
        with open(namespaces, 'r') as f_handle:
            lines = f_handle.readlines()
            for line in lines[1:]:
                split_line = re.split(r'\s', line)
                dict_ns_prefix_url[split_line[0]] = split_line[1]

        # update the file for prefix value labels
        # splitting the property-value uri's into base name and value label
        predicate_df = pd.read_csv(filename_raw)
        predicate_df.set_index('value', inplace=True, drop=False)
        # predicate_df.set_index()
        # with open("test.csv", 'w') as f_handle:
        #   predicate_df.to_csv(f_handle,index=True)

        predicate_df['value_label'] = " "
        predicate_df['value_label_split'] = " "
        predicate_df['prefix'] = " "
        # value_label_df = pd.DataFrame(columns=['value_label'])
        # prefix_df = pd.DataFrame(columns=['prefix'])
        for url in predicate_df['value']:
            url_parsed = urlparse(url)
            if url_parsed.fragment is not '':
                value_label = url_parsed.fragment
                url_base = urlunparse((url_parsed.scheme, url_parsed.netloc, url_parsed.path, "", "", ""))
            else:
                url_path_split = url_parsed.path.split('/')
                value_label = url_path_split[-1]
                url_base = urlunparse((url_parsed.scheme, url_parsed.netloc, '/'.join(url_path_split[:-1]), "", "", ""))

            for ns_prefix, ns_url in dict_ns_prefix_url.items():
                # if re.search(rf'^{ns_url}[.]+', url):
                try:
                    if re.match(rf'^{url_base}[/#]?',ns_url):
                        predicate_df.loc[url, 'value_label'] = value_label
                        predicate_df.loc[url, 'value_label_split'] = split_camelcase_predicates(value_label)
                        predicate_df.loc[url, 'prefix'] = ns_prefix
                except:
                    e = sys.exc_info()[0]
                    print(f"Exception encountered: {e}")

        with open(filename_pretty, 'w') as f_handle:
            predicate_df = predicate_df.drop(columns=['type'])
            predicate_df = predicate_df[['prefix', 'value_label', 'value_label_split', 'value']]
            predicate_df.to_csv(f_handle, index=False)


def split_camelcase_predicates(cc_predicate):
    return re.findall(r'[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', cc_predicate)


def dbpedia_property_vectorizer():
    glove_loading_kv = KeyedVectors.load_word2vec_format("./glove/glove.6B.300d.w2vformat.txt")
    glove_loading_kv.save('./glove_gensim_mmap')
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')
    vector = glove['stuff']
    emb_dim = vector.shape[0]
    with open("dbpedia_predicates_pretty.csv", 'r') as f:
        dbp_df_pretty = pd.read_csv(f)

    dbp_property_avgvector_prefix = dbp_df_pretty[['value_label', 'value_label_split', 'prefix', 'value']]
    dbp_property_avgvector_prefix['avg_vector'] = " "
    dbp_property_avgvector_prefix.set_index('value', drop=False)

    for index in dbp_property_avgvector_prefix.index.to_list():
        property_words = dbp_property_avgvector_prefix.loc[index, 'value_label_split']
        try:
            property_words = literal_eval(property_words)
            vector_list = []
            for word in property_words:
                try:
                    vector = glove[word]
                    vector_list.append(vector)
                except Exception as ek:
                    print(f"word: {word} not found a OOV error")
        except Exception as ev:
            print(f"ast returned error: {ev} for list {property_words}")
            # pass

        # vector_list = np.append([vector_list], axis=1)
        # avg_vector = sum(vector_list)/len(words)
        sum_vector = np.sum(vector_list, axis=0)
        avg_vector = np.divide(sum_vector, len(vector_list))
        try:
            assert avg_vector.shape[0] == emb_dim
        except Exception as e:
            print(f"{e} occurred ")
        dbp_property_avgvector_prefix.loc[index, 'avg_vector'] = avg_vector


    with open("dbpedia_property_avg_vector_prefix_uri.pkl", 'wb') as f:
        dbp_property_avgvector_prefix = dbp_property_avgvector_prefix[['value_label','avg_vector','prefix', 'value']]
        # dbp_property_avgvector_prefix.to_csv(f, index=False)
        pickle.dump(dbp_property_avgvector_prefix, f)

def get_property_using_cosine_similarity(vector, pd_property_vector="dbpedia_property_avg_vector_prefix_uri.pkl",
                                         recalculate_numpy_property_vector=False, top_n=1):
        # gives similarity between a vector and list of other vectors
        if recalculate_numpy_property_vector:
            with open(pd_property_vector, 'rb') as f:
                property_vector_df = pickle.load(f)
            # emb_dim = vector.shape[1]
            property_vector_df['avg_vector'] = property_vector_df['avg_vector'].fillna(np.nan)# pd.Series(np.zeros(vector.shape[1])))

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
                            np_property_vector = row[2].reshape(1,-1)
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
    # This will pull up all the predicates.
    # get_dbpedia_predicates(filename_raw="dbpedia_predicates.csv", filename_pretty="dbpedia_predicates_pretty.csv",
    #                        namespaces="dbp_namespaces_prefix.tsv", refresh=True)
    # this is required to create pretty looking dbpedia predicates, with their prefixes, and camel case
    # separated.

    # get_dbpedia_predicates(filename_raw="dbpedia_predicates.csv", filename_pretty="dbpedia_predicates_pretty.csv",
    #                        namespaces="dbp_namespaces_prefix.tsv", refresh=False)

    # print(split_camelcase_predicates("ateneoWChess"))
    # print(split_camelcase_predicates("CamelCaseXYZ"))
    # print(split_camelcase_predicates("XYZCamelCase"))

    # # Create a dataframe with vector representation of property values
    # dbpedia_property_vectorizer()

    # # test cosine_smilarity the function will return top-n property based on the cosine similarity
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')

    # # you may need to run this once to get numpy array of property_vector
    # vector = glove['stuff'].reshape((1,-1))
    # get_property_using_cosine_similarity(vector, recalculate_numpy_property_vector=True)

    # # further run don't require refresh, as the numpy 2d array of property_value is loaded from the
    # # first run with recaclulate_numpy_property_vector=True
    # vector = glove['location'].reshape(1,-1)
    # print(get_property_using_cosine_similarity(vector, recalculate_numpy_property_vector=False))
    
    # Analysing Glove embedding for feq predicates taken from simple questions q1
    with open("./analysis/q1_predicates.txt", 'r') as f:
        predicates = f.readlines()

    # vector = np.array()
    for predicate in predicates:
        word_count = 0
        for word in predicate.split():
            try:
                if word_count ==0:
                    word_vector = glove[word].reshape(1,-1)
                    vector = np.array(word_vector)
                else:
                    word_vector = glove[word].reshape(1, -1)
                    vector = np.append(vector, word_vector, axis=0)
            except KeyError as e:
                pass
        predicate_emb = np.average(vector, axis=0)
        db_property = get_property_using_cosine_similarity(predicate_emb, recalculate_numpy_property_vector=False)
        # db_property['predicate'] = predicate
        with open("./analysis/q1_predicate_dbproperty.txt", 'a') as f:
            f.write(f'{predicate.rstrip()}\t{db_property}\n')