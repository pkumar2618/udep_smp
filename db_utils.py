import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import re
from urllib.parse import urlparse, urlunparse
import sys

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

        predicate_df['value_label']=" "
        predicate_df['value_label_split']=" "
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

if __name__ == "__main__":
    # get_dbpedia_predicates(filename_raw="dbpedia_predicates.csv", filename_pretty="dbpedia_predicates_pretty.csv",
    #                        namespaces="dbp_namespaces_prefix.tsv", refresh=True)
    get_dbpedia_predicates(filename_raw="dbpedia_predicates.csv", filename_pretty="dbpedia_predicates_pretty.csv",
                           namespaces="dbp_namespaces_prefix.tsv", refresh=False)

    # print(split_camelcase_predicates("ateneoWChess"))
    # print(split_camelcase_predicates("CamelCaseXYZ"))
    # print(split_camelcase_predicates("XYZCamelCase"))