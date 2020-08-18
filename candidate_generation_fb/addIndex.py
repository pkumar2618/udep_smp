"""
modified file, original was taken from 
@author: Sakor
"""
from elasticsearch import Elasticsearch
import json
import re
from multiprocessing.pool import ThreadPool



docType = "doc"
# by default we connect to localhost:9200
es = Elasticsearch(['http://10.208.20.61:9200'])
path_to_data = '/app/'
# path_to_data = '../'

def addToIndexThread(line):
    if not re.match(r'\s*', line):
        try:
            lineObject=json.loads(line, strict=False)
            return addToIndex(lineObject["_source"]["uri"],lineObject["_source"]["label"])
        except:
            raise
            print ("error")
            return 'error'

def addToIndex(uri,label):
    try:
        es.index(index=indexName, doc_type=docType, body={"uri":uri, "label":label})
        #print (label)
        return True
    except:
        return 'error'

def propertyIndexAdd():
    global indexName
    indexName= "fbpropertyindex"
    with open('./predicates/predicate_dump.json',encoding="utf8") as f:
        lines = f.readlines()
        pool = ThreadPool(10)
        pool.map(addToIndexThread, lines)
        pool.close()
        pool.join()

def OntologyIndexAdd():
    global indexName
    indexName= "fbontologyindex"
    with open('./ontology/ontology_dump.json',encoding="utf8") as f:
        lines = f.readlines()
        pool = ThreadPool(10)
        pool.map(addToIndexThread, lines)
        pool.close()
        pool.join()

def entitiesIndexAdd():
    global indexName
    indexName = "fbentityindex"
    with open('./entities/entity_dump_part1.json',encoding="utf8") as f:
        lines = f.readlines()
        pool = ThreadPool(10)
        pool.map(addToIndexThread, lines)
        pool.close()
        pool.join()

def classesIndexAdd():
    global indexName
    indexName = "fbclassindex"
    with open('./elastic_dump/dbclassindex.json') as f:
        lines = f.readlines()
        pool = ThreadPool(12)
        pool.map(addToIndexThread, lines)
        pool.close()
        pool.join()

if __name__ == '__main__':
    entitiesIndexAdd()
    #propertyIndexAdd()
    #OntologyIndexAdd()
