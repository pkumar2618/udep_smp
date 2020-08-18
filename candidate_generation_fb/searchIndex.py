from elasticsearch import Elasticsearch


es = Elasticsearch(['http://10.208.20.61:9200'])
docType = "doc"



def entitySearch(query):
    indexName = "fbentityindex"
    results=[]
    ###################################################
    elasticResults=es.search(index=indexName, doc_type=docType, body={
              "query": {
                "match" : { "label" : query } 
              }
               ,"size":10})

    for result in elasticResults['hits']['hits']:
            results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*40,0])
    ###################################################
    elasticResults=es.search(index=indexName, doc_type=docType, body={
              "query": {
                "fuzzy" : { "label" : query  } 
              }
               ,"size":5})

    for result in elasticResults['hits']['hits']:
            results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*25,0])
    ###################################################
    elasticResults=es.search(index=indexName, doc_type=docType, body={
            "query": {
        "bool": {
            "must": {
                "bool" : { "should": [
                      { "multi_match": { "query": query , "fields": ["label"]  }}
                       ] }
            }
        }
    }
            ,"size":10})
    for result in elasticResults['hits']['hits']:
        results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*2,0])
    ###################################################
    return results
    #for result in results['hits']['hits']:
        #print (result["_score"])
        #print (result["_source"])
        #print("-----------")

def ontologySearch(query):
    indexName = "fbontologyindex"
    results=[]
    ###################################################
    ###################################################
    ###################################################
    elasticResults=es.search(index=indexName, doc_type=docType, body={
              "query": {
                "match" : { "label" : query } 
              }
               ,"size":10
    }
           )
    for result in elasticResults['hits']['hits']:
        results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*40,0])
    ###################################################
    elasticResults=es.search(index=indexName, doc_type=docType, body={
              "query": {
                "fuzzy" : { "label" : query  } 
              }
               ,"size":5
    }
           )
    for result in elasticResults['hits']['hits']:
            results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*25,0])
    return results
    #for result in results['hits']['hits']:
        #print (result["_score"])
        #print (result["_source"])
        #print("-----------")
        
def classSearch(query):
    indexName = "fbclassindex"
    results=[]
    elasticResults=es.search(index=indexName, doc_type=docType, body={
            "query": {
        "bool": {
            "must": {
                "bool" : { "should": [
                      { "multi_match": { "query": query , "fields": ["label"] , "fuzziness": "AUTO" }}
                ]}
            }
        }
    }
            ,"size":5
    })
    #print(elasticResults)
    for result in elasticResults['hits']['hits']:
            results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"],0])
    return results
    #for result in results['hits']['hits']:
        #print (result["_score"])
        #print (result["_source"])
        #print("-----------")

def propertySearch(query):
    indexName = "fbpropertyindex"
    results=[]
    elasticResults=es.search(index=indexName, doc_type=docType, body={
            "query": {
        "bool": {
            "must": {
                "bool" : { "should": [
                      { "multi_match": { "query": query , "fields": ["label"]  }}
                       ] }
            }
        }
    }
            ,"size":10})
    for result in elasticResults['hits']['hits']:
        results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*2,0])
    return results
    #for result in results['hits']['hits']:
        #print (result["_score"])
        #print (result["_source"])
        #print("-----------")


if __name__=="__main__":
    results_of_entity =entitySearch('Port Mungo')
    print(results_of_entity)
    result_relation = propertySearch('score')
    print(result_relation)
