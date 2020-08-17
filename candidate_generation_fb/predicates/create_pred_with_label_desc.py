import json
import subprocess
import time
import re
import os

#{"_index":"dbpropertyindex","_type":"doc","_id":"8rw9qGQBxnNHCvL3TLZl","_score":1,"_source":{"uri":"http://dbpedia.org/property/institution","label":"wont"}}
with open("predicate_dump.json", 'a') as f_write:
    # first we are going to index entities from domain common.topic
    prev_json_item = {"_index": "fbpropertyindex", "_type":"property", "_source":{'uri':'', 'label':'', 'description':''}}

    # Preserve the triples with the ids, and get additional triples on description
    query_count = 0
    mid_uri = "http://rdf.freebase.com/ns"

    with open('fb-rdf-pred-schema-properties-ids-1-byalpha-desc', 'r') as f_read_pred:
        mid_to_look_for = ''

        while True:
            line_label_desc  = f_read_pred.readline()
            if not line_label_desc:
                break

            # find the mid in the line_id
            mid_label_desc_split = line_label_desc.split('\t')
            mid = mid_label_desc_split[0].strip()
            if re.match(r'</m.[a-zA-Z0-9_]+>', mid):
                if mid == mid_to_look_for:
                    # description is found
                    mid_desc_split = line_label_desc.split('\t')
                    description = mid_desc_split[2]
                    if re.search(r'@en$', description):
                        span_en = re.search(r'@en$', description).span()
                        prev_json_item['_source']['description'] = description[:span_en[0]].strip('""')

                # a new mid is found
                else: 
                    # first write down the data of the old mid, prev_json_item, as a new mid has come
                    if not (prev_json_item['_source']['uri'] == ''): 
                        json_object = json.dumps(prev_json_item)
                        f_write.write(json_object+'\n')
                        # flush the old data in the json_item 
                        prev_json_item['_source']['uri'] = '' 
                        prev_json_item['_source']['label'] = ''
                        prev_json_item['_source']['description'] = ''

                    # processing the new mid
                    mid_to_look_for = mid
                    mid_id = mid.strip('<>')
                    mid = mid_uri + mid_id
                    prev_json_item['_source']['uri'] = mid 
                    label = mid_label_desc_split[2].strip()
                    label_split =label.split('/')
                    prev_json_item['_source']['label'] = label_split[-1]
