import json
import subprocess
import time
import re
import os

with open("entity_dump.json", 'a') as f_write:
    # first we are going to index entities from domain common.topic
    json_item = {"_index": "fbentityindex", "_type":"common.topic", "_source":{'uri':'', 'label': ''}}

    # Preserve the triples with the ids, and get additional triples on description
    query_count = 0
    mid_uri = "http://rdf.freebase.com/ns"

    with open("fb-rdf-topic-s02-c01", 'r') as f_read_topic:
        while True:
            line_id = f_read_topic.readline()
            if not line_id:
                break

            # find the mid in the line_id
            line_id_split = line_id.split('\t')
            mid = line_id_split[0].strip()

            if re.match(r'</m.[a-zA-Z0-9_]+>', mid):
                label_files = ["fb-rdf-name-s02-c01", "fb-rdf-w3-label-s02-c01"]

                skip_label_file = False
                # take up name file first
                for label_file in label_files:
                    fname_output = 'pipe_file'
                    fname_input = label_file
                    p = subprocess.Popen(['gawk',
                            "{ fname" + '="'+fname_output+'";' +
                            'if( $1 == "'+mid+'" )' + " { print $0 >> fname; } }",
                            fname_input])
                    p.communicate()
                    try:
                        with open('pipe_file', 'r') as pipe_file:
                            name_lines = pipe_file.readlines()
                            if len(name_lines) >0:
                                for line in name_lines:
                                    name  = line.split('\t')[2]
                                    if re.search(r'@en$', name):
                                        span_en = re.search(r'@en$', name).span()
                                        json_item['_source']['label'] = name[:span_en[0]].strip('""')
                                        mid_id = mid.strip('<>')
                                        mid = mid_uri + mid_id
                                        json_item['_source']['uri'] = mid
                                        json_object = json.dumps(json_item)
                                        f_write.write(json_object+'\n')
                                        skip_label_file = True
                                        break
                                    if re.search(r'@en-[a-zA-Z]+$', name):
                                        span_en = re.search(r'@en-[a-zA-Z]+$', name).span()
                                        json_item['_source']['label'] = name[:span_en[0]].strip('""')
                                        mid_id = mid.strip('<>')
                                        mid = mid_uri + mid_id
                                        json_item['_source']['uri'] = mid
                                        json_object = json.dumps(json_item)
                                        f_write.write(json_object+'\n')
                                        skip_label_file = True
                                        break

                        os.remove('pipe_file')
                    except Exception as e:
                        pass

                    if skip_label_file:
                        # do not process the label file as the name has been found
                        break
                
                query_count += 1
                if query_count % 1000 == 0:
                    print(str(query_count) + "\t" + mid)
