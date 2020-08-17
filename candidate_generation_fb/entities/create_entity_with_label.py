import json
import subprocess
import multiprocessing as mp
import time
import re
import os

def mids_with_label(start_line, end_line, file_name):
    # first we are going to index entities from domain common.topic
    json_item = {"_index": "fbentityindex", "_type":"common.topic", "_source":{'uri':'', 'label': ''}}
    with open(file_name, 'r') as f_read_topic:
        lines = f_read_topic.readlines()
        chunk = lines[start_line:end_line]

    result= []
    # Preserve the triples with the ids, and get label from label-file 
    mid_uri = "http://rdf.freebase.com/ns"
    for line_id in chunk:
        # find the mid in the line_id
        line_id_split = line_id.split('\t')
        mid = line_id_split[0].strip()

        if re.match(r'</m.[a-zA-Z0-9_]+>', mid):
            label_files = ["fb-rdf-name-s02-c01", "fb-rdf-w3-label-s02-c01"]
            skip_label_file = False
            # take up name file first
            for label_file in label_files:
                fname_output = f'pipe_file{start_line}'
                fname_input = label_file
                p = subprocess.Popen(['gawk',
                        "{ fname" + '="'+fname_output+'";' +
                        'if( $1 == "'+mid+'" )' + " { print $0 >> fname; } }",
                        fname_input])
                p.communicate()
                try:
                    with open(f'pipe_file{start_line}', 'r') as pipe_file:
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
                                    result.append(json_object)
                                    skip_label_file = True
                                    break
                                if re.search(r'@en-[a-zA-Z]+$', name):
                                    span_en = re.search(r'@en-[a-zA-Z]+$', name).span()
                                    json_item['_source']['label'] = name[:span_en[0]].strip('""')
                                    mid_id = mid.strip('<>')
                                    mid = mid_uri + mid_id
                                    json_item['_source']['uri'] = mid
                                    json_object = json.dumps(json_item)
                                    result.append(json_object)
                                    skip_label_file = True
                                    break

                    os.remove(f'pipe_file{start_line}')
                    if skip_label_file:
                        # do not process the label file as the name has been found
                        #print('skipping label file')
                        break
                except Exception as e:
                    continue
    try:
        print(f'total mids in line range {start_line}-{end_line}: {len(result)} e.g. {result[0]}')
        return result 
    except IndexError as e:
        return ["\n"]



if __name__ == '__main__':
    file_with_mids ="fb-rdf-topic-s02-c01"  
    with open(file_with_mids, 'r') as f_read_mids:
        mids_count = len(f_read_mids.readlines())
        #split this file into chunks say 10000
    total_run = 10000
    chunks_count = 75
    run_size = int(mids_count/total_run)
    print(f'run_size: {run_size}')
    chunk_size = int(run_size/chunks_count)
    print(f'chunk_size: {chunk_size}')
    for run in range(total_run):
        chunk_spans = [(run*run_size+l*chunk_size, run*run_size+l*chunk_size+chunk_size) for l in range(chunks_count)]
        print(f'taking up chunk span: {chunk_spans}')
        pool = mp.Pool(mp.cpu_count())
        result_mids_with_label = [pool.apply_async(mids_with_label,
                                        args=(start_line, end_line, file_with_mids)) for start_line, end_line in chunk_spans]
        pool.close()
        pool.join()
        #start_line, end_line = chunk_spans[0]
        #start_line, end_line = 12018, 12050 
        #result = mids_with_label(start_line, end_line, file_with_mids)

        print(f'writing to file pll_entity_dump for run number: {run}')
        with open("pll_entity_dump.json", 'a') as f_write:
            for mids_from_chunk in result_mids_with_label:
                try:
                    [f_write.write(mid_label) for mid_label in mids_from_chunk.get()]
                except Exception as e:
                    print(f'error in run block: {e}')
                    print(e)
