import json
import numpy as np
import operator # for sorting the dictionary by value


with open("output_prediction.json", 'r') as f_read:
    json_list_dict = json.load(f_read)
    total_block = 0
    accurate = 0
    mrr_list = []
    hit_at = {"1":[0, 0],"3":[0, 0], "5":[0, 0], "10":[0, 0], "15":[0, 0], "20":[0, 0]}
    # create a block with question as key and dict_items as list of values 
    blocks = {} 
    for dict_item in json_list_dict:
        # we should work on the block formed so far that makes for one query result.
        # once all the parameters have been calculated we should refresh the query block
        # and build it again.
        # data collected so far makes for a block
        if dict_item:
            try: 
                question = dict_item['question']
                blocks[f'{question}'].append(dict_item)
            except KeyError as e:
                blocks[f'{question}'] = [dict_item]
        else: # an empty dict is encountered. 
            continue

    for question, block in blocks.items():
        candidates = []
        for dict_item in block:
            #candidate = {'target_score':'', 'cross_emb_score':''}
            #candidate['target_score'] =dict_item["target_score"]
            #candidate['cross_emb_score'] =dict_item["cross_emb_score"]
            candidates.append(dict_item) 

        candidates_sorted = sorted(candidates, key=lambda x: x['cross_emb_score'], reverse=True)

        # hit at three, five ...
        for rank, dict_item in enumerate(candidates_sorted):
            if dict_item['target_score'] == 1:
                if rank == 0: 
                    accurate +=1
                mrr_list.append(1 / (1 + rank))
                for at_idx in [20,15,10,5,3,1]:
                    if rank < at_idx:
                        hit_at[f'{at_idx}'][0] += 1
                break

        # the denominator count will go by one for each block
        for at_idx in [20,15,10,5,3,1]:
                hit_at[f'{at_idx}'][1] += 1

        total_block += 1

    perc_accu = (accurate/total_block)*100
    mrr = np.average(np.array(mrr_list))

    print(f"Accuracy: {accurate}/{total_block} which is {perc_accu}%" )
    print(f"MRR: {mrr}")
    for k, v in hit_at.items():
        try:
            hit_rate = v[0]/v[1]
        except ZeroDivisionError:
            hit_rate ="__"
        print(f'hit@{k} : {v[0]}/{v[1]}, {hit_rate}')

