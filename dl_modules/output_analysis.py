import json
import numpy as np
import operator # for sorting the dictionary by value


with open("output_prediction.json", 'r') as f_read:
    json_list_dict = json.load(f_read)
    next_block_starts = 0
    target_score_list=[]
    cross_embedding_score_list=[]
    accurate = 0
    total_block = 0
    mrr_list = []
    hit_at = {"3":[0, 0], "5":[0, 0], "10":[0, 0], "20":[0, 0]}
    for dict_item in json_list_dict:
        if next_block_starts <2:
            next_block_starts += dict_item["target_score"]
            if next_block_starts == 2:
                # we should work on the block formed so far that makes for one query result.
                # once all the parameters have been calculated we should refresh the query block
                # and build it again.
                # data collected so far makes for a block
                target_score_block = np.array(target_score_list)
                idx_target = np.argmax(target_score_block)
                label_score_block = np.array(cross_embedding_score_list)
                idx_cemb = np.argmax(label_score_block)
                # calculating mrr
                mrr_list.append(1 / (1 + idx_cemb))
                #### Calculating Hit Rate
                hit_dict = {}
                for i, value in enumerate(label_score_block):
                    hit_dict[f'{i}'] = value

                # short hit_dict based on value:
                sorted_hit_dict_by_value = sorted(hit_dict.items(), key=operator.itemgetter(1), reverse=True)
                # hit at three, five ...
                for at_idx in [3, 5, 10, 20]:
                    if len(sorted_hit_dict_by_value) >= at_idx:
                        ranked_slice = sorted_hit_dict_by_value[:at_idx]
                        # the correct output should return at index 0, the key is '0'
                        for k, v in ranked_slice:
                            if k == '0':
                                hit_at[f'{at_idx}'][0] += 1
                        # if no hit happens, we still need to count the block
                        hit_at[f'{at_idx}'][1] += 1

                #### calculating accuracy
                if idx_target == idx_cemb:
                    accurate += 1
                    total_block += 1
                else:
                    total_block += 1

                # refresh the target_score_list for new block of samples
                target_score_list = [dict_item["target_score"]]
                cross_embedding_score_list = [dict_item["cross_emb_score"]]
                next_block_starts = 1

            else:
                target_score_list.append(dict_item["target_score"])
                cross_embedding_score_list.append(dict_item["cross_emb_score"])


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


