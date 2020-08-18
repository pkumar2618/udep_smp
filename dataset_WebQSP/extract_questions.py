import json
with open("WebQSP_processed.test.json", 'r') as f_read:
    json_list = json.load(f_read)

with open('webqsp_test_nlq', 'w') as f_write:
    for json_dict in json_list:
        f_write.write(json_dict['question']+'\n')

