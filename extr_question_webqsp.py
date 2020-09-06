import json
import argparse

arguments_parser = argparse.ArgumentParser() 

arguments_parser.add_argument("--webqsp_file", help="the json file containing questions for WebQSP")

args = arguments_parser.parse_args()
if __name__=='__main__':
    with open(args.webqsp_file, 'r') as f_read:
        file_write = open('temp_question_file.txt', 'w')
        lines = f_read.readlines()
        for line in lines:
            json_item = json.loads(line)
            question = json_item['question']
            file_write.write(question+'\n')
        file_write.close()
