import json
import argparse
import sys
import numpy as np
import os

arguments_parser = argparse.ArgumentParser(
    prog='split the question into pass fail for analysis',
    description="Find out question numbers which are failing.")

arguments_parser.add_argument("--question_file_original", help="file containing original questions")
arguments_parser.add_argument("--questions_pass_fail", help="file containing json dict of passing questions number and failing questions.")

args = arguments_parser.parse_args()

if __name__=='__main__':
    with open(args.questions_pass_fail, 'r') as f_read:
        json_item = json.loads(f_read.readline())
        pass_qn = json_item['pass']
        fail_qn = json_item['fail']

    with open(args.question_file_original, 'r') as f_read:
        lines = f_read.readlines()
        file_pass = open(f'split_pass_{args.question_file_original}', 'w+')
        file_fail = open(f'split_fail_{args.question_file_original}', 'w+')

        for question_numb, line in enumerate(lines):
            if question_numb in pass_qn:
                file_pass.write(line)
            elif question_numb in fail_qn:
                file_fail.write(line)
        
        file_fail.close()
        file_pass.close()


