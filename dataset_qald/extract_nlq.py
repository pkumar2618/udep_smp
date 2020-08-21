
import json
import argparse
import sys

arguments_parser = argparse.ArgumentParser(
    prog='read execution result',
    description="Examine output file.")

arguments_parser.add_argument("--read_from", help="file with combined question and query")
arguments_parser.add_argument("--write_to", help="file with just the question ")
args = arguments_parser.parse_args()

if __name__=='__main__':
    with open(args.read_from, 'r') as f_read:
        lines = f_read.readlines()
        for line in lines:
            line_split = line.split(',')
            query_str = line_split[4]
            question = line_split[3]
            with open(args.write_to, 'a') as f_write:
                f_write.write(question +'\n')
