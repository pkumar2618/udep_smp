import json
import argparse
import sys

arguments_parser = argparse.ArgumentParser(
    prog='read execution result',
    description="Examine output file.")

arguments_parser.add_argument("--output_file", help="file with query output and query string")
arguments_parser.add_argument("--stop_at", help="select linker for disambiguation among spotlight and "
                                                      "custom", choices=['query_output'])
args = arguments_parser.parse_args()

if __name__=='__main__':
    with open(args.output_file, 'r') as f_read:
        lines = f_read.readlines()
        for line in lines:
            json_item = json.loads(line)
            #if args.stop_at=="query_output":
            print(f'************************************************************************************************')
            print(f'question: {json_item["question"]}')
            for topk_query in json_item["topk_queries"]:
                print(f'---------------------------------------------------- ')
                print(f'query_string: {topk_query["query_string"]}')
                print(f'query_output: .........................................')
                for output in topk_query["query_output"]:
                    print(output)
