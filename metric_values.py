import json
import argparse
import sys

arguments_parser = argparse.ArgumentParser(
    prog='read execution result',
    description="Examine output file.")

arguments_parser.add_argument("--output_file", help="file with query output and query string")
arguments_parser.add_argument("--benchmark", help="benchmark dataset for question answering", choices=['qald-6', 'WebQSP'])

args = arguments_parser.parse_args()

if __name__=='__main__':
    if args.benchmark == 'WebQSP':
        test_file = 'dataset_WebQSP/data/WebQSP.test.json'
        webqsp_test_question_dict = {}

        """"Answers": [{
          "AnswerType": "Entity",
          "AnswerArgument": "m.01428y",
          "EntityName": "Jamaican English"
        },
        {
          "AnswerType": "Entity",
          "AnswerArgument": "m.04ygk0",
          "EntityName": "Jamaican Creole English Language"
        }]"""
        with open(test_file, 'r') as f_read:
            json_dict = json.load(f_read)
            for question_query_dict in json_dict["Questions"]:
                json_item = {'question':[], 'annotation':[], 'gold-query':[]}
                question = question_query_dict['RawQuestion'].strip()
                parses = question_query_dict['Parses']
                parse = parses[0]
                #query = parse['Sparql']
                answers = parse["Answers"]
                # create a dict of question and its correct answers as values 
                webqsp_test_question_dict[f'{question}'] = [answer['AnswerArgument'] for answer in answers]

        with open(args.output_file, 'r') as f_read:
            lines = f_read.readlines()
            total_count = 0
            correct_answer_count = 0
            for line in lines:
                json_item = json.loads(line)
                question = json_item["question"].strip()
                gold_answers_list = webqsp_test_question_dict[f'{question}']
                total_count +=1
                # find out if the two have common answers
                top_ranked_query = json_item["topk_queries"][0] 
                #for topk_query in json_item["topk_queries"]:
                predicted_answers = top_ranked_query["query_output"] 
                predicted_answers_list = []
                for output_dict in predicted_answers:
                    for bound_var, type_value_dict in output_dict.items():
                        for k3, v3 in type_value_dict.items():
                            if k3=="value":
                                predicted_answers_list.append(v3.split('/')[-1])

                # if gold_answers and predicated_answers_list have common answers
                gold_set = set(gold_answers_list)
                predicted_answers_set = set(predicted_answers_list)
                print('***********************************************************************')
                print(f'gold_answer: {gold_set}')
                print(f'predicted_answers_set: {predicted_answers_set}')
                if gold_set & predicted_answers_set:
                    print(f'!!!!!!!!!!Match!!!!!!!!!!')
                    correct_answer_count += 1
                else:
                    print(f'VVVVVVVVVVVVVVVVVVVV')
                accuracy_pct = correct_answer_count/total_count
            print(f'Accuracy: {accuracy_pct} ({correct_answer_count}/{total_count})')
