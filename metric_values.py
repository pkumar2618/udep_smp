import json
import argparse
import sys
import numpy as np

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
                
        # for the set of questio we are goint to store 1 if the answer exists in the possible queries generated and 0 if no answer found in
        # all the 50 possibel queries generated by the system. 
        correct_ans_found = []
        # we are also goin 
        #hit_at = {"3":[0, 0], "5":[0, 0], "10":[0, 0], "20":[0, 0]}

        # lets define the mrr as reciprocal rank of the query which gets us the right answer for a given question
        mrr_list = []
        # will go through the question in the output file one by one 
        with open(args.output_file, 'r') as f_read:
            lines = f_read.readlines()
            total_count = 0
            correct_answer_count = 0
            hit_at = {"1":[0,0], "3":[0, 0], "5":[0, 0], "10":[0, 0], "20":[0, 0], "50":[0, 0]}
            for line in lines:
                json_item = json.loads(line)
                question = json_item["question"].strip()
                gold_answers_list = webqsp_test_question_dict[f'{question}']
                total_count +=1
                # for every question we start with answer found flag as False, 
                # as we go over all the possible queries for this question we will change this flag True
                answer_found = False
                hit_list=[0 for p_at in hit_at.keys()]
                # iterating through the list of generated queries for a question
                # for mrr we will evaluate all the queries and find out if there was any correct answer,
                # in there are many queries that got as right answer, the one with highest reciprocal rank will go into the 
                # calculation for mrr for the set of question we will have. 
                query_rr = [] 
                for i, query in enumerate(json_item["topk_queries"]):
                    # find out if the two have common answers
                    predicted_answers = query["query_output"] 
                    predicted_answers_list = []
                    for output_dict in predicted_answers:
                        for bound_var, type_value_dict in output_dict.items():
                            for k3, v3 in type_value_dict.items():
                                if k3=="value":
                                    predicted_answers_list.append(v3.split('/')[-1])

                    # if gold_answers and predicated_answers_list have common answers
                    gold_set = set(gold_answers_list)
                    predicted_answers_set = set(predicted_answers_list)
                    if gold_set & predicted_answers_set:
                        #print('***********************************************************************')
                        #print(f'gold_answer: {gold_set}')
                        #print(f'predicted_answers_set: {predicted_answers_set}')
                        #print(f'!!!!!!!!!!Match!!!!!!!!!!')
                        answer_found = True
                        query_rr.append(1/(1+i))
                        # if answer is found we need to update the hit rate, 
                        for at_idx, hitat in enumerate(hit_at.keys()):
                            if i < int(hitat):# the keys are 1, 3, 5.., and idexes take value 0,1,2 therefore less than
                                hit_list[at_idx]=1
                    else:
                        #print('***********************************************************************')
                        #print(f'gold_answer: {gold_set}')
                        #print(f'predicted_answers_set: {predicted_answers_set}')
                        #print(f'VVVVVVVVVVVVVVVVVVVV')
                        query_rr.append(0)
                        continue
                max_query_rr = max(query_rr)
                mrr_list.append(max_query_rr)
                # hit_at update
                for at_idx, hitat in enumerate(hit_at.keys()):
                    hit_at[f'{hitat}'][0] += hit_list[at_idx]
                    hit_at[f'{hitat}'][1] += 1 # irrespective we are going to add 1 to keep the count of questions processed. 

                if answer_found:
                    correct_ans_found.append(1)
                else:
                    correct_ans_found.append(0)

            correct_answer_count =len([1 for hit in correct_ans_found if hit==1])
            total_count = len(correct_ans_found)
            accuracy_pct =correct_answer_count/total_count
            mrr = np.average(mrr_list) 
            print(f'Accuracy: {accuracy_pct} ({correct_answer_count}/{total_count})')
            print(f"MRR: {mrr}")
            for k, v in hit_at.items():
                hit_rate = v[0]/v[1]
                print(f'hit@{k} : {v[0]}/{v[1]}, {hit_rate}')