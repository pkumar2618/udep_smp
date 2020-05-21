import pandas as pd

from udeplib.nlqcanonical import NLQCanonical

file_input= 'kr2ml_select.tsv'
file_output= '../question_sparql_pairs.tsv'
question_sparql_pair= pd.DataFrame(columns=['question_id', 'gold_sparql',
                                            'para0', 'para0_deplambda', 'para0_sparql',
                                            'para1', 'para1_deplambda', 'para1_sparql',
                                            'para2', 'para2_deplambda', 'para2_sparql'])
with open(file_input, 'r') as f_read:
    for line in f_read:
        items = line.split('\t')
        sentences = items[1:4]
        question_id = items[0]
        gold_sparql = items[4]
        pd_row_dict = {'question_id': question_id, 'gold_sparql':gold_sparql,
                       'para0': sentences[0], 'para0_deplambda':None, 'para0_sparql':None,
                       'para1': sentences[1], 'para1_deplambda':None, 'para1_sparql':None,
                       'para2': sentences[2], 'para2_deplambda':None, 'para2_sparql':None}
        count = 0
        for sentence in sentences:
            nlq_canon = NLQCanonical(sentence)
            ug_logical_form = nlq_canon.direct_to_udeplambda(sentence)
            pd_row_dict[f'para{count}_deplambda'] = ug_logical_form.udep_lambda['dependency_lambda'][0]
            ug_sparql_graph = ug_logical_form.translate_to_sparql(kg='dbpedia')
            sparql_query = ug_sparql_graph.get_ug_sparql_query_string()
            pd_row_dict[f'para{count}_sparql'] = sparql_query
            count += 1

        question_sparql_pair = question_sparql_pair.append(pd_row_dict, ignore_index=True, verify_integrity=True)

with open(file_output, 'w') as f_write:
    question_sparql_pair.to_csv(f_write, sep='\t')

