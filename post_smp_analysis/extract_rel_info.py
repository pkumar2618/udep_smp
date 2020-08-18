import pandas as pd
import ast

extract_info_about = ['udep_lib.nlquestion', 'udep_lib.ug_logicalform:ug_gp_graphs:', 'udep_lib.ug_sparql_graph:ug_sparql', 'udep_lib.ug_sparql_graph:g-SPARQL']
columns_dict = {'question':[], 'ug_gp_graphs':[], 'ug_sparql':[], 'g-SPARQL':[]}

#extract_info_about = ['udep_lib.ug_sparql_graph']
# the last ug_sparql_graph corresponds to the ug-sparql-query, three appearances of spo-tripe and the g-sparql-query. 

with open('../run_02_qald6.log', 'r') as f_read:
    file_name= 'pp_run_02_qald6.txt'
    lines = f_read.readlines() 
    #next_line_write_flag = True
    for i in range(len(lines)):
        line_split = lines[i].split(':', 3)

        if line_split[0] == 'INFO':
            if line_split[2] in columns_dict.keys():
                if line_split[2] == 'g-SPARQL':
                    topk = line_split[3].strip().strip('[]').split(',')[0]
                    # columns_dict[line_split[2]].append(ast.literal_eval(line_split[3])[:1])
                    columns_dict[line_split[2]].append(topk)
                else:
                    columns_dict[line_split[2]].append(line_split[3].strip())

    df_table = pd.DataFrame(columns_dict)
    df_table.to_csv(file_name, sep=',', index=False)

