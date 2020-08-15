import pandas as pd

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
        #if line_split[0] not  in ['INFO', 'DEBUG']:
        #    if next_line_write_flag == True:
                #f_write.write(lines[i])
        #        continue

        if line_split[0] == 'INFO':
            if line_split[2] in columns_dict.keys():
                if line_split[2] == 'g-SPARQL':
                    columns_dict[line_split[2]].append(line_split[3])
                else:
                    columns_dict[line_split[2]].append(line_split[3])

                #f_write.write(lines[i])
                #next_line_write_flag = False

                ##check if the next line is also part of the query
                #if line_split[2] in ["ug_sparql", "g-SPARQL"]:
                #    try:
                #        next_line = lines[i+1].split(':')
                #        if next_line[0] not  in ['INFO', 'DEBUG']:
                #            next_line_write_flag = True
                #        else:
                #            next_line_write_flag = False
                #    except:
                #        continue
    df_table = pd.DataFrame([columns_dict])
    df_table.to_csv(file_name, sep='\t', index=False)

