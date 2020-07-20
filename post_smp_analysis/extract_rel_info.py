extract_info_about = ['udep_lib.nlquestion', 'udep_lib.ug_logicalform', 'udep_lib.ug_sparql_graph']
# the last ug_sparql_graph corresponds to the ug-sparql-query, three appearances of spo-tripe and the g-sparql-query. 

with open('../kr2ml.log', 'r') as f_read:
    f_write = open('pp_graph_structures.txt', 'w')
    lines = f_read.readlines() 
    next_line_write_flag = True
    for i in range(len(lines)):
        line_split = lines[i].split(':')
        if line_split[0] not  in ['INFO', 'DEBUG']:
            if next_line_write_flag == True:
                f_write.write(lines[i])

        elif line_split[0] == 'INFO':
            if line_split[1] in extract_info_about:
                f_write.write(lines[i])
                next_line_write_flag = False

                #check if the next line is also part of the query
                if line_split[2] in ["ug_sparql", "g-SPARQL"]:
                    try:
                        next_line = lines[i+1].split(':')
                        if next_line[0] not  in ['INFO', 'DEBUG']:
                            next_line_write_flag = True
                        else:
                            next_line_write_flag = False
                    except:
                        continue

    f_write.close()

