filename= 'kr2ml_select.tsv'
with open(filename, 'r') as f_read:
    lines =f_read.readlines()

filename= '../nlqs_select.txt'
list_nlqs_with_pp = []
for line in lines:
    [list_nlqs_with_pp.append(nlq) for nlq in line.split('\t')[1:4]]

with open(filename, 'w') as f_write:
    f_write.writelines('\n'.join(list_nlqs_with_pp))

