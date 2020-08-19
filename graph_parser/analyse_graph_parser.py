import json
import subprocess
def direct_to_udeplambda(sentence=None):
    if sentence is None:
        with open("./gp_sentence.txt", 'w') as f:
            f.write(f'{{"sentence":"{self.nlq_canonical.strip()}"}}')
    else:
        with open("./gp_sentence.txt", 'w') as f:
            f.write(f'{{"sentence":"{sentence}"}}')

    res = subprocess.check_output("./run_graph_parser.sh")

    # convert the bytecode into dictionary.
    output = json.loads(res.decode('utf-8'))
    return output

if __name__=='__main__':
    # running example sentences in siva_reddy graph-parser
    json_list = []
    with open('/home/pawan/projects/aihn_qa/graph-parser/input.txt', 'r') as f_handle:
        lines = f_handle.readlines()
        for line in lines:
            with open("./gp_sentence.txt", 'w') as f:
                f.write(line)

            res = subprocess.check_output("/home/pawan/projects/aihn_qa/udep_smp/run_graph_parser.sh")
            # convert the bytecode into dictionary.
            output = res.decode('utf-8').split('\n')
            json_item = {}
            json_item['question'] = f'{output[0]}\n{output[1]}'
            for idx, parse in enumerate(output[2:]):
                json_item[f'parse_{idx}'] = parse.split(':', 1)
            json_list.append(json_item)

        with open('gp_output.json', 'w') as f_handle:
            json.dump(json_list, f_handle, indent=4)
