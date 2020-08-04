#!/usr/bin/env bash
cp dep_parse.txt /home/pawan/projects/aihn_qa/UDepLambda/dep_parse.txt
cd /home/pawan/projects/aihn_qa/UDepLambda/
cat /home/pawan/projects/aihn_qa/UDepLambda/dep_parse.txt| /home/pawan/projects/aihn_qa/UDepLambda/run_LambdaToSQG.sh
cp ug_graph.txt /home/pawan/projects/aihn_qa/udep_smp/ug_graph.txt
