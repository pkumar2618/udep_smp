# Semantic Parsing
## Intermediate From
### Seq to Seq Model
# Running the parser with a question file
python -m pdb main_line.py --questions_file nlqs_select_2spo.txt
python -m pdb main_line.py --questions_file nlqs_select_2spo.txt --disambiguator elasticsearch --knowledge_graph dbpedia
python -m main_line --questions_file nlqs_select_2spo.txt --disambiguator elasticsearch --knowledge_graph dbpedia --log info
python -m main_line --questions_file diana.txt  --disambiguator elasticsearch --knowledge_graph dbpedia --log debug --logname diana
python -m main_line --questions_file nlqs_kr2ml.txt --batch 3 --disambiguator elasticsearch --knowledge_graph dbpedia --log debug --logname kr2ml
python -m pdb main_line --questions_file nlqs.txt  --disambiguator elasticsearch --knowledge_graph dbpedia --log debug --logname testing
python -m pudb.run main_line.py --questions_file nlqs.txt  --disambiguator elasticsearch --knowledge_graph dbpedia --log debug --logname einstein
## Logical Form
### lambda-DCS
### SPARQL
