
q2 = """
what is the population of Australia capital ?
"""

ug_q2 = """
SELECT DISTINCT ?0:x
  9 WHERE {?0:x <what> ?3:x . ?3:x <population> <Australia> . ?3:x a <population> . ?5:m.Australia a <capital>}
"""

g_q2 = """
SELECT DISTINCT ?0:x
WHERE {?3:x <http://dbpedia.org/property/populationM> <http://dbpedia.org/resource/Australia> . 
    ?0:x <http://dbpedia.org/property/situation> ?3:x .
 ?3:x a <http://dbpedia.org/resource/Population> .
  ?5:m.Australia a <http://dbpedia.org/resource/Capital>    }
"""


<http://dbpedia.org/resource/Australia> <http://dbpedia.org/resource/Capital> ?x
res: Australia
dbo: capital ?x

missed_g_q2 = """
SELECT DISTINCT ?0:x
WHERE {?3:x <http://dbpedia.org/property/populationM> <http://dbpedia.org/resource/Australia> . 
    ?0:x <http://dbpedia.org/property/situation> ?3:x .
 ?3:x a <http://dbpedia.org/resource/Population> .
  ?5:m.Australia a <http://dbpedia.org/resource/Capital>    }

"""

udep_q2 = ['QUESTION(0:x)', 'arg0(6:e , 5:m.Australia)', 'capital(6:s , 5:m.Australia)',
           'population(3:s , 3:x)', 'population.arg0(3:e , 3:x)', 'population.nmod.of(3:e , 5:m.Australia)', 'what(0:s , 0:x)',
           'what.arg0(0:e , 0:x)', 'what.arg1(0:e , 3:x)']


gold_q2 = """
PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX res: <http://dbpedia.org/resource/> 
SELECT DISTINCT ?num WHERE { res:Australia dbo:capital ?x . ?x bb
 ?num }
"""
from candidate_generation import searchIndex
onto_prop = searchIndex()