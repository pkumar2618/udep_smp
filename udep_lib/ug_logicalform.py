import re
import logging
from rdflib import URIRef, BNode
from udep_lib.ug_sparql_graph import UGSPARQLGraph
from udep_lib.sparql_builder import Query

logger = logging.getLogger(__name__)

class UGLogicalForm():
    def __init__(self, udep_lambda=None, ug_gpgraphs=None):
        self.udep_lambda = udep_lambda
        self.ug_gpgraphs = ug_gpgraphs
        try:
            logger.info(f"dependency_lambda: {udep_lambda['dependency_lambda'][0]}")
        except TypeError as e:
            logger.info(f"ug_gp_graphs: {ug_gpgraphs[0]}")


    def gpgraph_to_sparql(self, kg='dbpedia'):
        """Trnaslate the ug_gpgraph into ug_sparql_graph
        """
        # extract the nodes and relations from the ug_gpgraph and create sparql triplets
        # read out edges
        query = Query()
        variables_list = []
        for gpgraph in self.ug_gpgraphs:
        # create a dictionary of nodes using word position as index
            nodes_dict={}
            for node in gpgraph['nodes']:
               pass 
            for edge in gpgraph['Edges']:
               pass 

    def translate_to_sparql(self, kg='dbpedia'):
        """
        The logical form in udep_lambda is translated into SPARQL.
        Note: translate into SPARQL will note require reference to a Knowledge Graph for the purpose of
        denotation, as that is already done.

        :return: a Query object
        """
        query = Query()
        variables_list = []

        # # we will create a dictionary of event_id, which is dictionary of dictionary {key:predicate, value:{s_list, o_list}}
        event_triples_dict = {}
        for neod_lambda_term in self.udep_lambda['dependency_lambda'][0]:
            # neod_lambda term in the simplified represetation takes for a function name either a predicate or 
            # predicate.dependency or predicate.args
            # the arguments of the simplified formula are type-entity pair, where the type is either event type ':e'
            # or individual type ':m'. Note that in neod-grammar only 2 types are allowed. 
            # The type ':s' is used to represent that a word is a type, 
            # thus it may represent an ontology class. 
            # ontology type s, noun-phrase type m, 
            try:
                pred_dependency, type_entity = UGLogicalForm.get_atomic_name_atomic_args(neod_lambda_term)
            except TypeError as e_type:
                continue # when the name of the neod_lambda_term could n't be split into atomic_name and
                # atomic_arguments, better skip that term.
                # todo: there are many instances when the proper sparql-graph could not be found due to this 

            type_entity = type_entity.split(",")
            pred_dependency = pred_dependency.split(".")

            # if neod_term is Question then use the variable in it as select variables of the query object
            if re.match(r'^QUESTION', pred_dependency[0]):
                # Take the arguments inside it and change them into SPARQL Variable, prepend with '?'
                variables_list.append(f'?{type_entity[0]}'.replace(':', ''))


            # # identify the event_id and the predicate
            elif re.match(r'[\d]+:e', type_entity[0]):  # when identified for the first time, the event will
                # create new dictionary entry into the event_triples_dict.
                event_id = type_entity[0]
                if len(pred_dependency) == 1: # the atomic name only contains predicate
                    predicate = pred_dependency[0]
                    UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')

                elif len(pred_dependency) == 2: # the atomic name coontains predicate and the dependency-relations
                    # or it contains predicate and arg0-1
                    predicate = pred_dependency[0]
                    dependency_relations = pred_dependency[1]
                    try:
                        entity = type_entity[1].split('.')[1] # take the last word e.g. 3:m.Lovesick
                        rdf_type = 'URIRef'
                    except IndexError as e_index:  # index error
                        # the term is a varaible
                        entity = type_entity[1].replace(':', '')
                        rdf_type = 'BNode'

                    if not re.match(r'arg[\d]+', dependency_relations): # the term is dependency-relation
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
                        else: # put the entity into slist, however, check for lenght of the two lists.
                            # the triple are formed when the list are balanced.
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)

                    else: # the second term in atomic_name is arg[\d]
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
                        else:  # put the entity into slist
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)

                elif len(pred_dependency) ==3: # the third term is preopositional modifier,
                    # second term the dependency relations
                    predicate = pred_dependency[0]
                    dependency_relations = pred_dependency[1]
                    prepositional_modifier = pred_dependency[2]
                    try:
                        entity = type_entity[1].split('.')[1]
                        rdf_type = 'URIRef'
                    except IndexError as e_index:  # index error
                        # the term is a varaible
                        entity = type_entity[1].replace(':', '')
                        rdf_type = 'BNode'

                    if not re.match(r'arg[\d]+', dependency_relations):  # the term is dependency-relation
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity,
                                                       rdf_type=rdf_type)
                        else:  # put the entity into slist, however, check for length of the two lists.
                            # the triple are formed when the list are balanced.
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity,
                                                                rdf_type=rdf_type)

                    else:  # the second term in atomic_name is arg[\d]
                        UGLogicalForm.update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef')
                        # check if the entity exist in the s_list before updating it.
                        if UGLogicalForm.exists_in_slist(event_triples_dict, event_id, predicate, entity):
                            UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity,
                                                       rdf_type=rdf_type)
                        else:  # put the entity into slist
                            UGLogicalForm.update_bw_slist_olist(event_triples_dict, event_id, predicate, entity,
                                                                rdf_type=rdf_type)

            # if atomic expression is of type ':s' i.e. a type relations
            elif re.match(r'[\s]*[\d]+:s[\s]*$', type_entity[0]):
                # the atomic_expression is of type ':s'
                type_id = type_entity[0]
                variable_name = type_entity[1]
                type_label = pred_dependency[0]

                if len(pred_dependency) == 1: # only the type name is given, and the atomic_args contain variable
                    #if predicate is which/what/who/where and the variable inside is one of the variables that exist as arg
                    # in the QUESTION, we will skip counting that neod-term
                    if type_label in ['which', 'who', 'where', 'what']:
                        if f'?{variable_name.strip()}' in variables_list: #variables are store with prefix ?
                            continue # continue with the next neod-term and leave out the current word.

                    UGLogicalForm.update_plist(event_triples_dict, type_id, 'a', rdf_type='URIRef') # dbpedia uses a for type
                    UGLogicalForm.update_olist(event_triples_dict, type_id, 'a', type_label, rdf_type='URIRef')
                    UGLogicalForm.update_slist(event_triples_dict, type_id, 'a', variable_name, rdf_type='BNode')

                elif len(pred_dependency) == 2: # atomic expression for type ':s' don't have arg[\d]
                    pass


        # change variables in variables_list into ?0x type i.e. remove the colon, it is problem for sparql query
        # variables_list_original = variables_list.copy()
        # v_list_no_colon = [v_with_col.replace(':', '') for v_with_col in variables_list]
        # v_list_dict = dict(zip(variables_list_original, v_list_no_colon))
        query.select(variables_list)
        query.distinct()
        spo_triples = []
        for neod_type, kp_vso in event_triples_dict.items(): # kp_vso a dictionary of predicate as key and s_list, o_list as value
            # if neod_type is an event
            if re.match(r'[\d]+:e[\s]*', neod_type):
                # kp_vso: note that there may be many predicates and each predicate will have an associated s_list
                # and o_list, which in turn may form many subject-object pairs. There fore we create a different static
                # function to process it.
                spo_triples = spo_triples + UGLogicalForm.create_spo_triples(kp_vso)
            elif re.match(r'\s*[\d]+:s[\s]*', neod_type):
                spo_triples = spo_triples + UGLogicalForm.create_spo_triples(kp_vso)

        # put spo_tiples through connect, merge and fold operations. The expand is not required right now

        spo_triples = UGLogicalForm.graph_node_connect_merge_fold_expand(variables_list, spo_triples)
        query.where(spo_triples)
        return UGSPARQLGraph(query) # query.get_query_string()

    @staticmethod
    def get_atomic_name_atomic_args(neod_lambda_term):
        # pattern = r'(\w[\w\d._]*)\((.*)\)$'
        pattern = r'(\w[\w\d._\']*)\((.*)\)$' # to include apostrophe 's
        match = re.match(pattern, neod_lambda_term)
        if match:
            return match.group(1), match.group(2)
        else:
            return None

    @staticmethod
    def update_plist(event_triples_dict, event_id, predicate, rdf_type='URIRef'):
        # we are going to form a dictionary of dictionary of dictionary : event_id:predicate:{s_list, o_list}
        try:
            if predicate in event_triples_dict[event_id].keys():
                pass
            else :
                event_triples_dict[event_id][predicate] = {}
                # if rdf_type is 'URIRef':
                #     event_triples_dict[event_id][URIRef(predicate)] = {}
                # elif rdf_type is 'BNode':
                #         event_triples_dict[event_id][BNode(predicate)] = {}
        except KeyError as event_error:
            event_triples_dict[event_id] = {}
            event_triples_dict[event_id][predicate] = {}
            # if rdf_type is 'URIRef':
            #     event_triples_dict[event_id][URIRef(predicate)] = {}
            # elif rdf_type is 'BNode':
            #     event_triples_dict[event_id][BNode(predicate)] = {}

    @staticmethod
    def update_bw_slist_olist(event_triples_dict, event_id, predicate, entity, rdf_type=URIRef):
        try:
            s_list_len = len(event_triples_dict[event_id][predicate]['s_list'])
            o_list_len = len(event_triples_dict[event_id][predicate]['o_list'])
            if s_list_len == o_list_len:
                UGLogicalForm.update_slist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
            elif s_list_len > o_list_len:
                UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
            else:
                UGLogicalForm.update_slist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
        except KeyError as e_key:
            if e_key.args[0] is 'o_list':
                UGLogicalForm.update_olist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)
            elif e_key.args[0] is 's_list':
                UGLogicalForm.update_slist(event_triples_dict, event_id, predicate, entity, rdf_type=rdf_type)

    @staticmethod
    def update_slist(event_triples_dict, event_id, predicate, subject, rdf_type='URIRef'):
        try:
            sub_list_len = len(event_triples_dict[event_id][predicate]['s_list'])
            if sub_list_len >= 1:  # the subject_list already has a relation corresponding to the event.
                if rdf_type is 'URIRef':
                    event_triples_dict[event_id][predicate]['s_list'].append(URIRef(subject))
                elif rdf_type is 'BNode':
                    # when subject is a blank-node it comes with ? prefixed to it. 
                    event_triples_dict[event_id][predicate]['s_list'].append(BNode(f'?{subject.strip()}'.replace(':', '')))
        except KeyError as ek:
            # create s_list
            if rdf_type is 'URIRef':
                event_triples_dict[event_id][predicate]['s_list'] = [URIRef(subject)]  # assign the subject
            elif rdf_type is 'BNode':
                event_triples_dict[event_id][predicate]['s_list'] = [BNode(f'?{subject.strip()}'.replace(':', ''))]  # assign the subject

    @staticmethod
    def exists_in_slist(event_triples_dict, event_id, predicate, entity):
        try:
            if entity in event_triples_dict[event_id][predicate]['s_list']:
                return True
        except KeyError as e_key:
            # if KeyError is encountered, that mean s_list doesn't exist
            return False

    @staticmethod
    def update_olist(event_triples_dict, event_id, predicate, object, rdf_type='URIRef'):
        try:
            object_list_len = len(event_triples_dict[event_id][predicate]['o_list'])
            if object_list_len >= 1:  # the object_list already has a entity corresponding to the event.
                if rdf_type is 'URIRef':
                    event_triples_dict[event_id][predicate]['s_list'].append(URIRef(object))
                elif rdf_type is 'BNode':
                    event_triples_dict[event_id][predicate]['s_list'].append(BNode(f'?{object.strip()}'.replace(':', '')))
        except KeyError as ek:
            # create the object_list with entity_id/type_id
            if rdf_type  is 'URIRef':
                event_triples_dict[event_id][predicate]['o_list'] = [URIRef(object)] # assign the object
            elif rdf_type is 'BNode':
                event_triples_dict[event_id][predicate]['o_list'] = [BNode(f'?{object.strip()}'.replace(':', ''))]  # assign the object

    @staticmethod
    def exists_in_olist(event_triples_dict, event_id, predicate, entity):
        try:
            if entity in event_triples_dict[event_id][predicate]['o_list']:
                return True
        except KeyError as e_key:
            # when KeyError encountered, that mean object_list doen't exists.
            return False

    @staticmethod
    def create_spo_triples(kp_vso):
    # kp_vso: note that there may be many predicates and each predicate will have an associated s_list
    # and o_list, which in turn may form many subject-object pairs.
        spo_triples = []
        for predicate, slist_olist_dict in kp_vso.items():
            try:
                assert len(slist_olist_dict['s_list']) == len(slist_olist_dict['o_list']), "hanging edge: missing subject or object node"
                predicate = URIRef(predicate)
                spo_triples = spo_triples + [[s, predicate, o] for s,o in zip(slist_olist_dict['s_list'], slist_olist_dict['o_list'])]
            except (AssertionError, KeyError) as e:
                pass

        return spo_triples

    @staticmethod
    def graph_node_connect_merge_fold_expand(query_var, spo_triples):
        # merge_rule-1: merge triple having sub and obj connected by wh-phrase
        variables_list = set()
        replace_variable_dict = {}
        replace_triple_idx =[]
        for idx, spo in enumerate(spo_triples):
            if spo[0].startswith('?') and spo[2].startswith('?'): #if the variable node
                variables_list.add(f'{spo[0]}')
                variables_list.add(f'{spo[2]}')
                
                if re.match(r'[wW]h[a-zA-Z]+', spo[1]):
                    if f'{spo[0]}' in query_var:
                        replace_variable_dict[f'{spo[2]}'] = f'{spo[0]}' #note dict has key-value reversed
                    elif f'{spo[2]}' in query_var:
                        replace_variable_dict[f'{spo[0]}'] = f'{spo[2]}'
                    replace_triple_idx.append(idx)

        # remove the triple
        popped_count = 0
        for idx in replace_triple_idx:
            spo_triples.pop(idx-popped_count)
            popped_count += 1

        # correct the variables binding/name
        for idx, spo in enumerate(spo_triples):
            if f'{spo[0]}' in replace_variable_dict.keys():
                spo_triples[idx][0] = BNode(replace_variable_dict[f'{spo[0]}'])
            elif f'{spo[2]}' in replace_variable_dict.keys():
                spo_triples[idx][2] = BNode(replace_variable_dict[f'{spo[2]}'])

        return [tuple(spo) for spo in spo_triples]
        # rule-2:
