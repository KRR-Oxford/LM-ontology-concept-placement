# get the edge catelogue of the SNOMED-CT subset, using the generated entity catelogue (output of get_all_SNOMED_CT_entities.py)
# include both direct linked notes (A->B) and those as parent-children paths for a node (A->x->B).
# the complex concepts (in edges) starting with [AND] (i.e. [AND]/conjunction is the outmost logical operator) and those do not contain [EX.] (i.e. do not contain existential restriction) are filtered out.

from tqdm import tqdm
import json
import os
CONST_NULL_NODE = "SCTID_NULL" # null node
CONST_THING_NODE = "SCTID_THING" # Thing node
import argparse

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    print('saving',file_name)
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str) 

def is_complex_concept(concept_id):
    '''
        return True if it is a complex concept id
    '''
    return ("[" in concept_id) and ("]" in concept_id) and ("<" in concept_id) and (">" in concept_id) and ("(" in concept_id) and (")" in concept_id)
       
def is_to_filter_complex_concept(complex_concept_id):
    '''
        return True if the complex concept is to be filtered: i.e. 
        (i) can be decomposed by conjunction (starting with [AND])
        (ii) do not contain [EX.]
    '''
    return (not "[EX.]" in complex_concept_id) or complex_concept_id.startswith("[AND]")

parser = argparse.ArgumentParser(description="get the list file of SNOMED-CT edges")
parser.add_argument('--onto_ver', type=str,
                    help="SNOMED-CT subset version", default='20140901-Disease')
parser.add_argument('--concept_type', type=str,
                    help=" concept_type: \"atomic\", \"complex\", \"all\" (or any other strings)", default='atomic')
args = parser.parse_args()

add_synonyms_as_ents = False

#onto_ver = '20140901-Disease'
#onto_ver = '20140901-CPP'
onto_ver = args.onto_ver

concept_type = args.concept_type
ent_catelogue_fn='../ontologies/SNOMEDCT-US-%s%s_hyp%s.jsonl' % (onto_ver, '_syn_full' if add_synonyms_as_ents else '_syn_attr', (('-' + concept_type) if concept_type != 'atomic' else ''))

'''
An example of entity catalogue
{"text": "", 
"idx": "10001005", 
"title": "bacterial sepsis (disorder)", 
"entity": "bacterial sepsis (disorder)", 
"parents_idx": "87628006|91302008", 
"children_idx": "196853004|195284000|196111003|310669007|240385003|240389009|198462004|18700001|4089001|5567000|41936006|310649002", 
"parents": "bacterial infectious disease (disorder)|sepsis (disorder)", 
"children": "septicemic pasteurellosis (disorder)|hemorrhagic septicemia barbone (disorder)|bacterial hemorrhagic septicemia (disorder)|septicemia due to enterococcus (disorder)|septicemic glanders (disorder)|septicemic melioidosis (disorder)|hemorrhagic septicemia due to pasteurella multocida (disorder)|septicemia due to chromobacterium (disorder)|meningococcemia (disorder)|gas gangrene septicemia (disorder)|septicemia due to erysipelothrix insidiosa (disorder)|vancomycin resistant enterococcal septicemia (disorder)", 
"parents-children_idx": "87628006-196853004|91302008-196853004|87628006-195284000|91302008-195284000|87628006-196111003|91302008-196111003|87628006-310669007|91302008-310669007|87628006-240385003|91302008-240385003|87628006-240389009|91302008-240389009|87628006-198462004|91302008-198462004|87628006-18700001|91302008-18700001|87628006-4089001|91302008-4089001|87628006-5567000|91302008-5567000|87628006-41936006|91302008-41936006|87628006-310649002|91302008-310649002", 
"synonyms": "bacterial sepsis|bacterial septicemia|bacterial septicaemia"}
'''

with open(ent_catelogue_fn,encoding='utf-8-sig') as f_content:
    doc = f_content.readlines()

dict_pc_edge_tuples = {} # can be direct (A->B) or 1-degree linked (A->x->B).
list_edge_json_str = []
for ind_ent, entity_info_json in tqdm(enumerate(doc)):
    entity_info = json.loads(entity_info_json)
    concept_id = entity_info['idx']
    concept_tit = entity_info['title']
    parent_ids = entity_info['parents_idx']
    child_ids = entity_info['children_idx']
    parents = entity_info['parents']
    children = entity_info['children']
    #pc_paths = entity_info['parents-children_idx']
    #synonyms = entity_info['synonyms']

    parent_ids = CONST_THING_NODE if parent_ids == '' else parent_ids
    child_ids = CONST_NULL_NODE if child_ids == '' else child_ids
    parents = 'TOP' if parents == '' else parents
    children = 'NULL' if children == '' else children 

    list_parent_ids = parent_ids.split('|')
    list_child_ids = child_ids.split('|')
    list_parents = parents.split('|')
    list_children = children.split('|')
    #list_pc_path_tuples = [tuple(pc_path.split('-')) for pc_path in pc_paths.split('|')]

    # filter parent and children
    list_parent_ids_filtered = []
    list_parents_filtered = []
    for parent_id,parent in zip(list_parent_ids,list_parents):
        if is_complex_concept(parent_id):
            if not is_to_filter_complex_concept(parent_id):
                list_parent_ids_filtered.append(parent_id)
        else:
            list_parent_ids_filtered.append(parent_id)
    list_child_ids_filtered = []
    list_children_filtered = []
    for child_id,child in zip(list_child_ids,list_children):
        if is_complex_concept(child_id):
            if not is_to_filter_complex_concept(child_id):
                list_child_ids_filtered.append(child_id)
        else:
            list_child_ids_filtered.append(child_id)        
    
    for p_ind, parent_id in enumerate(list_parent_ids_filtered):
        if not (parent_id,concept_id) in dict_pc_edge_tuples:
            dict_pc_edge_tuples[(parent_id,concept_id)] = 1

            dict_edge_row = {}
            dict_edge_row['parent_idx'] = parent_id
            dict_edge_row['child_idx'] = concept_id
            dict_edge_row['parent'] = list_parents[p_ind]
            dict_edge_row['child'] = concept_tit
            dict_edge_row['degree'] = 0

            edge_json_str = json.dumps(dict_edge_row)
            list_edge_json_str.append(edge_json_str)
    
    for c_ind, child_id in enumerate(list_child_ids_filtered):
        if not (concept_id,child_id) in dict_pc_edge_tuples:
            dict_pc_edge_tuples[(concept_id,child_id)] = 1

            dict_edge_row = {}
            dict_edge_row['parent_idx'] = concept_id
            dict_edge_row['child_idx'] = child_id
            dict_edge_row['parent'] = concept_tit
            dict_edge_row['child'] = list_children[c_ind]
            dict_edge_row['degree'] = 0

            edge_json_str = json.dumps(dict_edge_row)
            list_edge_json_str.append(edge_json_str)
    
    for p_ind, parent_id in enumerate(list_parent_ids_filtered):
        for c_ind, child_id in enumerate(list_child_ids_filtered):
            if not (parent_id,child_id) in dict_pc_edge_tuples:
                dict_pc_edge_tuples[(parent_id,child_id)] = 1

                dict_edge_row = {}
                dict_edge_row['parent_idx'] = parent_id
                dict_edge_row['child_idx'] = child_id
                dict_edge_row['parent'] = list_parents[p_ind]
                dict_edge_row['child'] = list_children[c_ind]
                dict_edge_row['degree'] = 1

                edge_json_str = json.dumps(dict_edge_row)
                list_edge_json_str.append(edge_json_str)

output_data_folder_path = '../ontologies'

output_to_file('%s/SNOMEDCT-US-%s-edges%s.jsonl' % (output_data_folder_path, onto_ver,(('-'+concept_type) if concept_type != 'atomic' else '')), '\n'.join(list_edge_json_str))

'''
direct edges (A->B)
$ grep -c "\"degree\": 0" ../ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl 
126358
one-degree edges (A->x->B)
$ grep -c "\"degree\": 1" ../ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl 
106471
'''