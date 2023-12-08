# get entity catelogue of SNOMED_CT from .owl file - in BLINK format

'''
output format: each line is 
for BLINK:
{"text": " Autism is a developmental disorder characterized by difficulties with social interaction and communication, and by restricted and repetitive behavior. Parents usually notice signs during the first three years of their child's life. These signs often develop gradually, though some children with autism reach their developmental milestones at a normal pace before worsening. Autism is associated with a combination of genetic and environmental factors. Risk factors during pregnancy include certain infections, such as rubella, toxins including valproic acid, alcohol, cocaine, pesticides and air pollution, fetal growth restriction, and autoimmune diseases. Controversies surround other proposed environmental causes; for example, the vaccine hypothesis, which has been disproven. Autism affects information processing in the brain by altering connections and organization of nerve cells and their synapses. How this occurs is not well understood. In the DSM-5, autism and less severe forms of the condition, including Asperger syndrome and pervasive developmental disorder not otherwise specified (PDD-NOS), have been combined into the diagnosis of autism spectrum disorder (ASD). Early behavioral interventions or speech therapy can help children with autism gain self-care, social, and communication skills. Although there is no known cure, there have been cases of children who recovered. Not many children with autism live independently after reaching adulthood, though some are successful. An autistic culture has developed, with some individuals seeking a cure and others believing autism should be accepted as a difference and not treated as a disorder. Globally, autism is estimated to affect 24.8 million people . In the 2000s, the number of people affected was estimated at", "idx": "https://en.wikipedia.org/wiki?curid=25", "title": "Autism", "entity": "Autism"}
'''

from tqdm import tqdm
import json
import random,math
import argparse
# from deeponto import init_jvm
# init_jvm("32g")
# from deeponto.onto import Ontology, OntologyReasoner
from onto_snomed_owl_util import load_SNOMEDCT_deeponto,load_SNOMEDCT_owl2dict_ids,deeponto2dict_ids,deeponto2dict_ids_obj_prop,deeponto2dict_ids_ann_prop,get_definition_in_onto_from_iri,get_rdfslabel_in_onto_from_iri,get_preflabel_in_onto_from_iri,get_altlabel_in_onto_from_iri,get_title_in_onto_from_iri,get_entity_info,get_SCTID_from_OWLobj,get_iri_from_SCTID_id,get_SCTID_id_from_iri,get_in_KB_direct_children,get_in_KB_direct_parents,get_entity_graph_info,get_concept_tit_from_id_strs,load_deeponto_verbaliser,get_complex_entities,get_entity_id_and_tit_complex

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    print('saving',file_name)
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

def form_str_ent_row_BLINK(CUI_def,CUI,default_name,list_synonyms,parent_CUIs_str,parents_str,children_CUIs_str,children_str,pc_CUI_paths_str,add_synonyms=True,synonym_concat_w_title=False,synonym_as_entity=True,add_direct_hyps=True):
    dict_entity_row = {}
    dict_entity_row['text'] = CUI_def
    dict_entity_row['idx'] = CUI # CUI2num_id(CUI) # from CUI to its numeric ID
    dict_entity_row['title'] = default_name
    dict_entity_row['entity'] = default_name
    if add_direct_hyps:
        # parent_CUIs_str = '|'.join(list_parent_CUIs)
        # children_CUIs_str = '|'.join(list_children_CUIs)
        # parents_str = '|'.join(list_parents)
        # children_str = '|'.join(list_children)
        # pc_CUI_paths_str = '|'.join([pc_tuple[0] + '-' + pc_tuple[1] for pc_tuple in list_pc_CUI_paths])
        dict_entity_row['parents_idx'] = parent_CUIs_str
        dict_entity_row['children_idx'] = children_CUIs_str
        dict_entity_row['parents'] = parents_str
        dict_entity_row['children'] = children_str
        dict_entity_row['parents-children_idx'] = pc_CUI_paths_str
    if add_synonyms:
        if synonym_as_entity:
        # if each synonym as a single entity - return the json entities (as a string of *multiple* rows), each using a synonym as the title
            list_json_dump = []
            list_json_dump.append(json.dumps(dict_entity_row))
            for synonym in list_synonyms:
                dict_entity_row['title'] = synonym
                list_json_dump.append(json.dumps(dict_entity_row))
            return '\n'.join(list_json_dump)
        # otherwise, synonym as a part of entity - return the json entity (as a string of a *single* row) with synonym concatenated or as an attribute.
        elif synonym_concat_w_title:
            synonyms_str = ' '.join(list_synonyms)
            dict_entity_row['title'] = default_name + ((' ' + synonyms_str) if add_synonyms else '')
            dict_entity_row['title'] = dict_entity_row['title'].strip()
        else:
            # synonyms as an attribute in the json output            
            synonyms_str = '|'.join(list_synonyms)
            dict_entity_row['synonyms'] = synonyms_str          
    return json.dumps(dict_entity_row)

def form_str_ent_row_Sieve(CUI,default_name,list_synonyms,add_synonyms=True):
    if add_synonyms:
        synonyms_str = '|'.join(list_synonyms)
    entity_row_str = CUI + '||' + default_name + (('|' + synonyms_str) if add_synonyms else '')
    return entity_row_str

parser = argparse.ArgumentParser(description="get the list file of SNOMED-CT entities")
parser.add_argument('-o','--output_data_folder_path', type=str,
                    help="output data folder path", default='../ontologies')
parser.add_argument('-f','--output_format', type=str,
                    help="output data format, BLINK or Sieve", default='BLINK')                         
parser.add_argument("--add_synonyms",action="store_true",help="Whether to add synonyms to the entity list")
parser.add_argument("--synonym_concat_w_title",action="store_true",help="Whether to concat synonyms with title")
parser.add_argument("--synonym_as_entity",action="store_true",help="Whether to treat each synonym as an entity")
parser.add_argument("--add_direct_hyps",action="store_true",help="Whether to add direct hyponyms and hypernyms as attributes to the entity list")
parser.add_argument('--onto_ver', type=str,
                    help="SNOMED-CT subset version", default='20140901-Disease')
parser.add_argument('--concept_type', type=str,
                    help=" concept_type: \"atomic\", \"complex\", \"all\" (or any other strings)", default='atomic')

args = parser.parse_args()

output_data_folder_path = args.output_data_folder_path
output_format = args.output_format
add_synonyms = args.add_synonyms
add_direct_hyps = args.add_direct_hyps
synonym_concat_w_title = args.synonym_concat_w_title
synonym_as_entity = args.synonym_as_entity
synonym_as_attr = (not synonym_concat_w_title) and (not synonym_as_entity) # if both above are false - then it puts synonyms as another attribute in the json output for each entity

concept_type = args.concept_type

# setting the output_syn_mark to differentiate the file names of the output .jsonl, for BLINK only
output_syn_mark = ''
if add_synonyms and output_format == 'BLINK':
    if synonym_concat_w_title: 
        output_syn_mark = '_concat'
    if synonym_as_entity:
        output_syn_mark = '_full'
    if synonym_as_attr:
        output_syn_mark = '_attr'         

#loop over entities in the ontology and create an entity catelogue
onto_ver = args.onto_ver #'20140901-Disease' or '20170301-Disease'
ontology_name = "SNOMEDCT-US-%s-final.owl" % onto_ver
fn_ontology = "../ontologies/%s" % ontology_name 
onto_sno = load_SNOMEDCT_deeponto(fn_ontology)
onto_sno_verbaliser = load_deeponto_verbaliser(onto_sno)
dict_SCTID_onto = deeponto2dict_ids(onto_sno)
#dict_SCTID_onto_older_obj_prop = deeponto2dict_ids_obj_prop(onto_sno_older)
#dict_SCTID_onto_older_ann_prop = deeponto2dict_ids_ann_prop(onto_sno_older)

# loop over all iris 
list_entity_json_str = []
for iri in tqdm(dict_SCTID_onto.keys()):
    # entity information for in-KB use older onto
    concept_tit, concept_def, concept_syns = get_entity_info(onto_sno,iri,sorting=True)
    children_str, parents_str, pc_paths_str, children_tit_str, parents_tit_str = get_entity_graph_info(onto_sno,iri,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type=concept_type)
    #children_tit_str = get_concept_tit_from_id_strs(onto_sno,children_str)
    #parents_tit_str = get_concept_tit_from_id_strs(onto_sno,parents_str)

    SCTID = get_SCTID_id_from_iri(iri)
    list_synonyms = concept_syns.split('|') if concept_syns != '' else []
    if output_format == 'BLINK':
        entity_row_str = form_str_ent_row_BLINK(concept_def,SCTID,concept_tit,list_synonyms,parents_str,parents_tit_str,children_str,children_tit_str,pc_paths_str,add_synonyms=add_synonyms,synonym_concat_w_title=synonym_concat_w_title,
        synonym_as_entity=synonym_as_entity,add_direct_hyps=add_direct_hyps)
    elif output_format == 'Sieve':
        entity_row_str = form_str_ent_row_Sieve(SCTID,concept_tit,list_synonyms,add_synonyms=add_synonyms)
    list_entity_json_str.append(entity_row_str)

# TODO: get the parents / children of complex concepts - so far output empty
# loop over all complex concepts 
if concept_type != 'atomic':
    complex_owl_class_expressions = get_complex_entities(onto_sno)
    for complex_owl_class_expression in tqdm(complex_owl_class_expressions):
        #print('complex_owl_class_expression:',complex_owl_class_expression)
        complex_concept_id,complex_concept_tit = get_entity_id_and_tit_complex(onto_sno_verbaliser,complex_owl_class_expression)
        #print(complex_concept_tit)
        # only retain the complex concepts having existential restrictions [EX.] and cannot be separated with conjunctions (which are outside of [EX.]).
        if (not "[EX.]" in complex_concept_id) or complex_concept_id.startswith("[AND]"):            
            continue
        complex_concept_def = ""
        list_complex_synonyms = []
        children_str, parents_str, pc_paths_str, children_tit_str, parents_tit_str = "","","","",""
        #TODO: get the parents / children of complex concepts - so far output empty
        #print("complex_owl_class_expression.isAnonymous():",complex_owl_class_expression.isAnonymous())
        #children_str, parents_str, pc_paths_str, children_tit_str, parents_tit_str = get_entity_graph_info(onto_sno,complex_owl_class_expression.asOWLClass(),dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type=concept_type)
        if output_format == 'BLINK':
            complex_entity_row_str = form_str_ent_row_BLINK(complex_concept_def,complex_concept_id,complex_concept_tit,list_complex_synonyms,parents_str,parents_tit_str,children_str,children_tit_str,pc_paths_str,add_synonyms=add_synonyms,synonym_concat_w_title=synonym_concat_w_title,
            synonym_as_entity=synonym_as_entity,add_direct_hyps=add_direct_hyps)            
        elif output_format == 'Sieve':
            complex_entity_row_str = form_str_ent_row_Sieve(complex_concept_id,complex_concept_tit,list_synonyms,add_synonyms=add_synonyms)
        list_entity_json_str.append(complex_entity_row_str)

entity_json_str = '\n'.join(list_entity_json_str)

output_to_file('%s/SNOMEDCT-US-%s%s%s%s%s%s.jsonl' % (output_data_folder_path, onto_ver, '_syn' if add_synonyms else '', output_syn_mark, '_hyp' if add_direct_hyps else '', ('-'+concept_type) if concept_type != 'atomic' else '', ('_' + output_format) if output_format != 'BLINK' else ''),entity_json_str)

# we add a general out-of-KB / NIL entity to the list - so that all out-of-KB entities share a common ID. - only for BLINK (i.e. not for Sieve)
if output_format == 'BLINK':
    entity_row_str = form_str_ent_row_BLINK(CUI_def='',CUI='SCTID-less',default_name='NIL',list_synonyms=[],parent_CUIs_str='',parents_str='',children_CUIs_str='',children_str='',pc_CUI_paths_str='',add_synonyms=add_synonyms,synonym_concat_w_title=synonym_concat_w_title,synonym_as_entity=synonym_as_entity,add_direct_hyps=add_direct_hyps)
    entity_json_str = entity_json_str + '\n' + entity_row_str

    output_to_file('%s/SNOMEDCT-US-%s_with_NIL%s%s%s%s%s.jsonl' % (output_data_folder_path, onto_ver, '_syn' if add_synonyms else '', output_syn_mark, '_hyp' if add_direct_hyps else '', ('-'+concept_type) if concept_type != 'atomic' else '', ('_' + output_format) if output_format != 'BLINK' else ''),entity_json_str)
