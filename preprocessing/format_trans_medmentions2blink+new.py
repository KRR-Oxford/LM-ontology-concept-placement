# transform the format from medmentions for automated NIL generation for insertion of SNOMED-CT
# input both older and newer versions of SNOMED-CT

# To note: 
# 1. order of annotations not considered - this is OK when reading properties separately
# 2. mention to a UMLS CUI to multiple SNOMED CT ID - so far we treat this as different mentions, each to a different SNOMED CT ID

from pubtator_loader import PubTatorCorpusReader
from tqdm import tqdm
import json
import os,sys
import math
import argparse

# from deeponto import init_jvm
# init_jvm("32g")
# from deeponto.onto import Ontology, OntologyReasoner
from onto_snomed_owl_util import load_SNOMEDCT_deeponto,load_deeponto_verbaliser, load_SNOMEDCT_owl2dict_ids,deeponto2dict_ids,deeponto2dict_ids_obj_prop,deeponto2dict_ids_ann_prop,get_definition_in_onto_from_iri,get_rdfslabel_in_onto_from_iri,get_preflabel_in_onto_from_iri,get_altlabel_in_onto_from_iri,get_title_in_onto_from_iri,get_entity_info,get_SCTID_from_OWLobj,get_iri_from_SCTID_id,get_SCTID_id_from_iri,get_in_KB_direct_children,get_in_KB_direct_parents,get_entity_graph_info,get_concept_tit_from_id_strs,verbalise_concept

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

# get UMLS CUI to SNOMEDCT ID dict
def get_dict_CUI2SNOMEDCT(UMLS_file_path):
    # get dict of CUI to the list of SNOMEDCT_US IDs
    dict_CUI2SCTID = {}
    with open(UMLS_file_path,encoding='utf-8') as f_content:
        doc = f_content.readlines()

    for line in tqdm(doc):
        data_eles = line.split('|')
        source = data_eles[11]
        if source == 'SNOMEDCT_US':
            CUI = data_eles[0]
            SCTID = data_eles[13]
            if CUI in dict_CUI2SCTID:
                list_SCTID = dict_CUI2SCTID[CUI]
                if not SCTID in list_SCTID:
                    list_SCTID.append(SCTID)
                    dict_CUI2SCTID[CUI] = list_SCTID
            else:            
                list_SCTID = [SCTID]
                dict_CUI2SCTID[CUI] = list_SCTID
    return dict_CUI2SCTID

def get_dict_id_to_list_row_id_tit_from_ent_catelogue(ent_catelogue_fn='../ontologies/SNOMEDCT-US-20140901-Disease_with_NIL_syn_attr_hyp.jsonl'):
    '''
        get the dict of ontology id to the list of row ids and the list of corresponding titles in the entity catelogue (generated with "get_all_xxx_entities(+).py)
    '''
    dict_CUI_ind = {}
    dict_CUI_title = {}
    with open(ent_catelogue_fn,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()
    for ind_ent, entity_info_json in tqdm(enumerate(doc)):
        entity_info = json.loads(entity_info_json)
        concept_CUI = entity_info['idx']
        concept_tit = entity_info["title"]
        dict_CUI_ind = add_dict_list(dict_CUI_ind,concept_CUI,ind_ent)
        dict_CUI_title = add_dict_list(dict_CUI_title,concept_CUI,concept_tit)
    return dict_CUI_ind,dict_CUI_title

# add an element to a dict of list of elements for the id
def add_dict_list(dict,id,ele):
    if not id in dict:
        dict[id] = [ele] # one-element list
    else:
        list_ele = dict[id]
        list_ele.append(ele)
        dict[id] = list_ele
    return dict

def Merge(dict_1, dict_2):
	return dict(**dict_1,**dict_2)
	
parser = argparse.ArgumentParser(description="format the medmention dataset with different ontology settings")
parser.add_argument('--onto_ver', type=str,
                    help="UMLS version", default='2017AA')
parser.add_argument('--snomed_subset', type=str,
                    help="SNOMED-CT subset mark: Disease, CPP, etc.", default='Disease')
parser.add_argument("--add_synonyms_as_ents",action="store_true",help="Whether to add synonyms to the generated training data, with each synonym as an \'entity\'")
parser.add_argument('--data_setting', type=str,
                    help="data setting for medmentions: full or st21pv", default='st21pv') 
parser.add_argument('--concept_type', type=str,
                    help=" concept_type: \"atomic\", \"complex\", \"all\" (or any other strings)", default='atomic') 
parser.add_argument("--allow_complex_edge",action="store_true",help="Whether to allow one or two sides of the edge to have complex concepts - for cases of \"all\" and \"complex\"")

args = parser.parse_args()

# get dict of SNOMEDCT new ver subset
newer_ontology_name = "SNOMEDCT-US-20170301-%s-final.owl" % args.snomed_subset
fn_ontology = "../ontologies/%s" % newer_ontology_name 
onto_sno_newer = load_SNOMEDCT_deeponto(fn_ontology)
onto_sno_newer_verbaliser = load_deeponto_verbaliser(onto_sno_newer)
dict_SCTID_onto_newer = deeponto2dict_ids(onto_sno_newer)

# get dict of SNOMEDCT old ver subset
older_ontology_name = "SNOMEDCT-US-20140901-%s-final.owl" % args.snomed_subset
fn_ontology = "../ontologies/%s" % older_ontology_name 
onto_sno_older = load_SNOMEDCT_deeponto(fn_ontology)
onto_sno_older_verbaliser = load_deeponto_verbaliser(onto_sno_older)
dict_SCTID_onto_older = deeponto2dict_ids(onto_sno_older)
dict_SCTID_onto_older_obj_prop = deeponto2dict_ids_obj_prop(onto_sno_older)
#dict_SCTID_onto_older_ann_prop = deeponto2dict_ids_ann_prop(onto_sno_older)
dict_SCTID_onto_older_class_and_obj_prop = Merge(dict_SCTID_onto_older,dict_SCTID_onto_older_obj_prop) # combining both classes and object properties, only used for complex concept filtering.

onto_ver=args.onto_ver
UMLS_file_path = '../ontologies/UMLS%s/MRCONSO.RRF' % onto_ver # two split file combined via linux command 'cat MRCONSO.RRF.?? > MRCONSO.RRF'
dict_CUI2SCTID = get_dict_CUI2SNOMEDCT(UMLS_file_path)
print('dict_CUI2SCTID:',len(dict_CUI2SCTID))

add_synonyms_as_ents = args.add_synonyms_as_ents

concept_type = args.concept_type
allow_complex_edge = args.allow_complex_edge

ent_catelogue_fn='../ontologies/SNOMEDCT-US-20140901-%s_with_NIL%s_hyp%s.jsonl' % (args.snomed_subset,'_syn_full' if add_synonyms_as_ents else '_syn_attr', ('-'+concept_type) if concept_type != 'atomic' else '')
dict_SCTID_ind,dict_SCTID_title = get_dict_id_to_list_row_id_tit_from_ent_catelogue(ent_catelogue_fn)

context_length = 256 # the overall length of context (left + right)
data_setting = args.data_setting # full or st21pv
data_path = '../data/MedMentions/%s/data/corpus_pubtator.txt' % data_setting
#output_data_folder_path = '../data/MedMentions-preprocessed+/%s%s%s' % (data_setting,onto_ver.replace('/','-'), '_syn_full' if add_synonyms_as_ents else '_syn_attr', ('-'+concept_type) if concept_type != 'atomic' else '')
output_data_folder_path = '../data/MedMentions-preprocessed+/%s/%s%s%s%s' % (args.snomed_subset,data_setting, '_syn_full' if add_synonyms_as_ents else '_syn_attr', ('-'+concept_type) if concept_type != 'atomic' else '','-complexEdge' if allow_complex_edge else '') # not showing onto_ver here

dict_NIL_ori_concept2mention = {}
list_data_json_str_all = []
for data_split_mark in ['trng','dev','test']:
    list_data_json_str = [] # gather all the jsons (each for a mention and its entity) from the document (for BLINK) 

    list_NIL_mention_concept = [] # record a tuple of NIL mention (mention) and concept (original -> NIL)

    pmids_split_fn = '../data/MedMentions/full/data/corpus_pubtator_pmids_%s.txt' % data_split_mark

    with open(pmids_split_fn) as f_content:
        pmids_list = f_content.readlines()
    pmids_list = [pmid.strip() for pmid in pmids_list]
    #print(pmids_list[:3])
    dataset_reader = PubTatorCorpusReader(data_path)

    corpus = dataset_reader.load_corpus() 

    for doc in tqdm(corpus):
        #print(doc)
        pmid_doc = str(doc.id)
        #print(pmid_doc)
        if pmid_doc in pmids_list:
            entities = doc.entities
            for entity in entities:
                mention = entity.text_segment
                title_text = doc.title_text
                abstract_text = doc.abstract_text
                doc_text = title_text + ' ' + abstract_text
                mention_pos_start = entity.start_index
                mention_pos_end = entity.end_index
                assert mention == doc_text[mention_pos_start:mention_pos_end] # there is no discontinuous mentions in medmentions data
                doc_ctx_left = doc_text[:int(mention_pos_start)]
                doc_ctx_left_tokens = doc_ctx_left.split(' ')
                ctx_len_half = math.floor(context_length/2) #math.floor((context_length-1)/2)
                context_left = ' '.join(doc_ctx_left_tokens[-ctx_len_half:])
                doc_ctx_right = doc_text[int(mention_pos_end):]
                doc_ctx_right_tokens = doc_ctx_right.split(' ')
                context_right = ' '.join(doc_ctx_right_tokens[0:ctx_len_half])    
                
                concept = entity.entity_id
                concept = concept[5:] if concept[:5] == 'UMLS:' else concept
                #concept_ori = concept # for NIL concept, the original concept is saved here
                if concept in dict_CUI2SCTID:
                    list_concept_snomed = dict_CUI2SCTID[concept]
                else:
                    # only keep SNOMEDCT concepts
                    continue

                for concept_snomed in list_concept_snomed:
                    concept_snomed_ori = concept_snomed
                    concept_iri = get_iri_from_SCTID_id(concept_snomed)
                    if not concept_iri in dict_SCTID_onto_older:
                        if concept_iri in dict_SCTID_onto_newer:
                            dict_NIL_ori_concept2mention[concept_iri] = mention 
                            print(concept_iri, 'is new!')
                            
                            list_NIL_mention_concept.append((mention, concept + ' -> NIL'))
                            concept_snomed = 'SCTID-less'
                            
                        else:
                            #print(concept_iri, 'not in the newer snomedct subset')    
                            # only keep those in the newer snomed subset
                            continue
                    #else:
                    #    pass

                    if concept_snomed != 'SCTID-less':
                        # entity information for in-KB use older onto
                        concept_tit, concept_def, concept_syns = get_entity_info(onto_sno_older,concept_iri,sorting=True)

                        children_str, parents_str, pc_paths_str, children_tit_str, parents_tit_str = get_entity_graph_info(onto_sno_older,concept_iri,dict_SCTID_onto_older,onto_verbaliser=onto_sno_older_verbaliser,concept_type=concept_type,allow_complex_edge=allow_complex_edge)

                    else:
                        # entity information for NIL use new onto
                        concept_tit, concept_def, concept_syns = get_entity_info(onto_sno_newer,concept_iri,sorting=True)

                        children_str, parents_str, pc_paths_str, children_tit_str, parents_tit_str = get_entity_graph_info(onto_sno_newer,concept_iri,dict_SCTID_onto_newer,dict_SCTID_onto_filtering=dict_SCTID_onto_older_class_and_obj_prop,onto_older=onto_sno_older,onto_verbaliser=onto_sno_newer_verbaliser,concept_type=concept_type,allow_complex_edge=allow_complex_edge)
                        
                    # in the "complex" concept mode, do not include the mention if no parent is a complex concept.
                    if concept_type == "complex":
                        if parents_str == "":                       
                            continue

                    # if concept_type == "atomic"
                    #     # get the children and parent titles (for atomic concepts) - from the older onto as they should all be in the older onto
                    #     children_tit_str = get_concept_tit_from_id_strs(onto_sno_older, children_str)
                    #     parents_tit_str = get_concept_tit_from_id_strs(onto_sno_older,parents_str)

                    #form the data format for BLINK
                    #form the dictionary for this data row
                    dict_data_row = {}
                    dict_data_row['context_left'] = context_left
                    dict_data_row['mention'] = mention
                    dict_data_row['context_right'] = context_right
                    dict_data_row['label_concept_UMLS'] = concept
                    dict_data_row['label_concept'] = concept_snomed
                    dict_data_row['label_concept_ori'] = concept_snomed_ori
                    dict_data_row['label'] = concept_def # only use the first definition given that they are so similar to each other
                    #dict_data_row['synonyms'] = concept_syns
                    #dict_data_row['label_title'] = concept_tit
                    #get the hyps (direct hypernyms and hyponyms)
                    dict_data_row['parents_concept'] = parents_str
                    #print('parents_tit_str:',parents_tit_str)
                    dict_data_row['parents'] = parents_tit_str
                    if concept_type != "complex":
                        # these only exist in the "atomic" or "all" (i.e. atomic+complex) mode
                        dict_data_row['children_concept'] = children_str
                        dict_data_row['children'] = children_tit_str
                        dict_data_row['parents-children_concept'] = pc_paths_str
                    
                    if (not add_synonyms_as_ents) and data_split_mark == 'trng': # only add synonyms as attributes when chosen to and only in the training data
                        dict_data_row['synonyms'] = concept_syns
                    list_label_ids = dict_SCTID_ind.get(concept_snomed) # ent2id(concept,dict_CUI_ind)# if for_training else concept #CUI2num_id(concept) # the format for training and inference is different for the "label_id", for training, it is the row index in the entity catalogue, for inference it is the CUI or ID in the original ontology.
                    #if not concept_snomed in dict_SCTID_ind:
                    #    print(concept_snomed, 'not in dict_SCTID_ind')
                    assert concept_snomed in dict_SCTID_ind                    
                    list_concept_tits = dict_SCTID_title[concept_snomed]
                    if (not add_synonyms_as_ents) or data_split_mark != 'trng': 
                        # only use the default name, i.e. the first element in the list, if chosen not to add synonyms or when in the valid or test data.
                        list_label_ids,list_concept_tits = list_label_ids[:1],list_concept_tits[:1]
                    for label_id, concept_tit in zip(list_label_ids,list_concept_tits):
                        dict_data_row['label_id'] = label_id
                        dict_data_row['label_title'] = concept_tit
                        data_json_str = json.dumps(dict_data_row)
                        list_data_json_str.append(data_json_str)                        
                    
                    # data_json_str = json.dumps(dict_data_row)
                    # list_data_json_str.append(data_json_str)

    # create the output folder if not existed
    if not os.path.exists(output_data_folder_path):
        os.makedirs(output_data_folder_path)
    # update data split mark
    if data_split_mark == 'trng':
        data_split_mark_final = 'train'
    elif data_split_mark == 'dev':
        data_split_mark_final = 'valid'
    else:
        data_split_mark_final = data_split_mark
    # output the full, original training/testing set            
    output_to_file('%s/%s.jsonl' % (output_data_folder_path, data_split_mark_final),'\n'.join(list_data_json_str)) # for BLINK

    list_data_json_str_all.extend(list_data_json_str)

output_to_file('%s/%s.jsonl' % (output_data_folder_path, "train-all"),'\n'.join(list_data_json_str_all)) # for BLINK

for concept_ori_,mention_ in dict_NIL_ori_concept2mention.items():
    print('NIL concept created from ori concept:',concept_ori_,'mention as',mention_,'not in',older_ontology_name, ' diseases')