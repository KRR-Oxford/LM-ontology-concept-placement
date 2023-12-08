# generate data for edges - 
# for each mention row, display the (i) label concept parent id/name and (ii) label concept children id/name,
# and also the edge row id in the edge catalogue (e.g., ../ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl)

# input:  (i) the output data files of format_trans_medmentions2blink+new.py
#               e.g., data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge
#         (ii) the output edge catalogue file of get_all_SNOMED_CT_edges.py
# output: (i) mention-edge-pair data: the re-formatted data files for insertion into edges (full (in-KB+NIL), NIL, final).

                # the "final" version contains: 
                    # in-KB insertion full set (train+valid+test), "train-all.jsonl"
                    # in_KB insertion training set, "train"
                    # in-KB insertion validation set, "valid"
                    # in-KB insertion test set, "test-in-KB"
                    # unsupervised NIL insertion testing set

        # (ii) mention-level data: 
            # a) "-unsup", unsupervised,
                # adapt the mention-level data for edge insertion. 
                # by moving the NIL mentions in train and valid, to test.
                # This is equiv to the content of the mention-edge-pair-data                
            # b) "-filt", filtered,
                # given that we only selected the one-hop/degree and two-hop edges (regarding any paths in the ontology), we also filter out the mentions in the mention-level data.
                # we also filter out the edges, a few, which are not in the edge catalogue.
            # c) "-filt" for "full" (i.e. synonym as entity) setting (the above are "attr", synonym as attr, setting ).
            # d) "-filt" for Sieve format (from "attr") - by setting --update_sieve_data

from tqdm import tqdm
import json
import os,sys
import argparse
import math

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

def display_dict_w_freq(dict_id_freq):
    '''
        display the dict of key as id and value as freq of the key
    '''
    print('length of dict:', len(dict_id_freq))
    print('sum of values:', sum(dict_id_freq.values()))
    print('sorted descendently:')
    dict_id_freq_ordered = dict(sorted(dict_id_freq.items(), key=lambda x: x[1], reverse=True))
    for id, freq in dict_id_freq_ordered.items():
        print('\t', id, freq)

def get_dict_of_parent_child_edge_in_catalogue(edge_catalogue_fn='../ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl'):
    '''
        get the dict of parent_id-child_id tuple to the list of row ids and the list of corresponding titles (as a tuple) in the edge catelogue (generated with "get_all_xxx_edges(+).py)
    '''
    dict_edge_tuple_ind = {}
    dict_edge_tuple_title_tuple = {}
    with open(edge_catalogue_fn,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()
    for ind_ent, edge_info_json in tqdm(enumerate(doc)):
        edge_info = json.loads(edge_info_json)
        parent_id = edge_info['parent_idx']
        child_id = edge_info['child_idx']
        parent = edge_info['parent']
        child = edge_info['child']
        assert not (parent_id,child_id) in dict_edge_tuple_ind
        assert not (parent_id,child_id) in dict_edge_tuple_title_tuple
        dict_edge_tuple_ind[(parent_id,child_id)] = ind_ent
        dict_edge_tuple_title_tuple[(parent_id,child_id)] = (parent,child)
    return dict_edge_tuple_ind,dict_edge_tuple_title_tuple

parser = argparse.ArgumentParser(description="transform the mention-level data to edge-level data")
parser.add_argument('--onto_ver', type=str,
                    help="SNOMED-CT version", default='20140901')
parser.add_argument('--snomed_subset', type=str,
                    help="SNOMED-CT subset mark: Disease, CPP, etc.", default='Disease')
parser.add_argument('--concept_type', type=str,
                    help=" concept_type: \"atomic\", \"complex\", \"all\" (or any other strings)", default='atomic')
parser.add_argument("--allow_complex_edge",action="store_true",help="Whether to allow one or two sides of the edge to have complex concepts - for cases of \"all\" and \"complex\"")
parser.add_argument("--update_sieve_data",action="store_true",help="Whether to update the Sieve format data")
args = parser.parse_args()

#edge_catalogue_fn = '../ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl'
#edge_catalogue_fn = '../ontologies/SNOMEDCT-US-20140901-Disease-edges-all.jsonl'
edge_catalogue_fn = '../ontologies/SNOMEDCT-US-%s-%s-edges%s.jsonl' % (args.onto_ver,args.snomed_subset, ('-' + args.concept_type) if args.concept_type != 'atomic' else '')
dict_edge_tuple_ind,dict_edge_tuple_title_tuple = get_dict_of_parent_child_edge_in_catalogue(edge_catalogue_fn)

dataset_folder_path = "../data/MedMentions-preprocessed+"
dataset_folder_path_sieve = dataset_folder_path + "sieve" 

context_length = 256 # context length (ctxt_l + mention + ctxt_r) used to form context for the dataset (this is need to match mentions in .jsonl to Sieve format)

#dataset_name = "st21pv_syn_attr"
dataset_name = "%s/st21pv_syn_attr%s%s" % (args.snomed_subset,('-' + args.concept_type) if args.concept_type != 'atomic' else '','-complexEdge' if args.allow_complex_edge else '')
dataset_name_syn = "%s/st21pv_syn_full%s%s" % (args.snomed_subset,('-' + args.concept_type) if args.concept_type != 'atomic' else '','-complexEdge' if args.allow_complex_edge else '')

output_data_folder_path = '%s/%s-edges' % (dataset_folder_path,dataset_name)
output_data_folder_path_NIL = '%s/%s-edges-NIL' % (dataset_folder_path,dataset_name)
output_data_folder_path_final = '%s/%s-edges-final' % (dataset_folder_path,dataset_name)
# mention-level data
output_data_folder_path_mention_lvl_unsup = "%s/%s-unsup" % (dataset_folder_path,dataset_name) # unsupervised for NIL mention insertion
output_data_folder_path_mention_lvl_filt = '%s/%s-filt' % (dataset_folder_path,dataset_name)
output_data_folder_path_mention_lvl_syn_filt = '%s/%s-filt' % (dataset_folder_path,dataset_name_syn)
output_data_folder_path_mention_lvl_filt_sieve = '%s/%s-filt' % (dataset_folder_path_sieve,dataset_name) # data for sieve

dict_concept_NIL = {}
dict_data_split_mark_to_list_json_row_in_KB = {} # to a tuple of mention-edge-level-rows and mention-level-rows
dict_data_split_mark_to_list_json_row_NIL = {} # to a tuple of mention-edge-level-rows and mention-level-rows
dict_data_split_mark_to_list_json_row_filt = {} # to the mention-level-rows (syn-attr)
dict_data_split_mark_to_list_json_row_syn_filt = {} # to the mention-level-rows (syn-full)
for data_split_mark in ["train", "valid", "test"]:
    mention_filtered_out_by_edge_path = {} # dict of mention (w/ contexts concat: ctxt_l + mention + ctxt_r) be filtered by edge path

    list_mention_for_insertion_json_row = []
    list_mention_for_insertion_json_row_in_KB = []
    list_mention_for_insertion_json_row_NIL = []
    list_mention_json_row_in_KB =[]
    list_mention_json_row_NIL =[]
    list_mention_json_row_filt = []
    dict_concept_NIL_data_split = {}
    with open("%s/%s/%s.jsonl" % (dataset_folder_path,dataset_name,data_split_mark),encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)  

        has_edge_in_catalogue = False
        dict_mention_for_insertion_row={}    
        dict_mention_for_insertion_row["context_left"] = mention_info["context_left"]
        dict_mention_for_insertion_row["mention"] = mention_info["mention"]
        dict_mention_for_insertion_row["context_right"] = mention_info["context_right"]
        dict_mention_for_insertion_row["label_concept_UMLS"] = mention_info["label_concept_UMLS"]
        dict_mention_for_insertion_row["label_concept"] = mention_info["label_concept"]
        dict_mention_for_insertion_row["label_concept_ori"] = mention_info["label_concept_ori"]
        dict_mention_for_insertion_row["entity_label_id"] = mention_info["label_id"]
        dict_mention_for_insertion_row["entity_label"] = mention_info["label"]
        dict_mention_for_insertion_row["entity_label_title"] = mention_info["label_title"]
        pc_paths_str = mention_info["parents-children_concept"]
        if pc_paths_str != '':
            list_pc_paths = pc_paths_str.split('|')
            list_pc_paths_updated = list_pc_paths[:]
            for pc_path in list_pc_paths:
                pc_tuple = tuple(pc_path.split('-'))
                dict_mention_for_insertion_row['parent_concept'] = pc_tuple[0]
                dict_mention_for_insertion_row['child_concept'] = pc_tuple[1]
                if not pc_tuple in dict_edge_tuple_title_tuple:
                    # here will filter out the mentions with edges which are not in the edge catalogue (one-hop, including leaf-to-NULL edges, and two-hop edges)
                    print(pc_tuple, 'not in dict_edge_tuple_title_tuple')    
                    # update list_pc_paths to those only in the edge catalogue
                    list_pc_paths_updated.remove(pc_path)                
                    continue
                    # or we can keep the like this below to include all k-hop edges
                    #dict_edge_tuple_title_tuple[pc_tuple] = ("unknown","unknown")
                    #dict_edge_tuple_ind[pc_tuple] = -1

                has_edge_in_catalogue = True # True if for the mention there is at least one edge in the catalogue, otherwise False
                pc_title_tuple = dict_edge_tuple_title_tuple[pc_tuple]
                dict_mention_for_insertion_row['parent'] = pc_title_tuple[0]
                dict_mention_for_insertion_row['child'] = pc_title_tuple[1]
                assert pc_tuple in dict_edge_tuple_ind
                dict_mention_for_insertion_row['edge_label_id'] = dict_edge_tuple_ind[pc_tuple]

                mention_for_insertion_json_str = json.dumps(dict_mention_for_insertion_row)
                list_mention_for_insertion_json_row.append(mention_for_insertion_json_str)

                if dict_mention_for_insertion_row["label_concept"] == 'SCTID-less':
                    list_mention_for_insertion_json_row_NIL.append(mention_for_insertion_json_str)
                    
                    # gather and count the number of NIL concepts and each of their freq
                    #    for the data split
                    if dict_mention_for_insertion_row["label_concept_ori"] in dict_concept_NIL_data_split:
                        dict_concept_NIL_data_split[dict_mention_for_insertion_row["label_concept_ori"]] += 1
                    else:
                         dict_concept_NIL_data_split[dict_mention_for_insertion_row["label_concept_ori"]] = 1
                    #    for all 
                    if dict_mention_for_insertion_row["label_concept_ori"] in dict_concept_NIL:
                        dict_concept_NIL[dict_mention_for_insertion_row["label_concept_ori"]] += 1
                    else:
                         dict_concept_NIL[dict_mention_for_insertion_row["label_concept_ori"]] = 1
                else:
                    list_mention_for_insertion_json_row_in_KB.append(mention_for_insertion_json_str)     

            # update mention_info_json with the updated the list_pc_paths
            mention_info["parents-children_concept"] = '|'.join(list_pc_paths_updated)
            mention_info_json = json.dumps(mention_info)
        else:
            print('row', ind, 'no parent-child paths.')

        if has_edge_in_catalogue: # only record the mentions which have an edge in the catalogue
            if mention_info["label_title"] == "NIL":
                list_mention_json_row_NIL.append(mention_info_json.strip())
            else:
                list_mention_json_row_in_KB.append(mention_info_json.strip())            
            list_mention_json_row_filt.append(mention_info_json.strip())
        else:
            # record the ctxt mention tuples to be filtered out
            ctxt_men_id_tuple = (mention_info["mention"], mention_info["context_left"], mention_info["context_right"], mention_info["label_concept"])
            #print('ctxt_men_id_tuple:',ctxt_men_id_tuple)
            mention_filtered_out_by_edge_path[ctxt_men_id_tuple] = 1
            
    # for syn full setting - also loop over all the mentions
    list_mention_json_row_syn_filt = []
    with open("%s/%s/%s.jsonl" % (dataset_folder_path,dataset_name_syn,data_split_mark),encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()
    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)  
        has_edge_in_catalogue = False
        pc_paths_str = mention_info["parents-children_concept"]
        if pc_paths_str != '':
            list_pc_paths = pc_paths_str.split('|')
            list_pc_paths_updated = list_pc_paths[:]
            for pc_path in list_pc_paths:
                pc_tuple = tuple(pc_path.split('-'))
                if not pc_tuple in dict_edge_tuple_title_tuple:
                    # here it filters out the mentions with edges which are not in the edge catalogue (one-hop, including leaf-to-NULL edges, and two-hop edges)
                    print(pc_tuple, 'not in dict_edge_tuple_title_tuple')
                    # update list_pc_paths to those only in the edge catalogue
                    list_pc_paths_updated.remove(pc_path)            
                    continue
                has_edge_in_catalogue = True                
            # update mention_info_json with the updated the list_pc_paths
            mention_info["parents-children_concept"] = '|'.join(list_pc_paths_updated)
            mention_info_json = json.dumps(mention_info)
        if has_edge_in_catalogue: # only record the mentions which have an edge in the catalogue
            list_mention_json_row_syn_filt.append(mention_info_json.strip())

    # for sieve format ("filt" from "attr")
    print("mentions to be filtered for sieve:",len(mention_filtered_out_by_edge_path))
    if args.update_sieve_data:
        # get data_split_mark for sieve data 
        if data_split_mark == "train":
            data_split_mark_sieve = "trng" 
        elif data_split_mark == "valid": 
            data_split_mark_sieve = "dev"
        else: 
            data_split_mark_sieve = data_split_mark
        
        # create data split folders
        if not os.path.exists(os.path.join(output_data_folder_path_mention_lvl_filt_sieve,data_split_mark_sieve)):
            os.makedirs(os.path.join(output_data_folder_path_mention_lvl_filt_sieve,data_split_mark_sieve))        

        # loop over the mentions per each doc    
        sieve_data_split_path = "%s/%s/%s" % (dataset_folder_path_sieve,dataset_name,data_split_mark_sieve)
        #list_ann_fns = [filename for filename in os.listdir(sieve_data_split_path) if filename.endswith(".concept")]
        list_doc_fns = [filename for filename in os.listdir(sieve_data_split_path) if filename.endswith(".txt")]
        
        for ind_ann_doc_fn, doc_filename in enumerate(tqdm(list_doc_fns)):
            ann_filename = doc_filename[:len(doc_filename)-len(".txt")] + ".concept"
            if os.path.exists(os.path.join(sieve_data_split_path,ann_filename)):
                with open(os.path.join(sieve_data_split_path,ann_filename),encoding='utf-8') as f_content:
                    label_doc = f_content.read()
                    list_label_mentions = label_doc.split('\n')
            else:
                list_label_mentions = []
            #doc_filename = ann_filename[:len(ann_filename)-len(".concept")] + ".txt"
            with open(os.path.join(sieve_data_split_path,doc_filename),encoding='utf-8-sig') as f_content:
                doc = f_content.read()
                doc = doc.strip()
            label_mention_filt_list = []
            for label_mention in list_label_mentions:
                label_men_info = label_mention.split("||")
                context_pos = label_men_info[1]
                context_pos_tuple = label_men_info[1].split("|")
                snomed_concept = label_men_info[4]
                mention_pos_start = int(context_pos_tuple[0])
                mention_pos_end = int(context_pos_tuple[1])
                mention = doc[int(context_pos_tuple[0]):int(context_pos_tuple[1])]
                assert mention == label_men_info[3]
                doc_ctx_left = doc[:int(mention_pos_start)]
                doc_ctx_left_tokens = doc_ctx_left.split(' ')
                ctx_len_half = math.floor(context_length/2) #math.floor((context_length-1)/2)
                context_left = ' '.join(doc_ctx_left_tokens[-ctx_len_half:])
                doc_ctx_right = doc[int(mention_pos_end):]
                doc_ctx_right_tokens = doc_ctx_right.split(' ')
                if snomed_concept == "CUI-less":
                    snomed_concept = "SCTID-less"
                context_right = ' '.join(doc_ctx_right_tokens[0:ctx_len_half])    
                ctxt_men_id_tuple = (mention, context_left, context_right, snomed_concept)
                #print("ctxt_men_id_tuple:",ctxt_men_id_tuple)
                if ctxt_men_id_tuple in mention_filtered_out_by_edge_path:
                    print('mention filtered for sieve:', ctxt_men_id_tuple)
                    continue
                label_mention_filt_list.append(label_mention)
            # output doc file and updated ann file
            output_to_file("%s/%s/%s" % (output_data_folder_path_mention_lvl_filt_sieve,data_split_mark_sieve,doc_filename), doc)
            if len(list_label_mentions) > 0:
                output_to_file("%s/%s/%s" % (output_data_folder_path_mention_lvl_filt_sieve,data_split_mark_sieve,ann_filename), '\n'.join(label_mention_filt_list))

    # save the dict of data split mark to list of jsons
    dict_data_split_mark_to_list_json_row_in_KB[data_split_mark] = (list_mention_for_insertion_json_row_in_KB,list_mention_json_row_in_KB)
    dict_data_split_mark_to_list_json_row_NIL[data_split_mark] = (list_mention_for_insertion_json_row_NIL,list_mention_json_row_NIL)
    dict_data_split_mark_to_list_json_row_filt[data_split_mark] = list_mention_json_row_filt
    dict_data_split_mark_to_list_json_row_syn_filt[data_split_mark] = list_mention_json_row_syn_filt
    
    # create the output folder if not existed
    if not os.path.exists(output_data_folder_path):
        os.makedirs(output_data_folder_path)
    if not os.path.exists(output_data_folder_path_NIL):
        os.makedirs(output_data_folder_path_NIL)
    if not os.path.exists(output_data_folder_path_mention_lvl_unsup):
        os.makedirs(output_data_folder_path_mention_lvl_unsup)
    if not os.path.exists(output_data_folder_path_mention_lvl_filt):
        os.makedirs(output_data_folder_path_mention_lvl_filt) 
    if not os.path.exists(output_data_folder_path_mention_lvl_syn_filt):
        os.makedirs(output_data_folder_path_mention_lvl_syn_filt)            
    # output the in-KB+NIL original training/testing set            
    output_to_file('%s/%s.jsonl' % (output_data_folder_path, data_split_mark),'\n'.join(list_mention_for_insertion_json_row))    
    # for NIL only dataset
    output_to_file('%s/%s.jsonl' % (output_data_folder_path_NIL, data_split_mark),'\n'.join(list_mention_for_insertion_json_row_NIL))

    # get a randomly sampled subset for quick training/testing
    # shuffle the data list if not shuffled previously for splitting the validation set
    import random
    random.Random(1234).shuffle(list_mention_for_insertion_json_row)
    n_data_selected = 100
    # create the output folder if not existed
    output_to_file('%s/%s_sample%d.jsonl' % (output_data_folder_path, data_split_mark, n_data_selected),'\n'.join(list_mention_for_insertion_json_row[:n_data_selected]))

    # for NIL only dataset
    random.Random(1234).shuffle(list_mention_for_insertion_json_row_NIL)
    n_data_selected = 100
    # create the output folder if not existed
    output_to_file('%s/%s_sample%d.jsonl' % (output_data_folder_path_NIL, data_split_mark, n_data_selected),'\n'.join(list_mention_for_insertion_json_row_NIL[:n_data_selected]))

    # display the actual snomed-ct ids of the NIL mentions
    print('dict_concept_NIL_data_split for %s set' % data_split_mark)
    display_dict_w_freq(dict_concept_NIL_data_split)

# output dataset for unsupervised insertion
if not os.path.exists(output_data_folder_path_final):
    os.makedirs(output_data_folder_path_final)       

list_mention_for_insertion_json_row_in_KB_all = []
list_mention_json_row_in_KB_all = []    
for data_split_mark in ["train", "valid", "test"]:
    # create filt mention data (by edge catalogue)
    list_mention_json_row_ = dict_data_split_mark_to_list_json_row_filt[data_split_mark]
    output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_filt, data_split_mark),'\n'.join(list_mention_json_row_))

    list_mention_syn_json_row_ = dict_data_split_mark_to_list_json_row_syn_filt[data_split_mark]
    output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_syn_filt, data_split_mark),'\n'.join(list_mention_syn_json_row_))

    # create mention-edge-pair-level and unsup mention-level data
    if data_split_mark == 'train' or data_split_mark == 'valid':
        # 'train' and 'valid' set: in-KB
        list_mention_for_insertion_json_row_in_KB_, list_mention_json_row_in_KB_ = dict_data_split_mark_to_list_json_row_in_KB[data_split_mark]
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_final, data_split_mark),'\n'.join(list_mention_for_insertion_json_row_in_KB_))
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, data_split_mark),'\n'.join(list_mention_json_row_in_KB_))

        list_mention_for_insertion_json_row_in_KB_all.extend(list_mention_for_insertion_json_row_in_KB_)
        list_mention_json_row_in_KB_all.extend(list_mention_json_row_in_KB_)

        if data_split_mark == 'valid':
            # 'valid-NIL' (as train data) #mention-edge-lvl
            list_mention_for_insertion_json_row_NIL_valid = dict_data_split_mark_to_list_json_row_NIL['train'][0]
            output_to_file('%s/%s.jsonl' % (output_data_folder_path_final, data_split_mark + '-NIL'),'\n'.join(list_mention_for_insertion_json_row_NIL_valid))  #mention-edge-lvl

            # 'valid-NIL' (as train data) #mention-lvl
            list_mention_json_row_NIL_valid = dict_data_split_mark_to_list_json_row_NIL['train'][1]
            output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, data_split_mark + '-NIL'),'\n'.join(list_mention_json_row_NIL_valid))  #mention-lvl
            
    elif data_split_mark == 'test':
        # 'test-in-KB'
        list_mention_for_insertion_json_row_in_KB_, list_mention_json_row_in_KB_ = dict_data_split_mark_to_list_json_row_in_KB[data_split_mark]
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_final, data_split_mark + '-in-KB'),'\n'.join(list_mention_for_insertion_json_row_in_KB_)) #mention-edge-lvl
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, data_split_mark + '-in-KB'),'\n'.join(list_mention_json_row_in_KB_)) #mention-lvl
        list_mention_for_insertion_json_row_in_KB_all.extend(list_mention_for_insertion_json_row_in_KB_)     
        list_mention_json_row_in_KB_all.extend(list_mention_json_row_in_KB_)   
        # 'test-NIL-all' #mention-edge-lvl
        list_mention_for_insertion_json_row_NIL_all = dict_data_split_mark_to_list_json_row_NIL['train'][0] + dict_data_split_mark_to_list_json_row_NIL['valid'][0] + dict_data_split_mark_to_list_json_row_NIL['test'][0]
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_final, data_split_mark + '-NIL-all'),'\n'.join(list_mention_for_insertion_json_row_NIL_all))  #mention-edge-lvl
        # 'test-NIL' (as valid+test data) #mention-edge-lvl
        list_mention_for_insertion_json_row_NIL_test = dict_data_split_mark_to_list_json_row_NIL['valid'][0] + dict_data_split_mark_to_list_json_row_NIL['test'][0]
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_final, data_split_mark + '-NIL'),'\n'.join(list_mention_for_insertion_json_row_NIL_test))  #mention-edge-lvl

        # 'test-NIL-all' #mention-lvl
        list_mention_json_row_NIL_all = dict_data_split_mark_to_list_json_row_NIL['train'][1] + dict_data_split_mark_to_list_json_row_NIL['valid'][1] + dict_data_split_mark_to_list_json_row_NIL['test'][1]
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, data_split_mark + '-NIL-all'),'\n'.join(list_mention_json_row_NIL_all))  
        # 'test-NIL' (as valid+test data) #mention-lvl
        list_mention_json_row_NIL_test = dict_data_split_mark_to_list_json_row_NIL['valid'][1] + dict_data_split_mark_to_list_json_row_NIL['test'][1]
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, data_split_mark + '-NIL'),'\n'.join(list_mention_json_row_NIL_test))  #mention-lvl
    
        # 'test-full': in-KB + NIL (mention-level only)
        output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, data_split_mark),'\n'.join(list_mention_json_row_in_KB_ + list_mention_json_row_NIL_all)) #mention-lvl
    
output_to_file('%s/%s.jsonl' % (output_data_folder_path_final, 'train-all'),'\n'.join(list_mention_for_insertion_json_row_in_KB_all)) #mention-edge-lvl, all in-KB
output_to_file('%s/%s.jsonl' % (output_data_folder_path_mention_lvl_unsup, 'train-all'),'\n'.join(list_mention_json_row_in_KB_all)) #mention-lvl, all in-KB

# display the actual snomed-ct ids of the NIL mentions
print('dict_concept_NIL all data splits')
display_dict_w_freq(dict_concept_NIL)

'''
console output of NIL statistics, with some minor issues to fix: 

$ python format_mm_data_for_edge_insertion.py 
232829it [00:01, 153597.80it/s]
 44%|███████████████████████████████████████████████████████████████████████████▏                                                                                               | 5341/12151 [00:06<00:08, 848.65it/s]('19598007', 'SCTID_NULL') not in dict_edge_tuple_title_tuple
 61%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                  | 7409/12151 [00:09<00:08, 575.61it/s]('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
('110276005', '3214003') not in dict_edge_tuple_title_tuple
('110276005', '429427008') not in dict_edge_tuple_title_tuple
 64%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                              | 7725/12151 [00:10<00:05, 834.76it/s]('420134006', 'SCTID_NULL') not in dict_edge_tuple_title_tuple
('420134006', 'SCTID_NULL') not in dict_edge_tuple_title_tuple
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                     | 10652/12151 [00:14<00:03, 499.14it/s]('414029004', 'SCTID_NULL') not in dict_edge_tuple_title_tuple
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12151/12151 [00:17<00:00, 713.05it/s]
dict_concept_NIL_data_split for train set
length of dict: 66
sum of values: 662
sorted descendently:
         709044004 143
         703938007 98
         706970001 81
         443502000 27
         722688002 21
         65260001 20
         6624005 20
         22053006 15
         708030004 14
         451241000124108 13
         717055000 12
         716318002 10
         117051000119103 10
         709073001 10
         156370009 10
         716997004 8
         651000146102 8
         723188008 7
         14350001000004108 7
         315801000119108 7
         77547008 7
         8666004 6
         707341005 6
         24761000119107 6
         715923003 6
         11010461000119101 5
         247464001 5
         711329002 5
         716659002 5
         713456006 5
         721730009 5
         721822004 4
         321171000119102 4
         722722006 3
         83911000119104 3
         143411000119109 3
         715952000 3
         716653001 3
         253975004 2
         713425003 2
         713346006 2
         6411000179107 2
         15633171000119107 2
         122480009 2
         710027002 2
         716859000 2
         704203009 2
         710864009 1
         4661000119109 1
         720580006 1
         707585008 1
         708090002 1
         713313000 1
         721428008 1
         723082006 1
         101401000119103 1
         714253009 1
         721410006 1
         723170000 1
         721763002 1
         713609000 1
         122811000119101 1
         452241000124100 1
         234975001 1
         710106005 1
         721193002 1
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4409/4409 [00:07<00:00, 612.15it/s]
dict_concept_NIL_data_split for valid set
length of dict: 30
sum of values: 627
sorted descendently:
         703938007 245
         709044004 242
         14350001000004108 35
         443502000 13
         723188008 11
         706970001 9
         708013001 7
         716659002 6
         83911000119104 6
         722964001 6
         387800004 5
         717955002 5
         4046000 4
         722688002 4
         274152003 4
         99631000119101 3
         156370009 3
         11010461000119101 3
         721711009 2
         716768008 2
         709073001 2
         722606000 2
         714253009 1
         711329002 1
         713609000 1
         716305005 1
         720580006 1
         7111000119109 1
         715903004 1
         707585008 1
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4085/4085 [00:05<00:00, 697.76it/s]
dict_concept_NIL_data_split for test set
length of dict: 26
sum of values: 335
sorted descendently:
         709044004 231
         706970001 15
         11011871000119101 12
         371082009 11
         707585008 9
         721715000 8
         120541000119103 7
         156370009 6
         710167004 5
         721830003 4
         77547008 4
         443502000 4
         719590007 3
         14350001000004108 2
         719865001 2
         8666004 2
         15970001000004108 1
         715068009 1
         56276002 1
         709018004 1
         722975002 1
         83911000119104 1
         709109004 1
         723190009 1
         6624005 1
         722688002 1
dict_concept_NIL all data splits
length of dict: 94
sum of values: 1624
sorted descendently:
         709044004 616
         703938007 343
         706970001 105
         443502000 44
         14350001000004108 44
         722688002 26
         6624005 21
         65260001 20
         156370009 19
         723188008 18
         22053006 15
         708030004 14
         451241000124108 13
         709073001 12
         717055000 12
         11011871000119101 12
         707585008 11
         716659002 11
         77547008 11
         371082009 11
         716318002 10
         117051000119103 10
         83911000119104 10
         11010461000119101 8
         8666004 8
         716997004 8
         651000146102 8
         721715000 8
         315801000119108 7
         708013001 7
         120541000119103 7
         711329002 6
         707341005 6
         24761000119107 6
         715923003 6
         722964001 6
         247464001 5
         713456006 5
         721730009 5
         387800004 5
         717955002 5
         710167004 5
         721822004 4
         321171000119102 4
         4046000 4
         274152003 4
         721830003 4
         722722006 3
         143411000119109 3
         715952000 3
         716653001 3
         99631000119101 3
         719590007 3
         720580006 2
         253975004 2
         714253009 2
         713425003 2
         713346006 2
         6411000179107 2
         15633171000119107 2
         713609000 2
         122480009 2
         710027002 2
         716859000 2
         704203009 2
         721711009 2
         716768008 2
         722606000 2
         719865001 2
         710864009 1
         4661000119109 1
         708090002 1
         713313000 1
         721428008 1
         723082006 1
         101401000119103 1
         721410006 1
         723170000 1
         721763002 1
         122811000119101 1
         452241000124100 1
         234975001 1
         710106005 1
         721193002 1
         716305005 1
         7111000119109 1
         715903004 1
         15970001000004108 1
         715068009 1
         56276002 1
         709018004 1
         722975002 1
         709109004 1
         723190009 1
 '''