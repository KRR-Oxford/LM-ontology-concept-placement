# analyse and output candidates per mention from the output of eval_biencoder.py
# get results on insertion to edges with evaluation metrics (precision@k and recall@k)
# and also generate prompts 

import os 
import argparse
import torch
from tqdm import tqdm
import json
import pandas as pd
import csv
import pickle

import sys
from preprocessing.onto_snomed_owl_util import get_entity_graph_info, load_SNOMEDCT_deeponto, deeponto2dict_ids, load_deeponto_verbaliser, get_iri_from_SCTID_id, get_SCTID_id_from_iri, _extract_iris_in_parsed_complex_concept,extract_SNOMEDCT_deeponto_taxonomy,calculate_wu_palmer_sim,is_complex_concept,get_dict_iri_pair_to_lca

from blink.candidate_ranking.semantic_edge_evaluation import edge_wp_sim,edge_wp_sim_w_comp

from blink.biencoder.nn_prediction import add_dict_list, load_edge_catalogue

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

def form_edge_str(edge_json_info, with_degree=True):
    if with_degree:
        return '%s (%s) -> %s (%s), degree %d' % (edge_json_info["parent"], edge_json_info["parent_idx"], edge_json_info["child"], edge_json_info["child_idx"], edge_json_info["degree"])
    else:
        return '%s (%s) -> %s (%s)' % (edge_json_info["parent"], edge_json_info["parent_idx"], edge_json_info["child"], edge_json_info["child_idx"])
    
def construct_prompt_sub_single_pair(mention, context_left, context_right, parent, child):
    '''
    A potential prompt template
    Considering [mention] (marked with asterisk) in the context below: "[context-left] *[mention]* [context-right]" Is [mention] a child of [parent-concept]? Please answer briefly with yes or no.

    Considering [mention] (marked with asterisk) in the context below: "[context-left] *[mention]* [context-right]" Is [mention] a parent of [child-concept]? Please answer briefly with yes or no.

    #Considering [mention] (marked with asterisk) in the context below: "[context-left] *[mention]* [context-right]" Is [parent-concept] -> [mention] -> [child-concept] a taxonomy path from parent to child? Please answer briefly with yes or no.
    '''
    if parent != "TOP":
        prompt_parent = "Considering %s (marked with asterisk) in the context below: \"%s *%s* %s\" Is %s a child of %s? Please answer briefly with yes or no." % (mention, context_left, mention, context_right, mention, parent)
    else:
        prompt_parent = ""
    if child != "NULL":
        prompt_child = "Considering %s (marked with asterisk) in the context below: \"%s *%s* %s\" Is %s a parent of %s? Please answer briefly with yes or no." % (mention, context_left, mention, context_right, mention, child)
    else:
        prompt_child = ""
    #prompt_path = "Considering %s (marked with asterisk) in the context below: \"%s *%s* %s\" Is %s -> %s -> %s a taxonomy path from parent to child? Please answer briefly with yes or no." % (mention, context_left, mention, context_right, parent, mention, child)
    return prompt_parent, prompt_child#, prompt_path

# TODO - simply subsumptions to direct ones from all the predicted edges
def construct_prompt_sub(mention, context_left, context_right, list_edge_info):
    pass

def construct_prompt_edge(mention, context_left, context_right, list_edge_info):
    '''
    Can you identify the correct ontological edges for the given mention based on the provided context (context_left and context_right)? The ontological edge consists of a pair where the left concept represents the parent of the mention, and the right concept represents the child of the mention. If the mention is a leaf node, the right side of the edges will be NULL (SCTID_NULL). If the context is not relevant to the options, make your decision solely based on the mention itself. There may be multiple correct options. Please answer briefly using option numbers, separated by commas. If none of the options is correct, please answer None. 

    context_left:
    
    mention:

    context_right:
    
    options:
    '''

    # question_head = "Can you choose the correct ontological edges of the mention from the options given the context_left, context_right, and the mention below? An ontological edge is a pair where the left concept is the parent of the mention and the right concept is the child of the mention. The mention is a leaf node when NULL (SCTID_NULL) is on the right hand side of the edges. If the context is not relevant to the options, just make your decision based on the mention itself. There might be multiple correct options. Answer briefly, only with option numbers, separated by column, or None, if none of the options is correct." # manual 
    
    question_head = "Can you identify the correct ontological edges for the given mention based on the provided context (context_left and context_right)? The ontological edge consists of a pair where the left concept represents the parent of the mention, and the right concept represents the child of the mention. If the mention is a leaf node, the right side of the edges will be NULL (SCTID_NULL). If the context is not relevant to the options, make your decision solely based on the mention itself. There may be multiple correct options. Please answer briefly using option numbers, separated by a colon. If none of the options is correct, please answer None." # paraphrased by ChatGPT: "Can you help revise this paragraph so that you understand it better? \n [original-question-head-above]"

    list_edge_options = []
    for edge_rank_ind, edge_info in enumerate(list_edge_info):
        edge_str_without_degree = form_edge_str(edge_info,with_degree=False)
        list_edge_options.append('%d.%s' % (edge_rank_ind, edge_str_without_degree))
    edge_options = '\n'.join(list_edge_options)    
    prompt = "%s\n\ncontext_left:\n%s\n\nmention:\n%s\n\ncontext_right:\n%s\n\noptions:\n%s" % (question_head, context_left, mention, context_right, edge_options)
    
    return prompt

def eval_edges(dict_ctx_mention_to_list_2d_edge_strs, list_top_k=[1,3,5,10,20,50,100,150,200,250,300], tp_marking="(**p) (**c)", measure_wp=False):
    '''
    Evaluate the edges and calculate MR, MRR, micro-precision@k, micro-recall@k, mention-level r@k_any, mention-level r@k_all
    Input: the 2d list (list of mention/query of list of edge strings), the list of top-k values for precision and recall
    Output: the metric scores
    '''
    mr = 0.0
    mrr = 0.0

    for top_k_value in list_top_k:
        ave_edge_cand_len = 0
        tp = 0
        tp_any = 0 # for the mention if anyone of the gold edges are predicted
        tp_all = 0 # for the mention if all of the gold edges are predicted
        num_all_gold_edges = 0    
        # loop over the mentions: i.e. each mention's edge predictions (each as a 2d list, a |mentions|-element list of the |edges|-element list of edge-tp-marked preds)
        list_2d_men_pred_wp_scores = [] # 2d list of |mention|-pred_wp_scores 
        for ind_ment, list_2d_edge_strs in enumerate(dict_ctx_mention_to_list_2d_edge_strs.values()):
            any_edge_predicted = False
            all_edge_predicted = True
            ave_edge_cand_len += len(list_2d_edge_strs[0])
            # loop over the edges: i.e. the mention's prediction w.r.t. each edge. If the tp_marking is shown, then the mention-edge pair is predicted.
            list_2d_gold_pred_wp_scores = [] # a 2d list of rows as gold edges, columns as gold-pred wp scores.
            for ind, list_edge_strs in enumerate(list_2d_edge_strs):
                #print('len(list_edge_strs):',len(list_edge_strs))
                # if top_k_value > len(list_edge_strs):
                #     print('top-k value %d beyond predictions' % top_k_value)
                num_all_gold_edges += 1
                this_edge_predicted = False
                list_pred_wp_scores = []
                for edge_str in list_edge_strs[:top_k_value]:    
                    #print('edge_str:',edge_str)
                    # get option ind in the edge_str
                    pos_ind_op = edge_str.find('.')
                    #ind_option_str = edge_str[:pos_ind_op]            
                    # remove the option ind part in the str
                    edge_str = edge_str[pos_ind_op+1:]
                    if tp_marking in edge_str:
                        tp += 1
                        any_edge_predicted = True
                        this_edge_predicted = True
                        # remove the tp_marking in the edge str
                        edge_str = edge_str[len(tp_marking):]
                    else:
                        # remove the partial tp_marking
                        for tp_marking_partial in tp_marking.split(' '):
                            if tp_marking_partial in edge_str:
                                edge_str = edge_str[len(tp_marking_partial):]
                    # get wp score
                    wp_score = float(edge_str[edge_str.find('(')+1:edge_str.find(')')])
                    list_pred_wp_scores.append(wp_score)
                all_edge_predicted = all_edge_predicted and this_edge_predicted
                    # if tp_marking in edge_str:
                    #     mr += float(ind+1)/len(list_edge_strs)
                    #     mrr += 1/float(ind+1)
                    # else:
                    #     mr += 100
                list_2d_gold_pred_wp_scores.append(list_pred_wp_scores)
            if any_edge_predicted:
                tp_any += 1
            if all_edge_predicted:
                tp_all += 1
            if measure_wp:
                # transpose the list_2d_gold_pred_wp_scores: making list of pred edges with pred-gold wp scores the tuple for each pred edge. 
                list_pred_tuple_gold_wp_scores = zip(*list_2d_gold_pred_wp_scores)
                # get the max score for each pred edge over all gold edges of the mention
                list_pred_wp_scores = [max(tuple_gold_wp_scores) for tuple_gold_wp_scores in list_pred_tuple_gold_wp_scores]
                #print('list_pred_wp_scores (before filtering complex):', len(list_pred_wp_scores))
                # filter out the case when the pred edge is a complex edge, or all gold edges are complex, so that the pred_wp_score (after max over all gold edges) is still -1.
                list_pred_wp_scores_filtered = [pred_wp_score for pred_wp_score in list_pred_wp_scores if pred_wp_score != -1] 
                #print('list_pred_wp_scores (after filtering complex):', len(list_pred_wp_scores))
                if len(list_pred_wp_scores_filtered) == 0:
                    print('mention number %d (starting from 0) does not have pred or gold atomic edges' % ind_ment)
                else:
                    # then consider the pred_wp_scores for the mention a valid one to be added - and later we average on the scores of these mentions
                    list_2d_men_pred_wp_scores.append(list_pred_wp_scores_filtered)
        # mr = mr/len(list_2d_mention_list_edge_strs)

        num_mentions = len(dict_ctx_mention_to_list_2d_edge_strs)
        ave_edge_cand_len = ave_edge_cand_len/num_mentions
        p_at_k = float(tp)/num_mentions/top_k_value
        r_at_k = float(tp)/num_all_gold_edges
        r_at_k_any = float(tp_any)/num_mentions
        r_at_k_all = float(tp_all)/num_mentions
        if measure_wp:
            list_men_wp_scores_ave = [sum(list_pred_wp_scores_filtered) / len(list_pred_wp_scores_filtered) for list_pred_wp_scores_filtered in list_2d_men_pred_wp_scores]
            list_men_wp_scores_min = [min(list_pred_wp_scores_filtered) for list_pred_wp_scores_filtered in list_2d_men_pred_wp_scores]
            list_men_wp_scores_max = [max(list_pred_wp_scores_filtered) for list_pred_wp_scores_filtered in list_2d_men_pred_wp_scores]
            wp_at_k_denominator = len(list_2d_men_pred_wp_scores)
            wp_at_k_ave = sum(list_men_wp_scores_ave) / wp_at_k_denominator
            wp_at_k_min = sum(list_men_wp_scores_min) / wp_at_k_denominator
            wp_at_k_max = sum(list_men_wp_scores_max) / wp_at_k_denominator
        print('ave_edge_cand_len:',ave_edge_cand_len)
        print('tp-edges:',tp, 'tp-any-mentions:', tp_any, 'tp-all-mentions:', tp_all, 'num_mentions:', num_mentions,'num_gold_edges:',num_all_gold_edges)
        print('p_at_%d:' % top_k_value,p_at_k,'r_at_%d:' % top_k_value,r_at_k)
        print('r_at_%d_any:' % top_k_value, r_at_k_any)
        print('r_at_%d_all:' % top_k_value, r_at_k_all)
        if measure_wp:
            print('wp_at_%d_ave:' % top_k_value, wp_at_k_ave)
            print('wp_at_%d_min:' % top_k_value, wp_at_k_min)
            print('wp_at_%d_max:' % top_k_value, wp_at_k_max)
            print('wp_at_%d_denominator:' % top_k_value, wp_at_k_denominator)
        #return p_at_k,r_at_k

def add_dict_tuple_first_and_last_ele_list(dict,id,ele_first_str,ele_last):
    # update the ctx/doc ind list if the id exists 
    info_tuple = dict[id]
    info_list = list(info_tuple)
    # retrieve and update the first ele
    ctx_id_list = info_tuple[0]
    if not ele_first_str in ctx_id_list:
        ctx_id_list.append(ele_first_str)  
    info_list[0] = ctx_id_list    
    # retrieve and update the last ele    
    if type(ele_last) == bool:
        is_subsumption = info_tuple[-1]
        is_subsumption = is_subsumption or ele_last
        info_list[-1] = is_subsumption
    else:
        true_edge_id_list = info_tuple[-1]
        if not ele_last in true_edge_id_list:
            true_edge_id_list.append(ele_last)
        info_list[-1] = true_edge_id_list
    # update them into tuple (cast into list before updating)
    dict[id] = tuple(info_list)
    return dict

def main(params):
    #model_name = "mm+2017AA-tl-pubmedbert-NIL-tag-bs128"
    #num_top_k = 100
    fname = os.path.join(params["data_path"], "%s.t7" % params["data_split"]) # this file is generated with eval_biencoder.py (see https://github.com/facebookresearch/BLINK/issues/92#issuecomment-1126293605)
    print('fname:',fname)
    data = torch.load(fname)
    row_ids = data["row_ids"].tolist()
    #label_input = data["labels"]
    edge_inds = data["entity_inds"]
    #label_is_NIL_input = data["labels_is_NIL"]
    #print(len(edge_inds),edge_inds[0])

    iri_prefix = params['iri_prefix']

    if params["measure_wp"]:
        #load ontology and the taxonomy part of it for wu & palmer similarity
        onto_old = load_SNOMEDCT_deeponto(params["ontology_fn"])
        onto_old_taxo = extract_SNOMEDCT_deeponto_taxonomy(onto_old)
        
        #  embed and load the shortest node depth (snd) and lowest common ancestor (lca) dicts
        if 'SNOMEDCT' in params["ontology_fn"]:
            onto_name = 'snomed-ct'
            name_eles = params["ontology_fn"].split('-')
            onto_ver_subset = name_eles[3]
        elif 'DO' in params["ontology_fn"]:
            onto_name = 'doid'
            ontology_fn = params["ontology_fn"]
            ontology_fn = ontology_fn[:ontology_fn.find('_')]
            name_eles = ontology_fn.split('-')
            onto_ver_subset = '-'.join(name_eles[1:])
        else:
            print('onto name cannot be inferred - please set in the code')
            sys.exit(0)   
        
        snd_fn = "blink/prompting/%s-%s-%s.pkl" % (onto_name, onto_ver_subset, "shortest-node-depth")
        if os.path.exists(snd_fn):
            with open(snd_fn, 'rb') as data_f:
                dict_iri_to_snd = pickle.load(data_f)
                print('dict_iri_to_snd:',len(dict_iri_to_snd))
        else:
            dict_iri_to_snd = {}

        lca_fn = "blink/prompting/%s-%s-%s.pkl" % (onto_name, onto_ver_subset, "lowest-common-ancestor")
        if os.path.exists(lca_fn):
            with open(lca_fn, 'rb') as data_f:
                dict_iri_pair_to_lca = pickle.load(data_f)
                print('dict_iri_pair_to_lca:',len(dict_iri_pair_to_lca))
        else:
            dict_iri_pair_to_lca = {}
            #dict_iri_pair_to_lca = get_dict_iri_pair_to_lca(snomed_ct_old_taxo)
        
    #get edge catalogue info - load as a dict, 
    #where key is the edge index and 
    #      value is the dict of json info for each edge (in the jsonl edge catalogue file)
    dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent = load_edge_catalogue(params["edge_catalogue_fn"])
    print('edge catalogue loaded')
    print('dict_ind_edge_json_info:',len(dict_ind_edge_json_info))
    #print('dict_edge_tuple_to_ind:',len(dict_edge_tuple_to_ind))

    #get mention info
    fname_ori_data = os.path.join(params["original_data_path"], '%s.jsonl' % params["data_split"])

    with open(fname_ori_data,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    print("len(doc):", len(doc))
    print("len(edge_inds):", len(edge_inds))
    # if len(doc) != len(edge_inds):
    #     #doc = doc[:len(edge_inds)]
    #     doc = [doc[row_id] for row_id in row_ids]
    #     print("row_ids:",row_ids)
    #     print("len(doc):", len(doc))
    list_mention_w_edge_preds_strs = [] # a 1d list for diplaying
    #list_2d_mention_list_edge_strs = [] # a 2d list for evaluation
    dict_ctx_mention_to_list_2d_edge_strs = {} # dict of contextual mention to the list of all list of edge strs w.r.t each gold edge.
    dict_ctx_mention_to_list_2d_edge_strs_leaf = {} # for leaf node only
    dict_ctx_mention_to_list_2d_edge_strs_non_leaf = {} # for non-leaf node only
    dict_prompt_strs = {} # dict of prompt strs - by subsumption
    dict_prompt_strs_by_edge = {} # dict of prompt - strs by edge
    # loop over mentions (rows) in the mention-edge pair data file
    for ind, mention_info_json in enumerate(tqdm(doc)):
        if not ind in row_ids:
            continue
        else:
            edge_id = row_ids.index(ind)
        mention_info = json.loads(mention_info_json)  
        mention = mention_info["mention"]
        context_left = mention_info["context_left"]
        context_right = mention_info["context_right"]
        label_concept = mention_info["label_concept"]
        label_concept_ori = mention_info["label_concept_ori"]
        entity_label_title = mention_info["entity_label_title"]
        parent_concept = mention_info["parent_concept"]
        child_concept = mention_info["child_concept"]

        all_pred_inds = edge_inds[edge_id].tolist()
        topk_pred_inds = all_pred_inds[:params["top_k_filtering"]]
        #topk_pred_inds = all_pred_inds[:params["top_k_cand_seed"]] # select initial topk as seed (before any candidate enrichment steps)
        #topk_pred_inds = set(topk_pred_inds) # turning it into a set
        #print("topk_pred_inds:",len(topk_pred_inds))

        # # enrich edge candidates: 
        # # (i) adding a leaf edge if the parent is predicted in an edge
        # # (ii) enriching 2-hop or higher-hop edges from 1-hop edges
        # topk_pred_inds_enriched = enrich_edge_cands(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,topk_pred_inds)
        # #print("topk_pred_inds_enriched:",len(topk_pred_inds_enriched),topk_pred_inds_enriched)
        #print("topk_pred_inds:",len(topk_pred_inds),topk_pred_inds)

        list_edge_strs = [] # a list of edge predictions for each mention
        list_edge_info = []
        gold_edge_ind_id = -1
        edge_ind_id = 0
        for edge_ind in topk_pred_inds:
        #for edge_ind in topk_pred_inds_enriched:
            edge_info = dict_ind_edge_json_info[edge_ind]
            if params["filter_by_degree"] and (edge_info["degree"] == 0): # why filter out degree as 0 (i.e. direct relations) TODO? should be to filter out degree as 1. Answer: no need filtering - as the degree value can be different in different paths.
                continue
            
            # store pred edge info
            list_edge_info.append(edge_info)
            # form edge string from edge info
            edge_str = form_edge_str(edge_info)
            
            # calculate wu & palmer score for each pair of pred-edge and gold-edge.
            if (not params["measure_wp"]) or is_complex_concept(edge_info["parent_idx"]) or is_complex_concept(edge_info["child_idx"]) or is_complex_concept(parent_concept) or is_complex_concept(child_concept):
                # score as -1 for complex edges in either the prediction or the gold edges.
                edge_str = '(-1) ' + edge_str
            else:
                pred_edge_tuple = (get_iri_from_SCTID_id(edge_info["parent_idx"],prefix=iri_prefix), get_iri_from_SCTID_id(edge_info["child_idx"],prefix=iri_prefix))
                gold_edge_tuple = (get_iri_from_SCTID_id(parent_concept,prefix=iri_prefix), get_iri_from_SCTID_id(child_concept,prefix=iri_prefix))
                wp_edge_score, dict_iri_to_snd, dict_iri_pair_to_lca = edge_wp_sim(onto_old_taxo,pred_edge_tuple,gold_edge_tuple,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca)
                edge_str = ('(%.5f) ' % wp_edge_score) + edge_str

            # check if tp and add a marking if so
            is_child = edge_info["child_idx"] == child_concept
            is_parent = edge_info["parent_idx"] == parent_concept
            if is_child:
                edge_str = '(**c) ' + edge_str # child is correct, add marking
            if is_parent:
                edge_str = '(**p) ' + edge_str # parent is correct, add marking    
            if is_child and is_parent:
                gold_edge_ind_id = edge_ind_id          
            # add top-k order ind
            edge_str = str(edge_ind_id) + '.' + edge_str
            list_edge_strs.append(edge_str)         

            # construct prompts: subsumption level
            if params["gen_prompts"]:
                prompt_parent, prompt_child = construct_prompt_sub_single_pair(mention, context_left, context_right, edge_info["parent"], edge_info["child"]) #, prompt_path                            
                if prompt_parent != "":
                    # if edge_info["parent_idx"] == parent_concept:
                    #     prompt_parent = prompt_parent + ' ' + '[correct parent]'
                    # filter out the empty prompts due to a TOP parent or a NULL child

                    if prompt_parent in dict_prompt_strs:
                        # update the ctx/doc ind list if the prompt exists
                        dict_prompt_strs = add_dict_tuple_first_and_last_ele_list(dict_prompt_strs,prompt_parent,str(ind),is_parent)
                    else:    
                        dict_prompt_strs[prompt_parent] = ([str(ind)], 
                                                            mention, 
                                                            label_concept_ori, 
                                                            "parent", 
                                                            edge_info["parent"], 
                                                            mention, 
                                                            is_parent,
                                                            )
                if prompt_child != "":
                    # if edge_info["child_idx"] == child_concept:
                    #     prompt_child = prompt_child + ' ' + '[correct child]'
                    # filter out the empty prompts due to a TOP parent or a NULL child
                    if prompt_child in dict_prompt_strs:
                        # update the ctx/doc ind list if the prompt exists
                        dict_prompt_strs = add_dict_tuple_first_and_last_ele_list(dict_prompt_strs,prompt_child,str(ind),is_child)
                    else:    
                        dict_prompt_strs[prompt_child] = ([str(ind)], 
                                                            mention, 
                                                            label_concept_ori, 
                                                            "child", 
                                                            mention, 
                                                            edge_info["child"], 
                                                            is_child,
                                                        )
                    
            # update edge id
            edge_ind_id += 1
            # # up to k edges recommended
            # if len(list_edge_strs) == params["top_k_filtering"]:
            #     break
        
        # construct prompts: edge level
        if params["gen_prompts"]:
            prompt_by_edge = construct_prompt_edge(mention, context_left, context_right, list_edge_info)
            #dict_prompt_strs_by_edge[prompt_by_edge] = 1
            if prompt_by_edge in dict_prompt_strs_by_edge:
                # update the ctx/doc ind list if the prompt exists
                dict_prompt_strs_by_edge = add_dict_tuple_first_and_last_ele_list(dict_prompt_strs_by_edge,prompt_by_edge,str(ind),str(gold_edge_ind_id))
            else:
                dict_prompt_strs_by_edge[prompt_by_edge] = ([str(ind)], 
                                                          mention, 
                                                          label_concept_ori, 
                                                          [str(gold_edge_ind_id)],
                                                        )

        list_mention_w_edge_preds_strs.append(mention_info_json + ':\n\t' + '\n\t'.join(list_edge_strs))
        #list_2d_mention_list_edge_strs.append(list_edge_strs)
        dict_ctx_mention_to_list_2d_edge_strs = add_dict_list(
                                                dict=dict_ctx_mention_to_list_2d_edge_strs,
                                                id=(mention, context_left, context_right,label_concept_ori),
                                                ele=list_edge_strs,
                                                ) # add label_concept_ori to the key tuple so as to make a difference of the same mention machted to several SNOMED CT IDs.
        if child_concept == "SCTID_NULL":
            dict_ctx_mention_to_list_2d_edge_strs_leaf = add_dict_list(
                                                dict=dict_ctx_mention_to_list_2d_edge_strs_leaf,
                                                id=(mention, context_left, context_right,label_concept_ori),
                                                ele=list_edge_strs,
                                                )
        else:
            dict_ctx_mention_to_list_2d_edge_strs_non_leaf = add_dict_list(
                                                dict=dict_ctx_mention_to_list_2d_edge_strs_non_leaf,
                                                id=(mention, context_left, context_right,label_concept_ori),
                                                ele=list_edge_strs,
                                                )

    print('all edges results:')
    eval_edges(dict_ctx_mention_to_list_2d_edge_strs,measure_wp=params["measure_wp"])
    if params["eval_leaf_and_non_leaf_results"]:
        print('leaf edges results:')
        eval_edges(dict_ctx_mention_to_list_2d_edge_strs_leaf,measure_wp=params["measure_wp"])
        print('non-leaf edges results:')
        eval_edges(dict_ctx_mention_to_list_2d_edge_strs_non_leaf,measure_wp=params["measure_wp"])

    output_fn = os.path.join(params["data_path"], "%s-top%d-preds%s.txt" % (params["data_split"],
                                                                    params["top_k_filtering"],
                                                                    '-degree-1' if params["filter_by_degree"] else ''))
    output_to_file(output_fn,'\n'.join(list_mention_w_edge_preds_strs))

    if params["gen_prompts"]:
        # prompt by subsumptions
        output_fn_prompts = os.path.join(params["data_path"], "%s-top%d-preds%s-prompts.txt" % (params["data_split"],
                                                                    params["top_k_filtering"],
                                                                    '-degree-1' if params["filter_by_degree"] else ''))
        output_to_file(output_fn_prompts, '\n'.join(list(dict_prompt_strs.keys())))
        print('prompts in .txt saved to %s' % output_fn_prompts)

        # prompt by edges
        output_fn_prompts_by_edges = os.path.join(params["data_path"], "%s-top%d-preds%s-prompts-by-edges.txt" % (params["data_split"],
                                                                    params["top_k_filtering"],
                                                                    '-degree-1' if params["filter_by_degree"] else ''))
        output_to_file(output_fn_prompts_by_edges, '\n'.join(list(dict_prompt_strs_by_edge.keys())))
        print('prompts in .txt saved to %s' % output_fn_prompts_by_edges)

        # csv form - by subsumptions
        print('len(dict_prompt_strs):',len(dict_prompt_strs))
        output_fn_prompts_csv = output_fn_prompts[:len(output_fn_prompts)-len('.txt')] + '.csv'
        prompt_list = list(dict_prompt_strs.keys())
        ctx_id_list = ['|'.join(tuple_type_answer[0]) for tuple_type_answer in list(dict_prompt_strs.values())] # make it a comma-separated string
        mention_list = [tuple_type_answer[1] for tuple_type_answer in list(dict_prompt_strs.values())]
        snomedct_iri_ori_list = [iri_prefix + tuple_type_answer[2] for tuple_type_answer in list(dict_prompt_strs.values())]
        pc_type_list = [tuple_type_answer[3] for tuple_type_answer in list(dict_prompt_strs.values())]
        parent_list = [tuple_type_answer[4] for tuple_type_answer in list(dict_prompt_strs.values())]
        child_list = [tuple_type_answer[5] for tuple_type_answer in list(dict_prompt_strs.values())]
        anwser_list = [tuple_type_answer[6] for tuple_type_answer in list(dict_prompt_strs.values())]
        
        dict_data_prompts = {'ctx_id': ctx_id_list, 'prompt': prompt_list, 'mention': mention_list, 'snomedct_iri_ori': snomedct_iri_ori_list, 'parent': parent_list, 'child': child_list, 'type': pc_type_list, 'answer': anwser_list}
        df_data_prompts = pd.DataFrame.from_dict(dict_data_prompts)
        df_data_prompts.to_csv(output_fn_prompts_csv, index=True)
        print('prompts in .csv saved to %s' % output_fn_prompts_csv)

        # csv form - by edges
        print('len(dict_prompt_strs_by_edge):',len(dict_prompt_strs_by_edge)) # there can be fewer prompts than mentions (e.g., Disease 604 vs. 605, and CPP 999 vs 1000, as there is a same mention counted twice due to matching to two different SNOMED CT IDs)
        output_fn_prompts_by_edges_csv = output_fn_prompts_by_edges[:len(output_fn_prompts_by_edges)-len('.txt')] + '.csv'
        prompt_edge_list = list(dict_prompt_strs_by_edge.keys())
        ctx_id_list = ['|'.join(tuple_type_answer[0]) for tuple_type_answer in list(dict_prompt_strs_by_edge.values())] # make it a comma-separated string
        mention_list = [tuple_type_answer[1] for tuple_type_answer in list(dict_prompt_strs_by_edge.values())]
        snomedct_iri_ori_list = [iri_prefix + tuple_type_answer[2] for tuple_type_answer in list(dict_prompt_strs_by_edge.values())]
        anwser_list = [','.join(tuple_type_answer[3]) for tuple_type_answer in list(dict_prompt_strs_by_edge.values())]
        
        dict_data_prompts = {'ctx_id': ctx_id_list, 'prompt': prompt_edge_list, 'mention': mention_list, 'snomedct_iri_ori': snomedct_iri_ori_list, 'answer': anwser_list}
        df_data_prompts = pd.DataFrame.from_dict(dict_data_prompts)
        df_data_prompts.to_csv(output_fn_prompts_by_edges_csv, index=True)
        print('prompts in .csv saved to %s' % output_fn_prompts_by_edges_csv)

    if params["measure_wp"]:
        # save shortest node depth (snd) and lowest common ancestor (lca) dicts
        with open(snd_fn, 'wb') as data_f:
            pickle.dump(dict_iri_to_snd, data_f)        
            print('shortest node depth dict stored to',snd_fn)
        with open(lca_fn, 'wb') as data_f:
            pickle.dump(dict_iri_pair_to_lca, data_f)        
            print('lowest common ancestor dict stored to',lca_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="output candidates from bi-encoder")
    parser.add_argument('--data_path', type=str,
                        help="data path of candidates generated by the bi-encoder", 
                        default='') 
    parser.add_argument('--original_data_path', type=str,
                        help="original data path",
                        default="data/MedMentions-preprocessed+/st21pv_syn_attr-edges-NIL") 
    parser.add_argument('--data_split', type=str,
                        help="data split, which is a part of data filename",
                        default="test") 
    parser.add_argument('--data_splits', type=str,
                        help="data splits, which are a part of data filename. Can be separated by comma",
                        default="valid,test")
    parser.add_argument('--ontology_fn', type=str,help="path to the ontology .owl file",default='')
    parser.add_argument('--iri_prefix', type=str,help="http prefix of the iri",default='http://snomed.info/id/')
    parser.add_argument('--edge_catalogue_fn', type=str, 
                        help='filepath to entities to encode (.jsonl file)',
                        default="ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl")
    # parser.add_argument('--top_k_cand_seed', type=int, 
    #                     help='number of seeded candidates (before any candidate enrichment steps) for each mention',
    #                     default="10")
    parser.add_argument('--top_k_filtering', type=int, 
                        help='a filtered number of top-k',
                        default="50")                    
    parser.add_argument('--filter_by_degree', 
                        action="store_true",
                        help='whether to only generate edges with degree of 1, to note that the complex edges are of degree 0',
                        )                    
    parser.add_argument('--gen_prompts', 
                        action="store_true",
                        help='whether to generate prompts to query LMs',
                        )       
    parser.add_argument('--eval_leaf_and_non_leaf_results', 
                        action="store_true",
                        help='whether to separately evaluate leaf edges and non-leaf edges',
                        )
    parser.add_argument('--measure_wp', 
                        action="store_true",
                        help='whether to calculte wu & palmer similarity',
                        )
    args = parser.parse_args()
    print(args)
    params = args.__dict__

    data_split_lists = params["data_splits"].split(',') # param["mode"] as 'train,valid'
    for data_split in data_split_lists:
        new_params = params
        new_params["data_split"] = data_split
        main(new_params)
