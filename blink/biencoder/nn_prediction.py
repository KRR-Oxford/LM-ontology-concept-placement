# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# adapted by HD, added features for out-of-KB entity detection

import json
import logging #TODO
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats
from rank_bm25 import BM25Okapi # BM25-based candidate generation (replicating Ji et al., 2020)
import numpy as np
from preprocessing.onto_snomed_owl_util import get_SCTID_id_from_iri

def load_fix_emb_preds_in_KB(sample,dict_edge_tuple_to_ind):
    '''
    load the "sample" list, from the output of the fixed embedding approach, from blink/prompting/prompting_embedding_edges.py
    '''
    list_2d_mention_to_edge_inds = []
    for ind, preds_info in enumerate(tqdm(sample)):
        dict_edge_str_to_score = list(preds_info.values())[0]
        list_edge_inds = []
        for edge_str, _ in dict_edge_str_to_score.items():
            edge_eles = edge_str.split(" -> ")
            edge_eles = [get_SCTID_id_from_iri(iri) for iri in edge_eles]
            edge_ele_tuple = tuple(edge_eles)
            if edge_ele_tuple in dict_edge_tuple_to_ind:
                edge_ind = dict_edge_tuple_to_ind[edge_ele_tuple]
                list_edge_inds.append(edge_ind)
            else:
                print("load_fix_emb_preds_in_KB, data row %d pred:" % ind, edge_ele_tuple, ' not in edge catalogue')
                list_edge_inds.append(-1)
        list_2d_mention_to_edge_inds.append(list_edge_inds)
    return list_2d_mention_to_edge_inds

def load_edge_catalogue(edge_catalogue_fn):
    dict_ind_edge_json_info = {}
    dict_edge_tuple_to_ind = {}
    dict_parent_to_child = {}
    dict_child_to_parent = {}
    with open(edge_catalogue_fn,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    for ind, edge_info_json in enumerate(tqdm(doc)):
        edge_info = json.loads(edge_info_json)  
        #edge_str = '%s (%s) -> %s (%s), degree %d' % (edge_info["parent"], edge_info["parent_idx"], edge_info["child"], edge_info["child_idx"], edge_info["degree"])
        parent = edge_info["parent_idx"]
        child =  edge_info["child_idx"]
        edge_tuple = (parent, child)
        dict_ind_edge_json_info[ind] = edge_info
        dict_edge_tuple_to_ind[edge_tuple] = ind

        #if edge_info["degree"] == 0:
        dict_parent_to_child = add_dict_list(dict_parent_to_child,parent,child)
        dict_child_to_parent = add_dict_list(dict_child_to_parent,child,parent)            

    return dict_ind_edge_json_info, dict_edge_tuple_to_ind, dict_parent_to_child, dict_child_to_parent

# add an element to a dict of list of elements for the id
def add_dict_list(dict,id,ele):
    if not id in dict:
        dict[id] = [ele] # one-element list
    else:
        list_ele = dict[id]
        list_ele.append(ele)
        dict[id] = list_ele
    return dict

# def get_list_edge_scores(reranker, context_input, list_edges, cand_encs):
#     '''
#     generate a list of edges scores using the trained bi-encoder, 
    
#     input: 
#     (i) the model
#     (ii) the mention token ids.
#     (iii) the list of edge inds
#     (iv) dict_ind_edge_json_info
#     '''

#     scores = reranker.score_candidate(
#                 context_input, 
#                 None, 
#                 cand_encs=cand_encode_list[src]
#             )
#     pass

def enrich_edge_cands_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds_w_scores,score_over_all_cands,use_leaf_edge_score=False,LEAF_EDGE_SCORE=1000):
    # enrich leaf edges from predicted parents in existing edges
    list_edge_inds_enriched_w_scores = enrich_edge_cands_leaf_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,list_edge_inds_w_scores,score_over_all_cands,use_leaf_edge_score=use_leaf_edge_score,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
    # enrich 2-hop or higher-hop edges from 1-hop edges
    list_edge_inds_enriched_w_scores = enrich_edge_cands_hop_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds_enriched_w_scores,score_over_all_cands,use_leaf_edge_score=use_leaf_edge_score,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
    return list_edge_inds_enriched_w_scores

# def enrich_edge_cands(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds):
#     # enrich leaf edges from predicted parents in existing edges
#     list_edge_inds_enriched = enrich_edge_cands_leaf(dict_ind_edge_json_info,dict_edge_tuple_to_ind,list_edge_inds)
#     # enrich 2-hop or higher-hop edges from 1-hop edges
#     list_edge_inds_enriched = enrich_edge_cands_hop(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds_enriched)
#     return list_edge_inds_enriched

def enrich_edge_cands_leaf_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,list_edge_inds_w_scores,score_over_all_cands,use_leaf_edge_score=False,LEAF_EDGE_SCORE=1000):
    '''
    enrich edge candidates: adding leaf edges if the parent is predicted in an edge
    input: (i) dict of ind edges into json
           (ii) a list of edge indices to be enriched
    '''
    #list_edge_inds_enriched_w_scores = list_edge_inds_w_scores[:]
    #list_edge_inds_enriched = list_edge_inds.copy()
    set_edge_inds_enriched_w_scores = set(list_edge_inds_w_scores)
    for edge_ind, _ in list_edge_inds_w_scores:
        edge_info = dict_ind_edge_json_info[edge_ind]
        if edge_info["child_idx"] != "SCTID_NULL":
            leaf_edge_tuple = (edge_info["parent_idx"],"SCTID_NULL")
            #list_edge_inds_enriched_w_scores = enrich_edge_list_w_scores(dict_edge_tuple_to_ind,list_edge_inds_enriched_w_scores,leaf_edge_tuple,score_over_all_cands,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
            if leaf_edge_tuple in dict_edge_tuple_to_ind: # check if the constructed leaf edge is in the edge catalogue.
                leaf_edge_ind = dict_edge_tuple_to_ind[leaf_edge_tuple]
                leaf_edge_score = getScoreEdge(leaf_edge_tuple,leaf_edge_ind,score_over_all_cands,use_leaf_edge_score,LEAF_EDGE_SCORE)
                set_edge_inds_enriched_w_scores.add((leaf_edge_ind,leaf_edge_score))
            # if leaf_edge_tuple in dict_edge_tuple_to_ind:
            #     leaf_edge_ind = dict_edge_tuple_to_ind[leaf_edge_tuple]
            #     #print("leaf_edge_ind to enrich if new:",leaf_edge_ind)
            #     if not leaf_edge_ind in list_edge_inds_enriched:
            #         list_edge_inds_enriched.append(leaf_edge_ind)
    return list(set_edge_inds_enriched_w_scores)

def enrich_edge_cands_hop_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds_w_scores,score_over_all_cands,use_leaf_edge_score=False,LEAF_EDGE_SCORE=1000):
    '''
    enrich edge candidates: adding 2-degree or higher-degree edges by extending 1-degree edges
    input: (i) dict of ind edges into json
           (ii) a list of edge indices to be enriched
    '''
    #list_edge_inds = [edge_ind for edge_ind, score in list_edge_inds_w_scores]
    #list_edge_inds_enriched_w_scores = list_edge_inds_w_scores[:]
    set_edge_inds_enriched_w_scores = set(list_edge_inds_w_scores)
    for edge_ind, _ in list_edge_inds_w_scores:
        edge_info = dict_ind_edge_json_info[edge_ind]
        #if edge_info["degree"] == 0:
        parent = edge_info["parent_idx"]
        child =  edge_info["child_idx"]
        # enrich <p+,c>
        if parent in dict_child_to_parent:
            list_parents_upper = dict_child_to_parent[parent]
            for parent_upper in list_parents_upper:
                hop_edge_tuple = (parent_upper,child)
                #list_edge_inds_enriched_w_scores = enrich_edge_list_w_scores(dict_edge_tuple_to_ind,list_edge_inds_enriched_w_scores,hop_edge_tuple,score_over_all_cands,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
                if hop_edge_tuple in dict_edge_tuple_to_ind: # check if the constructed edge is in the edge catalogue.
                    hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                    hop_edge_score = getScoreEdge(hop_edge_tuple,hop_edge_ind,score_over_all_cands,use_leaf_edge_score,LEAF_EDGE_SCORE)
                    set_edge_inds_enriched_w_scores.add((hop_edge_ind,hop_edge_score))
                # if hop_edge_tuple in dict_edge_tuple_to_ind:
                #     hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                #     if not hop_edge_ind in list_edge_inds_enriched:
                #         list_edge_inds_enriched.append(hop_edge_ind)
        # enrich <p,c->
        if child in dict_parent_to_child:
            list_children_lower = dict_parent_to_child[child]
            for child_lower in list_children_lower:
                hop_edge_tuple = (parent,child_lower)
                #list_edge_inds_enriched_w_scores = enrich_edge_list_w_scores(dict_edge_tuple_to_ind,list_edge_inds_enriched_w_scores,hop_edge_tuple,score_over_all_cands,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
                if hop_edge_tuple in dict_edge_tuple_to_ind: # check if the constructed edge is in the edge catalogue.
                    hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                    hop_edge_score = getScoreEdge(hop_edge_tuple,hop_edge_ind,score_over_all_cands,use_leaf_edge_score,LEAF_EDGE_SCORE)
                    set_edge_inds_enriched_w_scores.add((hop_edge_ind,hop_edge_score))
                # if hop_edge_tuple in dict_edge_tuple_to_ind:
                #     hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                #     if not hop_edge_ind in list_edge_inds_enriched:
                #         list_edge_inds_enriched.append(hop_edge_ind)
        # enrich <p+,c-> (although usually with degree more than 2)
        if parent in dict_child_to_parent:
            list_parents_upper = dict_child_to_parent[parent]
            if child in dict_parent_to_child:
                list_children_lower = dict_parent_to_child[child]
                for parent_upper in list_parents_upper:
                    for child_lower in list_children_lower:
                        hop_edge_tuple = (parent_upper,child_lower)
                        #list_edge_inds_enriched_w_scores = enrich_edge_list_w_scores(dict_edge_tuple_to_ind,list_edge_inds_enriched_w_scores,hop_edge_tuple,score_over_all_cands,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
                        if hop_edge_tuple in dict_edge_tuple_to_ind: # check if the constructed edge is in the edge catalogue.
                            hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                            hop_edge_score = getScoreEdge(hop_edge_tuple,hop_edge_ind,score_over_all_cands,use_leaf_edge_score,LEAF_EDGE_SCORE)
                            set_edge_inds_enriched_w_scores.add((hop_edge_ind,hop_edge_score))
                        # if hop_edge_tuple in dict_edge_tuple_to_ind:
                        #     hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                        #     if not hop_edge_ind in list_edge_inds_enriched:
                        #         list_edge_inds_enriched.append(hop_edge_ind)
        
    return list(set_edge_inds_enriched_w_scores)

# def enrich_edge_list_w_scores(dict_edge_tuple_to_ind, list_edge_inds_w_scores, new_edge_tuple,score_over_all_cands,LEAF_EDGE_SCORE=1000,display=False):
#     '''
#     enrich the list_edge_inds with a new edge, if it is in the edge catalogue
#     '''
#     list_edge_inds = [edge_ind for edge_ind, _ in list_edge_inds_w_scores]
#     set_edge_inds = set(list_edge_inds)
#     if new_edge_tuple in dict_edge_tuple_to_ind:
#         new_edge_ind = dict_edge_tuple_to_ind[new_edge_tuple]
#         if not new_edge_ind in set_edge_inds:
#             if display:
#                 print("enrich new edge:",new_edge_ind,new_edge_tuple)
#             #get score
#             if not isLeafEdge(new_edge_tuple):
#                 new_edge_score = score_over_all_cands[new_edge_ind]
#             else:
#                 new_edge_score = LEAF_EDGE_SCORE
#             list_edge_inds_w_scores.append((new_edge_ind, new_edge_score))
#     return list_edge_inds_w_scores

def getScoreEdge(new_edge_tuple,new_edge_ind,score_over_all_cands,use_leaf_edge_score=False,LEAF_EDGE_SCORE=1000):
    #get score
    if not use_leaf_edge_score:
        return score_over_all_cands[new_edge_ind]
    if not isLeafEdge(new_edge_tuple):
        new_edge_score = score_over_all_cands[new_edge_ind]
    else:
        new_edge_score = LEAF_EDGE_SCORE
    return new_edge_score

def isLeafEdge(edge_tuple):
    '''
    check whether the edge is a leaf edge (i.e., parent -> NULL)
    '''
    return edge_tuple[1].endswith("_NULL")

# function adapted from https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-32.php
def is_Sublist(s, l):
	sub_set = False
	if s == []:
		sub_set = True
	elif s == l:
		sub_set = True
	elif len(s) > len(l):
		sub_set = False
	else:
		for i in range(len(l)):
			if l[i] == s[0]:
				n = 1
				while (n < len(s)) and ((i+n) < len(l)) and (l[i+n] == s[n]):
					n += 1
				
				if n == len(s):
					sub_set = True
	return sub_set

# count the set diff between two lists, considering both directions if chosen the pair comparision, otherwise just count those in the first set but not in the second set; in all cases, clip the count to 0 if below 0.
def count_list_set_diff(list_a,list_b,pair_comparison=True):
    if pair_comparison:
        return max(0,len(set(list_a) - set(list_b))) + max(0,len(set(list_b) - set(list_a)))
    else:
        return max(0,len(set(list_a) - set(list_b)))

# form a set of features per mention of whether the mention has no, one, or several matching names in the entities through string matching (exact or fuzzy) (Rao, McNamee, and Dredze 2013; McNamee et al. 2009)
# we only consider the entities from the candidate list
# input: (i) list_mention_input, the list of sub-token ids in a mention (as formed in data_process.get_mention_representation())
#        (ii) list_2d_label_input, the list of label titles + descriptions (as formed in data_process.get_context_representation()), where each is a list of sub-token ids; this list can be either the full candidate list or the top-k candidate list after the candidate generation stage
def get_is_men_str_matchable_features(list_mention_input,list_2d_label_input,index_title_special_token=3,fuzzy_tolerance=2):
    #print("mention_input:",len(list_mention_input),list_mention_input)
    #print("label_input:",list_2d_label_input,len(list_2d_label_input),len(list_2d_label_input[0]))

    #for mention_sub_token_list in list_2d_mention_input:
    # clean mention input
    mention_sub_token_list = [sub_token_id for sub_token_id in list_mention_input if sub_token_id >= 3]
    mention_matched_exact = 0
    mention_matched_exact_w_desc = 0
    mention_matched_fuzzy = 0
    mention_matched_fuzzy_w_desc = 0
    for label_sub_token_list in list_2d_label_input:
        # get list of *title* sub token ids 
        label_tit_sub_token_list = get_title_ids_from_label_sub_token_list(
                                        label_sub_token_list,
                                        index_title_special_token=index_title_special_token)
        
        # exact matching
        if mention_matched_exact < 2:
            if mention_sub_token_list == label_tit_sub_token_list:
                mention_matched_exact += 1

        if mention_matched_exact_w_desc < 2:
            if is_Sublist(mention_sub_token_list,label_sub_token_list):
                mention_matched_exact_w_desc += 1
        
        # fuzzy matching
        if mention_matched_fuzzy < 2:
            num_set_diff_men_tit = count_list_set_diff(mention_sub_token_list,label_tit_sub_token_list,pair_comparison=True)
            if num_set_diff_men_tit <= fuzzy_tolerance:
                mention_matched_fuzzy += 1

        if mention_matched_fuzzy_w_desc < 2:
            num_set_diff_men_tit_desc = count_list_set_diff(mention_sub_token_list,label_sub_token_list,pair_comparison=False)
            if num_set_diff_men_tit_desc <= fuzzy_tolerance:
                mention_matched_fuzzy_w_desc += 1
        
    mention_matchable_exact = mention_matched_exact > 0
    mention_matchable_exact_w_desc_one = mention_matched_exact_w_desc == 1
    mention_matchable_exact_w_desc_several = mention_matched_exact_w_desc > 1
    mention_matchable_fuzzy_one = mention_matched_fuzzy == 1
    mention_matchable_fuzzy_several = mention_matched_fuzzy > 1
    mention_matched_fuzzy_w_desc_one = mention_matched_fuzzy_w_desc == 1
    mention_matchable_fuzzy_w_desc_several = mention_matched_fuzzy_w_desc > 1

    is_men_str_matchable_features = [mention_matchable_exact,mention_matchable_exact_w_desc_one,mention_matchable_exact_w_desc_several,mention_matchable_fuzzy_one,mention_matchable_fuzzy_several,mention_matched_fuzzy_w_desc_one,mention_matchable_fuzzy_w_desc_several]

    #print('is_men_str_matchable_features:',is_men_str_matchable_features)
    return is_men_str_matchable_features

# normalise to the "syn as canonical ent" row-id from the "syn as ent" row-id
def _normalise_local_id(local_id,local_id2wikipedia_id,wikipedia_id2local_id):
    if local_id in local_id2wikipedia_id:
        local_id_normalised = wikipedia_id2local_id[local_id2wikipedia_id[local_id]]
    else:
        local_id_normalised = local_id
    return local_id_normalised

# normalise to the original ent row-id from the "syn as ent" row-id 
def _normalise_to_ori_local_id(local_id,local_id2wikipedia_id,wikipedia_id2original_local_id):
    return _normalise_local_id(local_id,local_id2wikipedia_id,wikipedia_id2original_local_id)

def _aggregating_indices_synonyms(indicies_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k):
    #aggregating indicies of synonyms (by maximum)
    indicies_per_datum_ori = indicies_per_datum[:]
    #normalise the indicies to those of the canonical names (not synonyms).
    indicies_per_datum = [_normalise_local_id(int(indice),local_id2wikipedia_id,wikipedia_id2local_id) for indice in indicies_per_datum]
    #remove duplicates in normalised indicies
    indicies_per_datum = list(dict.fromkeys(indicies_per_datum))[:top_k]    
    if len(indicies_per_datum) != top_k:
        print('indicies_per_datum:',len(indicies_per_datum),'top_k:',top_k)
        print('ori->new indicies_per_datum:',indicies_per_datum_ori,'->',indicies_per_datum)
    indicies_per_datum = np.array(indicies_per_datum)
    return indicies_per_datum

# we use max (see the function above) as averaging is not working well due to the discrepancy in scores.
def _aggregating_indices_synonyms_ave(indicies_per_datum,scores_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k):
    #aggregating indicies of synonyms (by average among top topk*10)
    #normalise the indicies to those of the canonical names (not synonyms).
    indicies_per_datum = [wikipedia_id2local_id[local_id2wikipedia_id[int(indice)]] for indice in indicies_per_datum]
    #get dict of indice to list of scores of all same entities (inc. synonyms)
    dict_indice_to_score = {}
    #print('scores_per_datum:',scores_per_datum)
    #normalise the score with softmax
    #scores_per_datum = _softmax(scores_per_datum)
    #print('scores_per_datum:',scores_per_datum)
    for indice, score in zip(indicies_per_datum,scores_per_datum):
        if not indice in dict_indice_to_score:
            dict_indice_to_score[indice] = [score]
        else:
            list_scores_indice = dict_indice_to_score[indice]
            list_scores_indice.append(score)
            dict_indice_to_score[indice] = list_scores_indice
    #average the list of scores to one and update the dict
    for indice, list_of_scores in dict_indice_to_score.items():
        score_ave = np.mean(np.array(list_of_scores))
        dict_indice_to_score[indice] = score_ave
    #print('dict_indice_to_score:',dict_indice_to_score)
    #rank by value (averaged scores)    
    dict_indice_to_score = {k: v for k, v in sorted(dict_indice_to_score.items(), key=lambda item: item[1])}    
    #output top_k
    indicies_per_datum = list(dict_indice_to_score.keys())[:top_k]
    assert len(indicies_per_datum) == top_k
    indicies_per_datum = np.array(indicies_per_datum)
    return indicies_per_datum

def get_title_ids_from_label_sub_token_list(label_sub_token_list,index_title_special_token=3):
    label_sub_token_list = label_sub_token_list[1:-1]
    # get the position of title mark 
    if index_title_special_token in label_sub_token_list:
        pos_title_mark = label_sub_token_list.index(index_title_special_token)
    else:
        # no title mark, thus everything is in title
        print('get_is_men_str_matchable_features(): no title mark found for ', label_sub_token_list)
        pos_title_mark = len(label_sub_token_list) 
    # get title sub tokens as a list
    label_tit_sub_token_list = label_sub_token_list[:pos_title_mark]
    # get desc sub tokens as a list
    #label_desc_sub_token_list = label_sub_token_list[pos_title_mark+1:]
    return label_tit_sub_token_list

# def get_list_2d_title_ids_from_candidate_pool(list_2d_canditate_pool,index_title_special_token=3):
#     list_2d_candidate_title_ids = []
#     for label_sub_token_list in list_2d_canditate_pool:
#         label_tit_sub_token_list = get_title_ids_from_label_sub_token_list(label_sub_token_list)
#         list_2d_candidate_title_ids.append(label_tit_sub_token_list)
#     return list_2d_candidate_title_ids

# get ranking indices with BM25
def get_ranking_indices_w_BM25(list_mention_input,list_2d_candidate_title_ids,topn=100,index_title_special_token=3):
    #clean the ids by removing special token ids and padding ids
    mention_sub_token_list = [sub_token_id for sub_token_id in list_mention_input if sub_token_id >= 3]
    # list_2d_candidate_title_ids = []
    # for label_sub_token_list in list_2d_canditate_pool:
    #     # clean label ids
    #     label_sub_token_list = label_sub_token_list[1:-1]
    #     # get the position of title mark 
    #     if index_title_special_token in label_sub_token_list:
    #         pos_title_mark = label_sub_token_list.index(index_title_special_token)
    #     else:
    #         # no title mark, thus everything is in title
    #         print('get_ranking_indices_w_BM25(): no title mark found for ', label_sub_token_list)
    #         pos_title_mark = len(label_sub_token_list) 
    #     # get title sub tokens as a list
    #     label_tit_sub_token_list = label_sub_token_list[:pos_title_mark]
    #     # get desc sub tokens as a list
    #     #label_desc_sub_token_list = label_sub_token_list[pos_title_mark+1:]
    #     list_2d_candidate_title_ids.append(label_tit_sub_token_list)
    label_tit_sub_word_id_bm25 = BM25Okapi(list_2d_candidate_title_ids)
    scores = label_tit_sub_word_id_bm25.get_scores(mention_sub_token_list)
    topn_indicies = np.argsort(scores)[::-1][:topn]
    topn_scores = scores[topn_indicies]
    #topn_by_subwords_ids = label_tit_sub_word_id_bm25.get_top_n(mention_sub_token_list, list_2d_candidate_title_ids, n=topn)
    #topn_indicies = [list_2d_candidate_title_ids.index(label_title_subword_ids) for label_title_subword_ids in topn_by_subwords_ids]
    #scores = []

    #print('topn_indicies:',topn_indicies, type(topn_indicies))
    #print('topn_scores:',topn_scores, type(topn_scores))
    return topn_indicies.tolist(), topn_scores

# some changes applied based on https://github.com/facebookresearch/BLINK/issues/115#issuecomment-1119282640
# this function generates new data to train cross-encoder, from the candidates (or nns) generated by biencoder.
def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool, # the candidate token ids (w [SYN]-concatenated in the syn mode), this is only used for output
    cand_encode_list,
    wikipedia_id2local_id,
    local_id2wikipedia_id,
    silent,
    logger,
    top_k=100,
    is_zeshel=False,
    save_predictions=False,
    save_true_predictions_only=False,
    add_NIL=False, # add NIL to the last element of the biencoder predicted entity indicies, if NIL was not predicted
    NIL_ent_id=88150,
    use_fix_embs=False,
    sample_fix_emb_preds_in_KB=None,
    use_BM25=False,
    candidate_pool_for_BM25=None, # the candidate token ids for BM25 (w syn as entities in the syn mode), this is used for BM25 only for searching ents from ments.
    get_is_men_str_mat_fts=False,
    index_title_special_token=3,
    edge_cand_enrich=False, # whether to enrich edge candidates
    edge_catalogue_fn="", # edge catalogue path, only used when edge_cand_enrich is True 
    top_k_cand_seed=10, # top-k candidate seed before edge candidate enrichment
    use_leaf_edge_score=False,
    LEAF_EDGE_SCORE=1000,
    edge_ranking_by_score=True,
    #aggregating_factor=20, # for top_k entities & synonyms aggregation (for synonyms as entities)
):
    reranker.model.eval()
    device = reranker.device
    #print('device:',device)
    logger.info("Getting top %d predictions." % top_k)
    if edge_cand_enrich:
        logger.info("Enriched from top %d predictions." % top_k_cand_seed)
        
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_row_ids = []
    nn_context = []
    nn_context_ori = []
    nn_candidates = []
    #nn_label_text = []
    nn_labels = []
    nn_labels_is_NIL = []
    nn_concept_id_ori_vecs = []
    nn_entity_inds = []
    nn_is_mention_str_matchable_fts = []
    nn_worlds = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        candidate_pool_for_BM25 = [candidate_pool_for_BM25]
        cand_encode_list = [cand_encode_list]

    # process candidate_pool_for_BM25
    if use_BM25:
        list_2d_canditate_pool = candidate_pool_for_BM25[0].cpu().tolist()
        # by list comprehension (instead of get_list_2d_title_ids_from_candidate_pool())
        list_2d_candidate_title_ids = [get_title_ids_from_label_sub_token_list(label_sub_token_list,index_title_special_token=index_title_special_token) for label_sub_token_list in list_2d_canditate_pool]
    
    #if edge_cand_enrich:
    dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent = load_edge_catalogue(edge_catalogue_fn)
    
    # process fix_emb_preds_in_KB_fn for the fixed embedding approach
    if use_fix_embs:
        # get indicies from the file output (through blink/prompting/prompting_embedding_edges.py)
        list_2d_mention_to_edge_inds = load_fix_emb_preds_in_KB(sample_fix_emb_preds_in_KB,dict_edge_tuple_to_ind)
        print('list_2d_mention_to_edge_inds:',len(list_2d_mention_to_edge_inds))

    #logger.info("World size : %d" % world_size)
    #print('candidate_pool:',candidate_pool)
    #print('candidate_pool:',candidate_pool,len(candidate_pool),candidate_pool[0].size())
    #1 torch.Size([88151, 128])
    #print('cand_encode_list:',cand_encode_list)
    '''
    candidate_pool: [tensor([[  101, 13878,  1010,  ...,     0,     0,     0],
        [  101, 21419, 13675,  ...,     0,     0,     0],
        [  101, 21419,  9253,  ...,     0,     0,     0],
        ...,
        [  101, 18404, 10536,  ...,     0,     0,     0],
        [  101,  1040,  7274,  ...,     0,     0,     0],
        [  101,  9152,  2140,  ...,     0,     0,     0]])]
    cand_encode_list: [tensor([[ 0.2102,  0.1818, -0.3594,  ..., -0.3182, -0.8104, -0.1960],
        [ 0.0642,  0.2399, -0.0787,  ..., -0.4488, -0.6695, -0.4290],
        [-0.0145,  0.1526, -0.0516,  ..., -0.4228, -0.4721, -0.2667],
        ...,
        [ 0.5740, -0.0637, -0.1766,  ...,  0.2560, -0.3511, -0.2073],
        [ 0.3947,  0.1827,  0.0299,  ...,  0.0638, -0.5476, -0.0607],
        [ 0.1874,  0.0835, -0.0825,  ..., -0.1674, -0.6785, -0.1951]])]
    '''

    for i in range(world_size):
        stats[i] = Stats(top_k)
    
    # get dict of wikipedia_id2original_local_id
    wikipedia_id2_ori_local_id = {k:ori_id for ori_id, (k,v) in enumerate(wikipedia_id2local_id.items())}
    #print('wikipedia_id2_ori_local_id:',list(wikipedia_id2_ori_local_id.items())[:100])

    oid = -1 # object id or row id (will add one each time before recorded, so starting from -1)
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        #context_input, _, srcs, label_ids = batch
        if is_zeshel:
            mention_input, context_input, contextual_vecs, label_text_input, srcs, label_ids, is_label_NIL, concept_id_ori_vecs = batch
        else:
            mention_input, context_input, contextual_vecs, label_text_input, label_ids, is_label_NIL, concept_id_ori_vecs = batch
            # here you can also know whether the label is NIL - may be useful later
            srcs = torch.tensor([0] * context_input.size(0), device=device)    
        src = srcs[0].item()
        #print('src:',src)

        if not use_BM25:
            cand_encode_list[src] = cand_encode_list[src].to(device)
            scores = reranker.score_candidate(
                context_input, 
                None, 
                #cand_encs=cand_encode_list[src].to(device)
                cand_encs=cand_encode_list[src]
            )
            #values, indicies = scores.topk(top_k*aggregating_factor)
            values, indicies = scores.topk(top_k)
            scores = scores.data.cpu().numpy()
            indicies = indicies.data.cpu().numpy()
            #print('indicies before aggregation:',indicies)
            # aggregating results
            #indicies = np.array([_aggregating_indices_synonyms(indicies_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k) for indicies_per_datum in indicies])
            #print('indicies after aggregation:',indicies)
            # # add NIL to the end
            # if add_NIL:
            #     for ind, indices_per_datum in enumerate(indicies):
            #         if not NIL_ent_id in indices_per_datum:
            #             indicies[ind][-1] = NIL_ent_id
            #indicies_np = indicies.data.cpu().numpy()
            #nn_entity_inds.extend(indicies_np)
        
            #print('values in get_topk_predictions():',values)
            #print('indicies in get_topk_predictions():',indicies)
        #print('label_ids:',label_ids) # the problem is that this is the label_ids, but the indices are the indexes in the entity catalogue.
        #print('is_label_NIL:',is_label_NIL)
        old_src = src
        
        # loop over items in a batch
        #print('context_input.size(0):',context_input.size(0))
        for i in range(context_input.size(0)):
            oid += 1
            
            if use_BM25:
                # get indicies through BM25 - the candidate pool (sub-tokens, w or w/o synonyms, according to the input) is used to match with mention sub-tokens
                inds, _ = get_ranking_indices_w_BM25(mention_input[i].cpu().tolist(),list_2d_candidate_title_ids,topn=top_k*aggregating_factor,
                index_title_special_token=index_title_special_token)
                # aggregating results
                inds = np.array(_aggregating_indices_synonyms(inds,local_id2wikipedia_id,wikipedia_id2local_id,top_k))
                #inds = inds.tolist()
                # add NIL to the end            
                if add_NIL:
                    if not NIL_ent_id in inds:
                        inds[-1] = NIL_ent_id

                #inds = torch.tensor(inds)
            elif use_fix_embs:
                if not oid < len(list_2d_mention_to_edge_inds):
                    print('oid out of mention id in list_2d_mention_to_edge_inds:',oid)
                list_edge_inds = list_2d_mention_to_edge_inds[oid]
                inds = np.array(list_edge_inds)
            else:                    
                if srcs[i] != old_src:
                    src = srcs[i].item()
                    # not the same domain, need to re-do
                    scores = reranker.score_candidate(
                        context_input[[i]], 
                        None,
                        cand_encs=cand_encode_list[src].to(device)
                    )
                    _, inds = scores.topk(top_k)
                    inds = inds[0]
                    scores = scores.data.cpu().numpy()
                
                inds = indicies[i]
                score_over_all_cands = scores[i]

            #if i<=3:
            #   print('cand inds (first %d/3):' % i, inds)
            
            # edge candidate enrichment - if chose to
            if edge_cand_enrich and (not use_fix_embs): # (the fixed embedding appraoch uses its own edge enrichment implemented in blink/prompting/prompting_embedding_edges.py)
                inds_seed = inds[:top_k_cand_seed]
                inds_non_seed = inds[top_k_cand_seed:]

                list_edge_inds_seed_w_scores = [(ind_seed, score_over_all_cands[ind_seed]) for ind_seed in inds_seed]
                list_inds_enriched_w_scores = enrich_edge_cands_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds_seed_w_scores,score_over_all_cands=score_over_all_cands,use_leaf_edge_score=use_leaf_edge_score,LEAF_EDGE_SCORE=LEAF_EDGE_SCORE)
                
                # enriched with original non-seeded part (from inds_non_seed) if not enough cands after the enrichment
                if len(list_inds_enriched_w_scores) < top_k: 
                    list_inds_enriched = [ind for ind,_ in list_inds_enriched_w_scores]
                    for ind_ in inds_non_seed:
                        if not ind_ in list_inds_enriched:
                            list_inds_enriched_w_scores.append((ind_,score_over_all_cands[ind_])) #list_inds_enriched = list_inds_enriched + [float("nan")] * (top_k - len(list_inds_enriched))
                            #set_inds_enriched.add(ind_)
                            if len(list_inds_enriched_w_scores) == top_k:
                                break
                
                if edge_ranking_by_score:
                    # rank the list_inds_enriched
                    #list_inds_enriched_w_scores = [(ind, score_over_all_cands[ind]) for ind in list_inds_enriched]
                    #print("list_inds_enriched_w_scores:",list_inds_enriched_w_scores)                
                    list_inds_enriched_w_scores = sorted(list_inds_enriched_w_scores, key=lambda x: x[1],reverse=True) #sort the edges by score
                    if oid < 10:
                        print("list_inds_enriched_w_scores, sorted:",list_inds_enriched_w_scores)

                # get the top-k after the ranking
                #list_inds_enriched = list(set_inds_enriched) 
                list_inds_enriched_w_scores_topk = list_inds_enriched_w_scores[:top_k]
                list_inds_enriched_topk = [ind for ind, _ in list_inds_enriched_w_scores_topk]
                #set_inds_enriched = set(list_inds_enriched)                    
                #list_inds_enriched = list(set_inds_enriched)
                inds = np.array(list_inds_enriched_topk)
            pointer = -1
            is_pointer_NIL = False
            label_id = label_ids[i].item()
            label_id_normalised = _normalise_local_id(label_id,local_id2wikipedia_id,wikipedia_id2local_id)

            #pad inds to the length of topk - if not enough predictions (for the fix emb setting)
            if use_fix_embs:
                inds = np.pad(inds, 
                              (0, top_k - len(inds)), 
                              mode='constant', 
                              constant_values=-1)

            for j in range(top_k):
                if j < len(inds):       
                    if inds[j].item() == label_id_normalised: #label_ids[i].item():
                        pointer = j
                        is_pointer_NIL = is_label_NIL[i]
                        break
                else:
                    print('less than top-%d inds predicted: %d' % (top_k,len(inds)))
                    #inds = np.append(inds,-1)
                    break  
            stats[src].add(pointer)
            
            #print('save_predictions:',save_predictions)            
            #print('save_true_predictions_only:',save_true_predictions_only)            
            if pointer == -1 and save_true_predictions_only: # not save predictions when the gold is not predicted in the top-k; otherwise, save all predictions
                continue
            if not save_predictions:
                continue
            
            # get current, topk candidates' token ids
            # transform inds (with syns as rows counted) to original inds (w only each entity as a row)
            inds_ori_local_id = [_normalise_to_ori_local_id(ind,local_id2wikipedia_id,wikipedia_id2_ori_local_id) for ind in inds]
            #if i<=3:
            #   print('inds_ori_local_id (first %d/3):' % i, inds_ori_local_id)
            #cur_candidates = candidate_pool[src][inds]
            cur_candidates = candidate_pool[srcs[i].item()][inds_ori_local_id]
            #print('cur_candidates:',cur_candidates,cur_candidates.size())

            # get features: does the mention have matching name in the entities
            #print('label_text_input:',len(label_text_input),label_text_input.size()) #label_text_input: 32 torch.Size([32, 128])
            #is_men_str_matchable_fts = get_is_men_str_matchable_features(mention_input[i].cpu().tolist(), label_text_input.cpu().tolist()) # search in a batch of labels
            #is_men_str_matchable_fts = get_is_men_str_matchable_features(mention_input[i].cpu().tolist(), candidate_pool[0].cpu().tolist()) # search in all labels from the entity catelogue

            #if not use_BM25:
            if get_is_men_str_mat_fts:
                is_men_str_matchable_fts = get_is_men_str_matchable_features(mention_input[i].cpu().tolist(), cur_candidates.cpu().tolist(),index_title_special_token=index_title_special_token) # search in the topk labels
                nn_is_mention_str_matchable_fts.append(is_men_str_matchable_fts)
            #else:
            #    is_men_str_matchable_fts = []
            # add examples in new_data
            nn_row_ids.append(oid)
            nn_context.append(context_input[i].cpu().tolist())
            nn_context_ori.append(contextual_vecs[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            #nn_label_text.append(label_text_input[i].cpu().tolist())
            nn_labels.append(pointer)
            nn_labels_is_NIL.append(is_pointer_NIL)
            nn_concept_id_ori_vecs.append(concept_id_ori_vecs[i].cpu().tolist())
            nn_entity_inds.append(inds)#(inds.data.cpu().numpy())            
            nn_worlds.append(src)
            
    # the stats and res below are only for zero-shot senario.
    if is_zeshel:
        res = Stats(top_k)
        for src in range(world_size):
            if stats[src].cnt == 0:
                continue
            if is_zeshel:
                logger.info("In world " + WORLDS[src])
            output = stats[src].output()
            logger.info(output)
            res.extend(stats[src])

        logger.info(res.output())

    nn_row_ids = torch.LongTensor(nn_row_ids)
    nn_context = torch.LongTensor(nn_context)
    nn_context_ori = torch.LongTensor(nn_context_ori)
    nn_candidates = torch.LongTensor(nn_candidates)
    #nn_label_text = torch.LongTensor(nn_label_text)
    nn_labels = torch.LongTensor(nn_labels)
    nn_labels_is_NIL = torch.Tensor(nn_labels_is_NIL).bool()
    nn_concept_id_ori_vecs = torch.Tensor(nn_concept_id_ori_vecs)

    if get_is_men_str_mat_fts:
        nn_is_mention_str_matchable_fts = torch.Tensor(nn_is_mention_str_matchable_fts)
    nn_data = {
        'row_ids': nn_row_ids,
        'context_vecs': nn_context,
        'contextual_vecs': nn_context_ori,
        'candidate_vecs': nn_candidates,
        #'label_text_vecs': nn_label_text,
        'labels': nn_labels,
        'labels_is_NIL': nn_labels_is_NIL, # whether the label is NIL - bool type, Tensor
        'concept_id_ori_vecs': nn_concept_id_ori_vecs,
        'entity_inds': nn_entity_inds, # the predicted entity indices from the bi-encoder, a list of np_arrays
        'mention_matchable_fts': nn_is_mention_str_matchable_fts if get_is_men_str_mat_fts else None, # mention matchable features
    }
    print('nn_data[\'labels\']:',nn_data['labels'])
    num_tp = len(nn_data['labels'][nn_data['labels'] != -1]) # get the ones not -1, i.e. gold in topk candidates
    num_ori_data = len(train_dataloader.dataset)
    logger.info('num of nn_data: %d' % len(nn_data['labels']))
    logger.info('biencoder recall@k: %.2f (%d/%d)' % (float(num_tp)/num_ori_data, num_tp, num_ori_data))
    #print('num of nn_data:',len(nn_data['entity_inds']))
    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data