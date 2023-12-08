# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.nn.utils.rnn import pad_sequence
import sys

import numpy as np
from itertools import zip_longest
from tqdm import tqdm
import blink.biencoder.data_process as data # here re-use the biencoder's data_process
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_SYN_TAG, ENT_NIL_TAG



def prepare_crossencoder_mentions(
    tokenizer,
    samples,
    max_context_length=32,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    use_context=True,    
):

    context_input_list = []  # samples X 128

    for i, sample in enumerate(tqdm(samples)):
        context_tokens = data.get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
            use_context=use_context,
        )
        if i<=5:
            print("context_tokens in prepare_crossencoder_mentions:",context_tokens)
        tokens_ids = context_tokens["ids"]
        context_input_list.append(tokens_ids)

    context_input_list = np.asarray(context_input_list)
    return context_input_list

def prepare_crossencoder_mentions_w_concept_ori(
    tokenizer,
    samples,
    max_context_length=32,
    mention_key="mention",
    context_key="context",
    label_concept_ori_key="label_concept_ori",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    use_context=True,    
):
    '''
    get the (contextual) mention's token, and also the concept id ori
    '''

    context_input_list = []  # samples X 128
    context_input_ori_list = []
    concept_id_ori_token_ids_list = []
    for i, sample in enumerate(tqdm(samples)):
        context_tokens = data.get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
            use_context=use_context,
        )
        if i<=5:
            print("context_tokens in prepare_crossencoder_mentions:",context_tokens)
        tokens_ids = context_tokens["ids"]
        context_input_list.append(tokens_ids)
        tokens_ids_ori = context_tokens["contextual_ids"]
        context_input_ori_list.append(tokens_ids_ori)

        #label = sample[label_key] # label/entity description
        concept_id_ori = sample[label_concept_ori_key] # the mention's original concept ID in the KB (before KB versioning) - then to be encoded (here using a tokeniser)
        concept_id_ori_token = tokenizer.tokenize(concept_id_ori)
        concept_id_ori_token_ids = tokenizer.convert_tokens_to_ids(concept_id_ori_token)
        # pad the label_concept_ori encoding to the max_context_length as well
        if len(concept_id_ori_token_ids) < max_context_length:
            concept_id_ori_token_ids = concept_id_ori_token_ids + [0]*(max_context_length-len(concept_id_ori_token_ids))
        else:
            concept_id_ori_token_ids = concept_id_ori_token_ids[:max_context_length]
        concept_id_ori_token_ids_list.append(concept_id_ori_token_ids)

    context_input_list = np.asarray(context_input_list)
    context_input_ori_list = np.asarray(context_input_ori_list)
    concept_id_ori_token_ids_list = np.asarray(concept_id_ori_token_ids_list)
    #concept_id_ori_token_ids_list = np.array(list(zip_longest(*concept_id_ori_token_ids_list, fillvalue=0))).T
    
    return context_input_list, context_input_ori_list, concept_id_ori_token_ids_list

def prepare_crossencoder_candidates(
    tokenizer, labels, nns, id2title, id2synonyms, 
    id2text, ori_local_id2wikipedia_id, local_id2wikipedia_id, max_cand_length=128, topk=100, NIL_ent_id=88150, use_extra_id=True,use_NIL_tag=False,use_NIL_desc=False,use_NIL_desc_tag=False,use_synonyms=True,
): # use_extra_id if set as True, then use the extra id as the length of nn (as id starts from 0), otherwise use -1 (the original choice, which is then filtered by filter_crossencoder_tensor_input()).

    START_TOKEN = tokenizer.cls_token
    END_TOKEN = tokenizer.sep_token

    candidate_input_list = []  # samples X topk=10 X 128
    label_input_list = []  # samples
    label_in_KB_input_list = []
    label_NIL_input_list = [] # the one just like label_input_list, but for NIL entities only
    nns_filtered = []
    nns_filtered_in_KB = []
    nns_filtered_NIL = []
    list_is_NIL_labels = [] # a list of boolean - whether each label in 'labels' is NIL.
    idx = 0
    print('topk in prepare_crossencoder_candidates:',topk)
    for ind, (label, nn) in enumerate(zip(labels, nns)):
        candidates = []
        #print('label:',label,'NIL_ent_id:',NIL_ent_id,label == NIL_ent_id)
        if label == NIL_ent_id:
            list_is_NIL_labels.append(True)
        else:
            list_is_NIL_labels.append(False)
        #label ids (or position in nn), if the label is not found in nn, then set the position as -1 (if filtering it later, otherwise as the extra index, len(nn).
        extra_id = len(nn)
        label_id = -1 if not use_extra_id else extra_id
        label_id_in_KB = -1 if not use_extra_id else extra_id
        label_id_NIL = -1 if not use_extra_id else extra_id
        for jdx, candidate_id in enumerate(nn[:topk]):
            
            # create gold label id as the index in the topk predictions in nn, if existed in nn, otherwise as -1. 
            label_concept = ori_local_id2wikipedia_id[int(label)]
            candidate_concept = local_id2wikipedia_id[candidate_id]
            if label_concept == candidate_concept:
                #print('gold label found in nn: label',type(label),label,'candidate_id',jdx,type(candidate_id),candidate_id)
                #output: gold label found in nn: label <class 'numpy.ndarray'> [11747] candidate_id 0 <class 'numpy.int64'> 11747
                label_id = jdx # the index of labels in the crossencoder is its index in the predictions of biencoder, thus changed to between 0 and topk-1
                if label == NIL_ent_id:
                    if ind < 10:
                        print('gold NIL entity found in nn (within first 10 samples)', 'ind:', ind)
                    label_id_NIL = jdx # the index of the NIL entity if it is the gold label
                else:
                    label_id_in_KB = jdx
            synonyms = id2synonyms[candidate_id] if candidate_id in id2synonyms else None        
            if (synonyms is None or synonyms == '') and ind < 10:
               print('cands w/o syns (within cands of first 10 samples):', candidate_id,id2text[candidate_id],id2title[candidate_id], 'ind:', ind)
            rep = data.get_candidate_representation(
                id2text[candidate_id],
                synonyms,
                tokenizer,
                max_cand_length,
                id2title[candidate_id],
                use_NIL_tag=use_NIL_tag,
                use_NIL_desc=use_NIL_desc,
                use_NIL_desc_tag=use_NIL_desc_tag,
                use_synonyms=use_synonyms,
            )
            if ind <=2 and jdx<=2:
                print('rep in prepare_crossencoder_candidates:',rep)
            tokens_ids = rep["ids"]

            assert len(tokens_ids) == max_cand_length
            candidates.append(tokens_ids)

        label_input_list.append(label_id)
        label_in_KB_input_list.append(label_id_in_KB)
        label_NIL_input_list.append(label_id_NIL)
        candidate_input_list.append(candidates)

        # filters mentions in nns by those not having *correct* candidates in each category - only used when *not* keep all mentions for cross-encoders
        if label_id != -1:
            nns_filtered.append(nn)
        if label_id_in_KB != -1:
            nns_filtered_in_KB.append(nn)
        if label_id_NIL != -1:
            nns_filtered_NIL.append(nn)    
            
        idx += 1
        sys.stdout.write("{}/{} \r".format(idx, len(labels)))
        sys.stdout.flush()

    label_input_list = np.asarray(label_input_list)
    label_in_KB_input_list = np.asarray(label_in_KB_input_list)
    label_NIL_input_list = np.asarray(label_NIL_input_list)
    candidate_input_list = np.asarray(candidate_input_list)

    return label_input_list, candidate_input_list, list_is_NIL_labels, label_in_KB_input_list, label_NIL_input_list, nns_filtered, nns_filtered_in_KB, nns_filtered_NIL

def prepare_crossencoder_edge_candidates(
    tokenizer, 
    labels, 
    nns, 
    id2title, 
    id2synonyms, 
    id2text, 
    ori_local_id2wikipedia_id, 
    local_id2wikipedia_id, 

    max_cand_length=128, 
    topk=100, 
    #NIL_ent_id=88150, 
    use_extra_id=True,
    #use_NIL_tag=False,
    #use_NIL_desc=False,
    #use_NIL_desc_tag=False,
    #use_synonyms=True,
): # use_extra_id if set as True, then use the extra id as the length of nn (as id starts from 0), otherwise use -1 (the original choice, which is then filtered by filter_crossencoder_tensor_input()).

    START_TOKEN = tokenizer.cls_token
    END_TOKEN = tokenizer.sep_token

    candidate_input_list = []  # samples X topk=10 X 128

    #label_gold_local_ids_list = []
    label_input_list = []  # samples
    label_in_KB_input_list = []
    label_NIL_input_list = [] # the one just like label_input_list, but for NIL entities only
    nns_filtered = []
    nns_filtered_in_KB = []
    nns_filtered_NIL = []
    list_is_NIL_labels = [] # a list of boolean - whether each label in 'labels' is NIL.
    idx = 0
    print('topk in prepare_crossencoder_candidates:',topk)
    for ind, (label, nn) in enumerate(zip(labels, nns)):
        candidates = []
        #label_gold_local_ids = []    
        #print('label:',label,'NIL_ent_id:',NIL_ent_id,label == NIL_ent_id)
        # if label == NIL_ent_id:
        #     list_is_NIL_labels.append(True)
        # else:
        #     list_is_NIL_labels.append(False)
        list_is_NIL_labels.append(False)
        #label ids (or position in nn), if the label is not found in nn, then set the position as -1 (if filtering it later, otherwise as the extra index, len(nn).
        extra_id = len(nn)
        label_id = -1 if not use_extra_id else extra_id
        label_id_in_KB = -1 if not use_extra_id else extra_id
        label_id_NIL = -1 if not use_extra_id else extra_id
        for jdx, candidate_id in enumerate(nn[:topk]):
            
            # create gold label id as the index in the topk predictions in nn, if existed in nn, otherwise as -1. 
            label_concept = ori_local_id2wikipedia_id[int(label)]
            candidate_concept = local_id2wikipedia_id[candidate_id]
            #print('label_concept:',label_concept,'candidate_concept:',candidate_concept)
            if label_concept == candidate_concept:
                #print('gold label found in nn: label',type(label),label,'candidate_id',jdx,type(candidate_id),candidate_id)
                #output: gold label found in nn: label <class 'numpy.ndarray'> [11747] candidate_id 0 <class 'numpy.int64'> 11747
                label_id = jdx # the index of labels in the crossencoder is its index in the predictions of biencoder, thus changed to between 0 and topk-1
                # if label == NIL_ent_id:
                #     if ind < 10:
                #         print('gold NIL entity found in nn (within first 10 samples)', 'ind:', ind)
                #     label_id_NIL = jdx # the index of the NIL entity if it is the gold label
                # else:
                #     label_id_in_KB = jdx

                # add the gold local_id to label_gold_local_ids
                #label_gold_local_ids.append(candidate_id)
            #synonyms = id2synonyms[candidate_id] if candidate_id in id2synonyms else None        
            # if (synonyms is None or synonyms == '') and ind < 10:
            #     print('cands w/o syns (within cands of first 10 samples):', candidate_id,id2text[candidate_id],id2title[candidate_id], 'ind:', ind)
            parent_title, child_title = id2title[candidate_id]                
            rep = data.get_edge_candidate_representation(
                # id2text[candidate_id],
                # synonyms,
                tokenizer,
                max_cand_length,
                parent_title,
                child_title,
                # use_NIL_tag=use_NIL_tag,
                # use_NIL_desc=use_NIL_desc,
                # use_NIL_desc_tag=use_NIL_desc_tag,
                # use_synonyms=use_synonyms,
            )
            if ind <=2 and jdx<=2:
                print('rep in prepare_crossencoder_candidates:',rep)
            tokens_ids = rep["ids"]

            assert len(tokens_ids) == max_cand_length
            candidates.append(tokens_ids)

        #label_gold_local_ids_list.append(label_gold_local_ids)
        label_input_list.append(label_id)
        label_in_KB_input_list.append(label_id_in_KB)
        label_NIL_input_list.append(label_id_NIL)
        candidate_input_list.append(candidates)

        # filters mentions in nns by those not having *correct* candidates in each category - only used when *not* keep all mentions for cross-encoders
        if label_id != -1:
            nns_filtered.append(nn)
        if label_id_in_KB != -1:
            nns_filtered_in_KB.append(nn)
        if label_id_NIL != -1:
            nns_filtered_NIL.append(nn)    
            
        idx += 1
        sys.stdout.write("{}/{} \r".format(idx, len(labels)))
        sys.stdout.flush()

    label_input_list = np.asarray(label_input_list)
    label_in_KB_input_list = np.asarray(label_in_KB_input_list)
    label_NIL_input_list = np.asarray(label_NIL_input_list)
    candidate_input_list = np.asarray(candidate_input_list)
    # transform the 2D list label_gold_local_ids_list of different lengths into a numpy array (see https://stackoverflow.com/a/63879852/5319143)
    #label_gold_local_ids_list = np.array(list(zip_longest(*label_gold_local_ids_list, fillvalue=-1))).T

    return label_input_list, candidate_input_list, list_is_NIL_labels, label_in_KB_input_list, label_NIL_input_list, nns_filtered, nns_filtered_in_KB, nns_filtered_NIL#, label_gold_local_ids_list

def filter_crossencoder_tensor_input(
    context_input_list, label_input_list, candidate_input_list, list_is_NIL_labels
):
    # remove the - 1 : examples for which gold is not among the candidates
    context_input_list_filtered = [
        x
        for x, y, z, a in zip(context_input_list, candidate_input_list, label_input_list, list_is_NIL_labels)
        if z != -1
    ]
    label_input_list_filtered = [
        z
        for x, y, z, a in zip(context_input_list, candidate_input_list, label_input_list, list_is_NIL_labels)
        if z != -1
    ]
    candidate_input_list_filtered = [
        y
        for x, y, z, a in zip(context_input_list, candidate_input_list, label_input_list, list_is_NIL_labels)
        if z != -1
    ]
    list_is_NIL_labels_filtered = [
        a
        for x, y, z, a in zip(context_input_list, candidate_input_list, label_input_list, list_is_NIL_labels)
        if z != -1
    ]
    return (
        context_input_list_filtered,
        label_input_list_filtered,
        candidate_input_list_filtered,
        list_is_NIL_labels_filtered
    )


def prepare_crossencoder_data(
    tokenizer, samples, labels, nns, id2title, id2synonyms, 
    id2text, ori_local_id2wikipedia_id, local_id2wikipedia_id, max_cand_length=128, topk=100, keep_all=False, filter_within=True, NIL_ent_id=88150, test_NIL_label_only=False, 
    use_NIL_tag=False,use_NIL_desc=False,use_NIL_desc_tag=False,use_synonyms=True,use_context=True
): # filter_within: filter the data which have the correct entity in the candidates generated by bi-encoder

    # encode mentions
    context_input_list = prepare_crossencoder_mentions(tokenizer, samples,use_context=use_context)

    # encode candidates (output of biencoder)
    label_input_list, candidate_input_list,list_is_NIL_labels, label_in_KB_input_list, label_NIL_input_list, nns_filtered, nns_filtered_in_KB, nns_filtered_NIL = prepare_crossencoder_candidates(
        tokenizer, labels, nns, id2title, id2synonyms, 
        id2text, ori_local_id2wikipedia_id, local_id2wikipedia_id, max_cand_length=max_cand_length, topk=topk, NIL_ent_id=NIL_ent_id, use_extra_id=False,use_NIL_tag=use_NIL_tag,use_NIL_desc=use_NIL_desc,use_NIL_desc_tag=use_NIL_desc_tag,use_synonyms=use_synonyms, #use the extra id will change -1 to len(nn)
    )

    if not keep_all:
        # remove examples where the gold entity is not among the candidates
        if test_NIL_label_only is None:
            # this is the original setting: testing all entities (if there is no out-of-KB / NIL entity in the entity catalogue, this is the same as in-KB entities)
            if filter_within:
                (
                    context_input_list,
                    label_input_list,
                    candidate_input_list,
                    list_is_NIL_labels,
                ) = filter_crossencoder_tensor_input(
                    context_input_list, label_input_list, candidate_input_list, list_is_NIL_labels
                )
        elif not test_NIL_label_only:
            # here it tests in-KB entities
            if filter_within:
                (
                    context_input_list,
                    label_in_KB_input_list,
                    candidate_input_list,
                    list_is_NIL_labels,
                ) = filter_crossencoder_tensor_input(
                    context_input_list, label_in_KB_input_list, candidate_input_list, list_is_NIL_labels
                )
        else:            
            # here it tests out-of-KB entities: it uses label_NIL_input_list, the annotation vector of index of gold NIL labels in nn (if existed, otherwise -1)
            if filter_within:
                (
                    context_input_list,
                    label_NIL_input_list,
                    candidate_input_list,
                    list_is_NIL_labels,
                ) = filter_crossencoder_tensor_input(
                    context_input_list, label_NIL_input_list, candidate_input_list, list_is_NIL_labels             
                )
    else:
        # keep all data - regardless of whether bi-encoder has generated the correct candidate or not.
        label_input_list = [0] * len(label_input_list)
        label_in_KB_input_list = [0] * len(label_in_KB_input_list)
        label_NIL_input_list = [0] * len(label_NIL_input_list)        

    context_input = torch.LongTensor(context_input_list)
    label_input = torch.LongTensor(label_input_list)
    label_in_KB_input = torch.LongTensor(label_in_KB_input_list)
    label_NIL_input = torch.LongTensor(label_NIL_input_list)
    #try:
    candidate_input = torch.LongTensor(candidate_input_list)
    #except TypeError:
    #    print('TypeError for candidate_input_list:',candidate_input_list)
    tensor_is_NIL_labels = torch.as_tensor(list_is_NIL_labels).bool()
    print('label_input_list:',label_input_list)
    print('label_input:',label_input)
    print('tensor_is_NIL_labels:',tensor_is_NIL_labels)

    # the label_input to be returned
    if test_NIL_label_only is None:
        label_input_to_return = label_input
        nns_filtered_to_return = nns_filtered        
    elif not test_NIL_label_only:
        label_input_to_return = label_in_KB_input
        nns_filtered_to_return = nns_filtered_in_KB
    else:
        label_input_to_return = label_NIL_input
        nns_filtered_to_return = nns_filtered_NIL

    # get list_is_NIL_labels, storing whether each label in label_XX_input is an NIL entity
    # list_is_NIL_labels = label_NIL_input.cpu().numpy() == -1

    return (
        context_input,
        candidate_input,
        label_input_to_return, 
        nns_filtered_to_return if filter_within else nns,
        tensor_is_NIL_labels, # same shape as label_input_to_return, storing whether the label input is an NIL entity
    )

def prepare_crossencoder_for_insertion_data(
    tokenizer, samples, labels, nns, id2title, id2synonyms, 
    id2text, ori_local_id2wikipedia_id, local_id2wikipedia_id, max_cand_length=128, topk=100, keep_all=False, filter_within=True, test_NIL_label_only=False, use_context=True
    #use_NIL_tag=False,use_NIL_desc=False,use_NIL_desc_tag=False,use_synonyms=True,
): # filter_within: filter the data which have the correct entity in the candidates generated by bi-encoder
    
    '''
        the function called during the final inference (by blink/main_dense_plus.py), if crossencoder is used for re-ranking of the edges as a (multi-label) classfication.
    '''
    # encode mentions and concept_id_ori of the mention
    context_input_list, context_input_ori_list, concept_id_ori_token_ids_list = prepare_crossencoder_mentions_w_concept_ori(tokenizer, samples,use_context=use_context)

    # encode candidates (output of biencoder) 
    label_input_list, candidate_input_list,list_is_NIL_labels, label_in_KB_input_list, label_NIL_input_list, nns_filtered, nns_filtered_in_KB, nns_filtered_NIL = prepare_crossencoder_edge_candidates(
        tokenizer, labels, nns, id2title, id2synonyms, 
        id2text, ori_local_id2wikipedia_id, local_id2wikipedia_id, max_cand_length=max_cand_length, topk=topk, use_extra_id=False,#use_NIL_tag=use_NIL_tag,use_NIL_desc=use_NIL_desc,use_NIL_desc_tag=use_NIL_desc_tag,use_synonyms=use_synonyms, #use the extra id will change -1 to len(nn)
    )

    if not keep_all:
        # remove examples where the gold entity is not among the candidates
        if test_NIL_label_only is None:
            # this is the original setting: testing all entities (if there is no out-of-KB / NIL entity in the entity catalogue, this is the same as in-KB entities)
            if filter_within:
                (
                    context_input_list,
                    context_input_ori_list,
                    label_input_list,
                    candidate_input_list,
                    list_is_NIL_labels,
                    concept_id_ori_token_ids_list
                ) = filter_crossencoder_tensor_input(
                    context_input_list, 
                    context_input_ori_list, 
                    label_input_list, 
                    candidate_input_list, 
                    list_is_NIL_labels,
                    concept_id_ori_token_ids_list
                )
        elif not test_NIL_label_only:
            # here it tests in-KB entities
            if filter_within:
                (
                    context_input_list,
                    context_input_ori_list,
                    label_in_KB_input_list,
                    candidate_input_list,
                    list_is_NIL_labels,concept_id_ori_token_ids_list
                ) = filter_crossencoder_tensor_input(
                    context_input_list, 
                    context_input_ori_list, 
                    label_in_KB_input_list, 
                    candidate_input_list, 
                    list_is_NIL_labels,
                    concept_id_ori_token_ids_list
                )
        else:            
            # here it tests out-of-KB entities: it uses label_NIL_input_list, the annotation vector of index of gold NIL labels in nn (if existed, otherwise -1)
            if filter_within:
                (
                    context_input_list,
                    context_input_ori_list,
                    label_NIL_input_list,
                    candidate_input_list,
                    list_is_NIL_labels,
                    concept_id_ori_token_ids_list
                ) = filter_crossencoder_tensor_input(
                    context_input_list, 
                    context_input_ori_list, 
                    label_NIL_input_list, 
                    candidate_input_list, 
                    list_is_NIL_labels,
                    concept_id_ori_token_ids_list           
                )
    else:
        # keep all data - regardless of whether bi-encoder has generated the correct candidate or not.
        label_input_list = [0] * len(label_input_list)
        label_in_KB_input_list = [0] * len(label_in_KB_input_list)
        label_NIL_input_list = [0] * len(label_NIL_input_list)   
             

    #label_gold_local_ids = torch.LongTensor(label_gold_local_ids_list)
    context_input = torch.LongTensor(context_input_list)
    context_input_ori = torch.LongTensor(context_input_ori_list)
    label_input = torch.LongTensor(label_input_list)
    label_in_KB_input = torch.LongTensor(label_in_KB_input_list)
    label_NIL_input = torch.LongTensor(label_NIL_input_list)
    #try:
    candidate_input = torch.LongTensor(candidate_input_list)
    #except TypeError:
    #    print('TypeError for candidate_input_list:',candidate_input_list)
    tensor_is_NIL_labels = torch.as_tensor(list_is_NIL_labels).bool()
    concept_id_ori_vecs = torch.LongTensor(concept_id_ori_token_ids_list)
    print('label_input_list:',label_input_list)
    print('label_input:',label_input)
    print('tensor_is_NIL_labels:',tensor_is_NIL_labels)

    # the label_input to be returned
    if test_NIL_label_only is None:
        label_input_to_return = label_input
        nns_filtered_to_return = nns_filtered        
    elif not test_NIL_label_only:
        label_input_to_return = label_in_KB_input
        nns_filtered_to_return = nns_filtered_in_KB
    else:
        label_input_to_return = label_NIL_input
        nns_filtered_to_return = nns_filtered_NIL

    # get list_is_NIL_labels, storing whether each label in label_XX_input is an NIL entity
    # list_is_NIL_labels = label_NIL_input.cpu().numpy() == -1

    return (
        context_input,
        context_input_ori,
        candidate_input,
        label_input_to_return, 
        nns_filtered_to_return if filter_within else nns,
        tensor_is_NIL_labels, # same shape as label_input_to_return, storing whether the label input is an NIL entity
        #label_gold_local_ids
        concept_id_ori_vecs, # mention's original concept id in the KB (before KB versioning)
    )