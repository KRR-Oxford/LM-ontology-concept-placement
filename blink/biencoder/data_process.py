# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

#from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG, ENT_SYN_TAG, ENT_NIL_TAG, ENT_PARENT_TAG, ENT_CHILD_TAG, ENT_TOP_TAG, ENT_NULL_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    use_context=True, # if using context, default as True
    #max_mention_length=20, # only for extra feature calculation. Deprecated, just using max_seq_length
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    mention_tokens_ori = mention_tokens[:]
    #if use_context:
    #get contextual mention tokens as context_tokens (and mention tokens alone, which can be updated or shrinked due to mention being too long, over the quota of contextual mention tokens)
    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    mention_quota = len(mention_tokens)
    left_add = len(context_left)
    right_add = len(context_right)
    #adjust the left (and right) quota based on the other to fill the max_seq_length
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add 
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add
    #the case when no left quota or right quota remained (they are 0 or negative) - here mention tokens might get shrinked
    if left_quota < 0:
        #print('left_quota:', left_quota)
        mention_quota = mention_quota + left_quota
        print('mention shrinked by %d:' % -left_quota, mention_tokens)
        left_quota = 0
    if right_quota < 0:
        #print('right_quota:',right_quota)
        mention_quota = mention_quota + right_quota
        print('mention shrinked by %d:' % -right_quota, mention_tokens)
        right_quota = 0        
    mention_tokens = mention_tokens[:mention_quota]        
    # if context_left[-left_quota:] != context_left[len(context_left)-left_quota:]:
    #     print('context_left:',context_left)
    #     print('left_quota:',left_quota)
    #     print("context_left[-left_quota:]",context_left[-left_quota:])
    #     print("context_left[len(context_left)-left_quota:]",context_left[len(context_left)-left_quota:])
    #     '''
    #         context_left: ['medication', ',', 'rehabilitation', 'and', 'health', 'care', 'consumption', 'in', 'adults', 'with']
    #         left_quota: 12
    #         context_left[-left_quota:] ['medication', ',', 'rehabilitation', 'and', 'health', 'care', 'consumption', 'in', 'adults', 'with']
    #         context_left[len(context_left)-left_quota:] ['adults', 'with']
    #         context_left: ['patient', '-', 'physician', 'disco', '##rdan', '##ce', 'in', 'global', 'assessment', 'in']
    #         left_quota: 11
    #         context_left[-left_quota:] ['patient', '-', 'physician', 'disco', '##rdan', '##ce', 'in', 'global', 'assessment', 'in']
    #         context_left[len(context_left)-left_quota:] ['in']
    #         context_left: ['spinal', 'load', 'in', 'nurses', 'during', 'emergency', 'lifting', 'of']
    #     '''
    if left_quota != 0:
        assert context_left[-left_quota:] == context_left[max(0,len(context_left)-left_quota):]
    context_tokens = (
        #context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
        context_left[max(0,len(context_left)-left_quota):] + mention_tokens + context_right[:right_quota]
    )
    context_tokens_ori = context_tokens
    #else:
    #    context_tokens = mention_tokens
    
    if not use_context:
        # if not using context, setting both the context and mention tokens as the same, original mention tokens
        context_tokens = mention_tokens_ori
        mention_tokens = mention_tokens_ori

    # get both mention and context ids (only contain mention if use_context is True) - and pad them into the specified lengths
    # also include the contextual ids (with context even use_context is False) - and pad them into the specified lengths
    mention_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
    # pad or truncate mention_ids
    if len(mention_ids) < max_seq_length:
        padding = [0] * (max_seq_length - len(mention_ids))
        mention_ids += padding
    else:
        print('mention_id beyond length of %d' % max_seq_length) 
        mention_ids = mention_ids[:max_seq_length]
        print('mention_id overlength issue fixed')
    #print('mention_ids:',mention_ids)
    
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)        
    # pad or truncate input_ids    
    if len(input_ids) < max_seq_length:
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
    else:
        # truncate but keep [CLS] and [SEP]
        input_ids = input_ids[0:max_seq_length-1] + input_ids[-1:]
    #assert len(input_ids) == max_seq_length
    if len(input_ids) != max_seq_length:
        print('len(input_ids):',len(input_ids),'max_seq_length:',max_seq_length)
        print('context_tokens:',context_tokens)
        print('mention_tokens:',mention_tokens)
    
    context_tokens_ori = ["[CLS]"] + context_tokens_ori + ["[SEP]"]
    input_ids_ori = tokenizer.convert_tokens_to_ids(context_tokens_ori)        
    # pad or truncate input_ids    
    if len(input_ids_ori) < max_seq_length:
        padding = [0] * (max_seq_length - len(input_ids_ori))
        input_ids_ori += padding
    else:
        # truncate but keep [CLS] and [SEP]
        input_ids_ori = input_ids_ori[0:max_seq_length-1] + input_ids_ori[-1:]

    return {
        "mention_tokens": mention_tokens, # add mention tokens (with ent_start_token and ent_end_token)
        "mention_ids": mention_ids, # add mention token ids (with ent_start_token and ent_end_token)
        "tokens": context_tokens, # contain contexts if use_context is True, otherwise only mentions
        "ids": input_ids,
        "contextual_tokens": context_tokens_ori, # contextual token ids (even use_context is False)
        "contextual_ids": input_ids_ori, # contextual token ids (even use_context is False)        
    }

# preprocess the entity candidates into tokens and input_ids using the tokenizer of the BERT models used in bi-encoder
# with synonyms as an option - the second argument
#   when synonyms is None, not using them, otherwise, using them with the special token of ENT_SYN_TAG
def get_candidate_representation(
    candidate_desc, 
    synonyms,
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
    syn_tag=ENT_SYN_TAG,
    use_NIL_tag=False,
    NIL_tag=ENT_NIL_TAG,
    use_NIL_desc=False,
    use_NIL_desc_tag=False,
    use_synonyms=True,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    #print('candidate_title:',candidate_title)
    #print('use_NIL_tag in data_process.get_candidate_representation:',use_NIL_tag)
    #print('use_NIL_desc in data_process.get_candidate_representation:',use_NIL_desc)
    if candidate_title is not None:
        if candidate_title == 'NIL':
            if use_NIL_tag:
                title_tokens = [NIL_tag]
            else:
                title_tokens = tokenizer.tokenize(candidate_title)      
            if use_NIL_desc:
                if use_NIL_desc_tag: #TODO and run
                    cand_tokens = tokenizer.tokenize("It is a") + [NIL_tag] + tokenizer.tokenize("option.") 
                else:
                    cand_tokens = tokenizer.tokenize("It is a NIL option.")
        else:
            title_tokens = tokenizer.tokenize(candidate_title)
        
        if use_synonyms and not synonyms is None:
            #print('synonyms found')
            all_syns_tokens = []
            list_synonyms = synonyms.split('|')
            for synonym in list_synonyms:
                syn_tokens = tokenizer.tokenize(synonym)
                all_syns_tokens = all_syns_tokens + syn_tokens + [syn_tag]
            cand_tokens = title_tokens + [title_tag] + all_syns_tokens + cand_tokens # title + synonyms + description/definition
        else:
            cand_tokens = title_tokens + [title_tag] + cand_tokens # title + description/definition
        #print('cand_tokens:',cand_tokens)
        '''
        e.g. for {"text": "A disorder characterized by a defect in mitral valve function or structure.", "idx": "C0026265", "title": "Diseases of mitral valve", "entity": "Diseases of mitral valve"} 
        Label tokens : [CLS] mit ##ral valve ins ##uf ##fi ##ciency [unused2] blood flows backward because the mit ##ral valve closes improper ##ly [SEP]
        for {"text": "", "idx": "CUI-less", "title": "NIL", "entity": "NIL"}
        Label tokens : [CLS] ni ##l [unused2] [SEP] 
        - so NIL entity is treated here too.

        another example with synonyms
        cand_tokens: ['ta', '##chy', '##p', '##nea', '[unused2]', 'ta', '##chy', '##p', '##no', '##ea', '[unused3]', 'rapid', 'breathing', '[unused3]', 'ta', '##chy', '##p', '##ne', '##ic', '[unused3]', 'ta', '##chy', '##p', '##no', '##ei', '##c', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##nea', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##no', '##ea', '[unused3]', 'rapid', 'res', '##piration', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##nea', '(', 'context', '-', 'dependent', 'category', ')', '[unused3]', 'ta', '##chy', '##p', '##nea', '(', 'finding', ')', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##nea', '(', 'situation', ')', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##no', '##ea', '(', 'situation', ')', '[unused3]', 'z', '##vy', '##sen', '##a', 'ry', '##ch', '##los', '##t', 'd', '##ych', '##ani', '(', 'dec', '##ho', '##va', 'fr', '##ek', '##ven', '##ce', ')', '.']
        '''
    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }

# preprocess the edge candidates into tokens and input_ids using the tokenizer of the BERT models used in bi-encoder
def get_edge_candidate_representation(
    #candidate_desc, 
    #synonyms,
    tokenizer, 
    max_seq_length, 
    parent_candidate_title=None,
    child_candidate_title=None,
    title_tag=ENT_TITLE_TAG,
    parent_title_tag=ENT_PARENT_TAG,
    child_title_tag=ENT_CHILD_TAG,
    top_tag=ENT_TOP_TAG,
    bottom_tag=ENT_NULL_TAG,
    #syn_tag=ENT_SYN_TAG,
    #use_NIL_tag=False,
    #NIL_tag=ENT_NIL_TAG,
    #use_NIL_desc=False,
    #use_NIL_desc_tag=False,
    #use_synonyms=True,
):
    '''
        adapted to edges for insertion
    '''
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    # cand_tokens = tokenizer.tokenize(candidate_desc)
    #print('candidate_title:',candidate_title)
    #print('use_NIL_tag in data_process.get_candidate_representation:',use_NIL_tag)
    #print('use_NIL_desc in data_process.get_candidate_representation:',use_NIL_desc)
    # if candidate_title is not None:
    #     if candidate_title == 'NIL':
    #         if use_NIL_tag:
    #             title_tokens = [NIL_tag]
    #         else:
    #             title_tokens = tokenizer.tokenize(candidate_title)      
    #         if use_NIL_desc:
    #             if use_NIL_desc_tag: #TODO and run
    #                 cand_tokens = tokenizer.tokenize("It is a") + [NIL_tag] + tokenizer.tokenize("option.") 
    #             else:
    #                 cand_tokens = tokenizer.tokenize("It is a NIL option.")
    #     else:
    #         title_tokens = tokenizer.tokenize(candidate_title)
        
    #     if use_synonyms and not synonyms is None:
    #         #print('synonyms found')
    #         all_syns_tokens = []
    #         list_synonyms = synonyms.split('|')
    #         for synonym in list_synonyms:
    #             syn_tokens = tokenizer.tokenize(synonym)
    #             all_syns_tokens = all_syns_tokens + syn_tokens + [syn_tag]
    #         cand_tokens = title_tokens + [title_tag] + all_syns_tokens + cand_tokens # title + synonyms + description/definition
    #     else:
    #         cand_tokens = title_tokens + [title_tag] + cand_tokens # title + description/definition
    #print('cand_tokens:',cand_tokens)
    '''
    e.g. for {"text": "A disorder characterized by a defect in mitral valve function or structure.", "idx": "C0026265", "title": "Diseases of mitral valve", "entity": "Diseases of mitral valve"} 
    Label tokens : [CLS] mit ##ral valve ins ##uf ##fi ##ciency [unused2] blood flows backward because the mit ##ral valve closes improper ##ly [SEP]
    for {"text": "", "idx": "CUI-less", "title": "NIL", "entity": "NIL"}
    Label tokens : [CLS] ni ##l [unused2] [SEP] 
    - so NIL entity is treated here too.

    another example with synonyms
    cand_tokens: ['ta', '##chy', '##p', '##nea', '[unused2]', 'ta', '##chy', '##p', '##no', '##ea', '[unused3]', 'rapid', 'breathing', '[unused3]', 'ta', '##chy', '##p', '##ne', '##ic', '[unused3]', 'ta', '##chy', '##p', '##no', '##ei', '##c', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##nea', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##no', '##ea', '[unused3]', 'rapid', 'res', '##piration', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##nea', '(', 'context', '-', 'dependent', 'category', ')', '[unused3]', 'ta', '##chy', '##p', '##nea', '(', 'finding', ')', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##nea', '(', 'situation', ')', '[unused3]', '[', 'd', ']', 'ta', '##chy', '##p', '##no', '##ea', '(', 'situation', ')', '[unused3]', 'z', '##vy', '##sen', '##a', 'ry', '##ch', '##los', '##t', 'd', '##ych', '##ani', '(', 'dec', '##ho', '##va', 'fr', '##ek', '##ven', '##ce', ')', '.']
    '''
    
    if parent_candidate_title == 'SCTID_THING':
        parent_title_tokens = [top_tag]
    else:   
        parent_title_tokens = tokenizer.tokenize(parent_candidate_title)
    
    if child_candidate_title == 'SCTID_NULL':
        children_title_tokens = [bottom_tag]
    else:            
        children_title_tokens = tokenizer.tokenize(child_candidate_title)

    cand_tokens = parent_title_tokens + [parent_title_tag] + children_title_tokens + [child_title_tag] + [title_tag]

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }

# remove the synonym rows from data (for the synonym as entities setting)
# ignore the cases when synonyms are used as the entitiy; this appears when using the _syn_full data setting and when the mention and contexts repeats (the first appearance is the canonical name of the entity). 
# this may occasionally remove simply duplicated data (e.g. 862817 kept out of 863798 in the NILK training data) - this is OK.
def remove_syn_rows_from_data(samples,mention_key="mention",context_key="context",):
    dict_mention_in_context = {}
    sample_w_o_syn_rows = []
    for sample in samples:
        mention = sample[mention_key]
        context_left = sample[context_key + "_left"]
        context_right = sample[context_key + "_right"]
        triple_men_in_contexts = (mention,context_left,context_right)
        if not triple_men_in_contexts in dict_mention_in_context:
            dict_mention_in_context[triple_men_in_contexts] = 1
            sample_w_o_syn_rows.append(sample)                
    return sample_w_o_syn_rows
    
# generate TensorData for batch iteration from the raw data
def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    synonym_key="synonyms",
    context_key="context",
    use_context=True,
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    debug_max_lines=200,
    logger=None,
    NIL_ent_id=88150, # added NIL_ent_id here
    use_NIL_tag=False,
    use_NIL_desc=False,
    use_NIL_desc_tag=False,
    use_synonyms=True,
    remove_syn_rows=False,
):
    processed_samples = []
    # remove the synonym rows from data for inference, but not for training (if there are any, for the synonym as entities setting)
    print('num of samples:',len(samples))
    if remove_syn_rows:
        samples = remove_syn_rows_from_data(samples)
        print('num of samples after removing syn (or repeated) rows:',len(samples))
    if debug:
        samples = samples[:debug_max_lines]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
            use_context=use_context,
        )

        label = sample[label_key] # label/entity description
        title = sample.get(title_key, None) # title or canonical name
        synonyms = sample.get(synonym_key, None)
        #print('use_NIL_tag in process_mention_data:',use_NIL_tag)
        #print('use_NIL_desc in process_mention_data:',use_NIL_desc)
        #print('use_NIL_desc_tag in process_mention_data:',use_NIL_desc_tag)
        label_tokens = get_candidate_representation(
            label, synonyms, tokenizer, max_cand_length, title,
            use_NIL_tag=use_NIL_tag,
            use_NIL_desc=use_NIL_desc,
            use_NIL_desc_tag=use_NIL_desc_tag,
            use_synonyms=use_synonyms,
        )

        # # process UMLS CUI into integers if needed 'C0003507'
        # if type(sample["label_id"]) is int:
        #     label_idx = int(sample["label_id"])
        # else:
        #     label_idx = int(sample["label_id"][1:])
        label_idx = int(sample["label_id"])
        is_label_NIL = label_idx == NIL_ent_id

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "is_label_NIL": [is_label_NIL], # is_label_NIL added
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    #if debug and logger:
    if logger:
        logger.info("====Processed samples (first 10): ====")
        for sample in processed_samples[:10]:
            logger.info("Mention tokens : " + " ".join(sample["context"]["mention_tokens"]))
            logger.info(
                "Mention ids : " + " ".join([str(v) for v in sample["context"]["mention_ids"]])
            )
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            if "src" in sample:
                logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d is %s" % (sample["label_idx"][0], 'NIL' if sample["is_label_NIL"][0] else 'in-KB'))

    mention_vecs = torch.tensor(
        select_field(processed_samples, "context", "mention_ids")
    )
    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    print('cand_vecs:',cand_vecs)
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )

    is_label_NIL = torch.tensor(
        select_field(processed_samples, "is_label_NIL"), dtype=torch.bool,
    )

    data = {
        "mention_vecs": mention_vecs,
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
        "is_NIL": is_label_NIL,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(mention_vecs, context_vecs, cand_vecs, src_vecs, label_idx, is_label_NIL)
    else:
        tensor_data = TensorDataset(mention_vecs, context_vecs, cand_vecs, label_idx, is_label_NIL)
    return data, tensor_data

# generate TensorData for batch iteration from the raw data
def process_mention_for_insertion_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    #synonym_key="synonyms",
    context_key="context",
    use_context=True,
    #label_key="label",
    label_concept_ori_key="label_concept_ori",
    parent_title_key='parent',
    child_title_key='child',
    #title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    parent_title_token=ENT_PARENT_TAG,
    child_title_token=ENT_CHILD_TAG,
    top_token=ENT_TOP_TAG,
    bottom_token=ENT_NULL_TAG,
    debug=False,
    debug_max_lines=200,
    logger=None,
    NIL_ent_id=88150, # added NIL_ent_id here
    # use_NIL_tag=False,
    # use_NIL_desc=False,
    # use_NIL_desc_tag=False,
    #use_synonyms=False,
    #remove_syn_rows=False,
):
    '''
        adapted to mention for insertion data
    '''
    processed_samples = []
    # remove the synonym rows from data for inference, but not for training (if there are any, for the synonym as entities setting)
    print('num of samples:',len(samples))
    # if remove_syn_rows:
    #     samples = remove_syn_rows_from_data(samples)
    #     print('num of samples after removing syn (or repeated) rows:',len(samples))
    if debug:
        samples = samples[:debug_max_lines]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
            use_context=use_context,
        )

        #label = sample[label_key] # label/entity description
        concept_id_ori = sample[label_concept_ori_key] # the mention's original concept ID in the KB (before KB versioning) - then to be encoded (here using a tokeniser) - this is used for disambiguate different data rows (same mention w/o context, but matched to two ori concept ids, then later can be inserted into two different set of edges)
        concept_id_ori_token = tokenizer.tokenize(concept_id_ori)
        concept_id_ori_token_ids = tokenizer.convert_tokens_to_ids(concept_id_ori_token)
        # pad the label_concept_ori encoding to the max_context_length as well
        if len(concept_id_ori_token_ids) < max_context_length:
            concept_id_ori_token_ids = concept_id_ori_token_ids + [0]*(max_context_length-len(concept_id_ori_token_ids))
        else:
            concept_id_ori_token_ids = concept_id_ori_token_ids[:max_context_length]
        #title = sample.get(title_key, None) # title or canonical name
        concept_id_ori_encoded = {"tokens":concept_id_ori_token, "ids": concept_id_ori_token_ids}

        parent_title = sample.get(parent_title_key, None)
        child_title = sample.get(child_title_key, None)
        #synonyms = sample.get(synonym_key, None)
        #print('use_NIL_tag in process_mention_data:',use_NIL_tag)
        #print('use_NIL_desc in process_mention_data:',use_NIL_desc)
        #print('use_NIL_desc_tag in process_mention_data:',use_NIL_desc_tag)
        label_tokens = get_edge_candidate_representation(
            tokenizer, max_cand_length, parent_title, child_title,
            title_tag=title_token,
            parent_title_tag=parent_title_token,
            child_title_tag=child_title_token,
            top_tag=top_token,
            bottom_tag=bottom_token,
            # use_NIL_tag=use_NIL_tag,
            # use_NIL_desc=use_NIL_desc,
            # use_NIL_desc_tag=use_NIL_desc_tag,
            # use_synonyms=use_synonyms,
        )

        # # process UMLS CUI into integers if needed 'C0003507'
        # if type(sample["label_id"]) is int:
        #     label_idx = int(sample["label_id"])
        # else:
        #     label_idx = int(sample["label_id"][1:])
        label_idx = int(sample["edge_label_id"])
        ent_label_idx = int(sample["entity_label_id"])
        is_label_NIL = ent_label_idx == NIL_ent_id

        record = {
            "context": context_tokens,
            "label": label_tokens, #label_text_vecs
            "label_idx": [label_idx],
            "is_label_NIL": [is_label_NIL], # is_label_NIL added
            "concept_id_ori": concept_id_ori_encoded,
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    #if debug and logger:
    if logger:
        logger.info("====Processed samples (first 10): ====")
        for sample in processed_samples[:10]:
            logger.info("Mention tokens : " + " ".join(sample["context"]["mention_tokens"]))
            logger.info(
                "Mention ids : " + " ".join([str(v) for v in sample["context"]["mention_ids"]])
            )
            logger.info("Concept ID ori tokens : " + " ".join(sample["concept_id_ori"]["tokens"]))
            logger.info(
                "Concept ID ori ids : " + " ".join([str(v) for v in sample["concept_id_ori"]["ids"]])
            )
            logger.info("(Context) tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Contextual tokens : " + " ".join(sample["context"]["contextual_tokens"]))
            logger.info(
                "Contextual ids : " + " ".join([str(v) for v in sample["context"]["contextual_ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            if "src" in sample:
                logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d is %s" % (sample["label_idx"][0], 'NIL' if sample["is_label_NIL"][0] else 'in-KB'))

    mention_vecs = torch.tensor(
        select_field(processed_samples, "context", "mention_ids")
    )
    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    contextual_vecs = torch.tensor(
        select_field(processed_samples, "context", "contextual_ids"), dtype=torch.long,
    )
    label_text_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    print('label_text_vecs:',label_text_vecs)
    concept_id_ori_vecs = torch.tensor(
        select_field(processed_samples, "concept_id_ori", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )

    is_label_NIL = torch.tensor(
        select_field(processed_samples, "is_label_NIL"), dtype=torch.bool,
    )

    data = {
        "mention_vecs": mention_vecs,
        "context_vecs": context_vecs, # the one with contexts, but only when use_context is True, otherwise mentions only
        "contextual_vecs": contextual_vecs, # the one with contexts (for later aggregating mentions from different contexts when use_context is False)
        "label_text_vecs": label_text_vecs,
        "label_idx": label_idx,
        "is_NIL": is_label_NIL,
        "concept_id_ori_vecs": concept_id_ori_vecs
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(mention_vecs, context_vecs, contextual_vecs, label_text_vecs, src_vecs, label_idx, is_label_NIL, concept_id_ori_vecs)
    else:
        tensor_data = TensorDataset(mention_vecs, context_vecs, contextual_vecs, label_text_vecs, label_idx, is_label_NIL, concept_id_ori_vecs)
    return data, tensor_data
