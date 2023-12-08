# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# examples running script for this program with a new dataset:
# PYTHONPATH=. python blink/biencoder/eval_biencoder.py   --data_path preprocessing/share_clef_2013_sampled_100    --output_path models/biencoder    --max_context_length 128    --max_cand_length 128    --eval_batch_size 32    --bert_model bert-base-uncased    --data_parallel --mode train --cand_pool_path preprocessing/saved_cand_ids_umls2012AB_re_tr.pt --cand_encode_path models/UMLS2012AB_ent_enc_re_tr/UMLS2012AB_ent_enc_re_tr.t7 --save_topk_result --top_k 100

#PYTHONPATH=. python blink/biencoder/eval_biencoder.py   --data_path preprocessing/share_clef_2013_sampled_100    --output_path models/biencoder    --max_context_length 128    --max_cand_length 128    --eval_batch_size 32    --bert_model bert-large-uncased --path_to_model models/biencoder_wiki_large.bin  --data_parallel --mode train --cand_pool_path preprocessing/saved_cand_ids_umls2012AB.pt --cand_encode_path models/UMLS2012AB_ent_enc/UMLS2012AB_ent_enc.t7 --save_topk_result

#added function to get and save BM25 results.

import argparse
import json
import logging
import os
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

#from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.biencoder import BiEncoderRanker
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats
from blink.common.params import BlinkParser, ENT_TITLE_TAG


def load_entity_dict(logger, params, is_zeshel):
    if is_zeshel:
        return load_entity_dict_zeshel(logger, params)

    path = params.get("entity_dict_path", None)
    assert path is not None, "Error! entity_dict_path is empty."

    entity_list = []
    logger.info("Loading entity description from path: " + path)
    with open(path, 'rt') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_list.append((title, text))
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_list

# zeshel version of get candidate_pool_tensor
def get_candidate_pool_tensor_zeshel(
    entity_dict,
    tokenizer,
    max_seq_length,
    logger,
):
    candidate_pool = {}
    for src in range(len(WORLDS)):
        if entity_dict.get(src, None) is None:
            continue
        logger.info("Get candidate desc to id for pool %s" % WORLDS[src])
        candidate_pool[src] = get_candidate_pool_tensor(
            entity_dict[src],
            tokenizer,
            max_seq_length,
            logger,
        )

    return candidate_pool


def get_candidate_pool_tensor_helper(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
    is_zeshel,
):
    if is_zeshel:
        return get_candidate_pool_tensor_zeshel(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )
    else:
        return get_candidate_pool_tensor(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )


def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = [] 
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = data.get_candidate_representation(
                entity_text, 
                None, # synonyms
                tokenizer, 
                max_seq_length,
                title,
                use_NIL_tag=params["use_NIL_tag"],
                use_NIL_desc=params["use_NIL_desc"],
                use_NIL_desc_tag=params["use_NIL_desc_tag"],
                use_synonyms=params["use_synonyms"],
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool) 
    return cand_pool


def encode_candidate(
    reranker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
    is_zeshel,
):
    if is_zeshel:
        src = 0
        cand_encode_dict = {}
        for src, cand_pool in candidate_pool.items():
            logger.info("Encoding candidate pool %s" % WORLDS[src])
            cand_pool_encode = encode_candidate(
                reranker,
                cand_pool,
                encode_batch_size,
                silent,
                logger,
                is_zeshel=False,
            )
            cand_encode_dict[src] = cand_pool_encode
        return cand_encode_dict
        
    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_or_generate_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    is_zeshel = params.get("zeshel", None)
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        print('candidate_pool is not provided: None')
        # compute candidate pool from entity list
        entity_desc_list = load_entity_dict(logger, params, is_zeshel)
        candidate_pool = get_candidate_pool_tensor_helper(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger,
            is_zeshel,
        )

        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)

    return candidate_pool

# here it load all the candidate entities, which need both the catalogue of the entity and the encoding of the entity
def _load_candidates(
    entity_catalogue, logger=None
):
    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2synonyms = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_id2wikipedia_id = {}
    local_idx = 0
    with open(entity_catalogue, "r", encoding="utf-8-sig") as fin: # encoding adapted to utf-8-sig if needed
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                #assert wikipedia_id not in wikipedia_id2local_id # this does not hold any more with the "entity as synonym" setting
                #thus, only record the first idx if the entity is in the 
                if wikipedia_id not in wikipedia_id2local_id:
                    wikipedia_id2local_id[wikipedia_id] = local_idx
                    # processing the synonyms
                    if not local_idx in id2synonyms:
                        id2synonyms[local_idx] = entity["title"]
                    else:
                        id2synonyms[local_idx] = id2synonyms[local_idx] + '|' + entity["title"]
                local_id2wikipedia_id[local_idx] = wikipedia_id
            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            if "synonyms" in entity:
                id2synonyms[local_idx] = entity["synonyms"]                
            id2text[local_idx] = entity["text"]
            local_idx += 1
    print('local_id2wikipedia_id:',len(local_id2wikipedia_id))
    print('wikipedia_id2local_id:',len(wikipedia_id2local_id))
    print('id2synonyms:',len(id2synonyms))
    return (
        title2id,
        id2title,
        id2synonyms,
        id2text,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
    )

# here it load all the candidate edges, which need both the catalogue of the entity and the encoding of the entity
def _load_edge_candidates(
    entity_catalogue, logger=None
):
    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2synonyms = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_id2wikipedia_id = {}
    local_idx = 0
    with open(entity_catalogue, "r", encoding="utf-8-sig") as fin: # encoding adapted to utf-8-sig if needed
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            entity["idx"] = entity["parent_idx"] + '-' + entity["child_idx"]
            entity["title"] = entity["parent"] + '-' + entity["child"]
            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                #assert wikipedia_id not in wikipedia_id2local_id # this does not hold any more with the "entity as synonym" setting
                #thus, only record the first idx if the entity is in the 
                if wikipedia_id not in wikipedia_id2local_id:
                    wikipedia_id2local_id[wikipedia_id] = local_idx
                    # processing the synonyms
                    if not local_idx in id2synonyms:
                        id2synonyms[local_idx] = entity["title"]
                    else:
                        id2synonyms[local_idx] = id2synonyms[local_idx] + '|' + entity["title"]
                local_id2wikipedia_id[local_idx] = wikipedia_id
            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            if "synonyms" in entity:
                id2synonyms[local_idx] = entity["synonyms"]                
            #id2text[local_idx] = entity["text"]
            local_idx += 1
    print('local_id2wikipedia_id:',len(local_id2wikipedia_id))
    print('wikipedia_id2local_id:',len(wikipedia_id2local_id))
    print('id2synonyms:',len(id2synonyms))
    return (
        title2id,
        id2title,
        id2synonyms,
        #id2text,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
    )

def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model 
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    #print(tokenizer)
    model = reranker.model
    
    device = reranker.device
    
    cand_encode_path = params.get("cand_encode_path", None)
    
    # candidate encoding is not pre-computed. 
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    candidate_pool = load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
    )       

    # load/generate candidate pool used for BM25 if provided and use BM25 for entity cand retrieval.
    cand_pool_path_for_BM25 = params.get("cand_pool_path_for_BM25", None)
    if (not cand_pool_path_for_BM25 is None) and params["use_BM25"]:
        candidate_pool_for_BM25 = load_or_generate_candidate_pool(
            tokenizer,
            params,
            logger,
            cand_pool_path_for_BM25,
        )
    else:
        candidate_pool_for_BM25 = None

    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(
            reranker,
            candidate_pool,
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
            is_zeshel=params.get("zeshel", None)
            
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)

    (
        title2id,
        id2title,
        id2synonyms,
        #id2text,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
    ) = _load_edge_candidates(
        params["entity_dict_path"], 
        #candidate_encoding, 
        logger=logger,
    )

    test_samples = utils.read_dataset(params["mode"], params["data_path"], debug=params["debug"],debug_max_lines=params["debug_max_lines"],random_sampling=params["debug_random_sample"])
    logger.info("Read %d test samples." % len(test_samples))
    
    test_data, test_tensor_data = data.process_mention_for_insertion_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        #context_key=params['context_key'],
        use_context=params['use_context'],
        silent=params["silent"],
        debug=params["debug"],
        debug_max_lines=params["debug_max_lines"],
        logger=logger,
        NIL_ent_id=params["NIL_ent_ind"],        
        # use_NIL_tag=params["use_NIL_tag"],
        # use_NIL_desc=params["use_NIL_desc"],
        # use_NIL_desc_tag=params["use_NIL_desc_tag"],
        # use_synonyms=params["use_synonyms"],
        # remove_syn_rows=True,
    )
    #print('test_data:',test_data)
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["eval_batch_size"]
    )
    
    fix_emb_preds_in_KB_fn = ""
    sample_fix_emb_preds_in_KB = None
    if params["use_fix_embs"]:
        #set fix_emb_preds_in_KB_fn
        fix_emb_preds_in_KB_fn = "blink/prompting/edge_preds_micro%dto%d%s%s_%s.jsonl" % (
            params["top_k_cand_seed"], 
            params["top_k"],
            '_enrich' if params["edge_cand_enrich"] else '',
            '' if params["use_context"] else '_w_o_ctx',
            params["mode"],
            )
        print("fix_emb_preds_in_KB_fn:",fix_emb_preds_in_KB_fn)

        sample_fix_emb_preds_in_KB = utils.read_dataset_from_fn(fix_emb_preds_in_KB_fn, debug=params["debug"],debug_max_lines=params["debug_max_lines"],random_sampling=params["debug_random_sample"])
        logger.info("Read %d test samples." % len(test_samples))

    #print('index of ENT_TITLE_TAG:',tokenizer.convert_tokens_to_ids(ENT_TITLE_TAG))
    index_title_special_token = tokenizer.convert_tokens_to_ids(ENT_TITLE_TAG)
    save_results = params.get("save_topk_result") # should be set as true to generate data for cross-encoder training
    #save_true_predictions_only = not params.get("save_all_predictions")
    if "-NIL" in params["mode"]:
        save_true_predictions_only = not params.get("save_all_predictions") # for out-of-KB data, save everything if chosen to (for evaluation and prompt generation using candidate_analysis.py)        
    else:
        save_true_predictions_only = True # for in-KB data, there is no need to save everything (at the moment, even for test-in-KB)
    new_data = nnquery.get_topk_predictions(
        reranker,
        test_dataloader,
        candidate_pool,
        candidate_encoding,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
        params["silent"],
        logger,
        params["top_k"],
        params.get("zeshel", None),
        save_predictions=save_results,
        save_true_predictions_only=save_true_predictions_only,
        add_NIL=params["add_NIL_to_bi_enc_pred"], # add NIL to the last element of the biencoder predicted entity indicies, if NIL was not predicted
        NIL_ent_id=params["NIL_ent_ind"],
        use_fix_embs=params["use_fix_embs"],
        sample_fix_emb_preds_in_KB=sample_fix_emb_preds_in_KB,
        use_BM25=params["use_BM25"],
        candidate_pool_for_BM25=candidate_pool_for_BM25,
        get_is_men_str_mat_fts=params["use_extra_features"],
        index_title_special_token=index_title_special_token,
        edge_cand_enrich=params["edge_cand_enrich"],
        edge_catalogue_fn=params["edge_catalogue_fn"],
        top_k_cand_seed=params["top_k_cand_seed"],
        use_leaf_edge_score=params["use_leaf_edge_score"],
        LEAF_EDGE_SCORE=params["LEAF_EDGE_SCORE"],
        edge_ranking_by_score=params["edge_ranking_by_score"],
        #aggregating_factor=params["aggregating_factor"],
    )
    #print('new_data:',new_data)

    if save_results: 
        # get whether the cand enc files has NIL encoded.
        #assert ('w_NIL' in cand_pool_path) == ('w_NIL' in cand_encode_path)
        #is_w_NIL = ('w_NIL' in cand_pool_path) and ('w_NIL' in cand_encode_path)

        save_data_dir = os.path.join(
            params['output_path'],
            "top%d_candidates%s%s" % (params['top_k'],
                                      '_BM25' if params["use_BM25"] else '',
                                      '_fix_embs' if params["use_fix_embs"] else '',)
                                      #'_w_o_NIL' if not is_w_NIL else ''),
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params['mode'])
        torch.save(new_data, save_data_path)
        print('data saved to %s' % save_data_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    parser.add_argument('--entity_dict_path', type=str, required=True, help='filepath to entities (or edges for insertion) to encode (.jsonl file)')
    parser.add_argument('--cand_pool_path_for_BM25', type=str, help='Path for cached candidate pool (id tokenization of candidates) for BM25 - this is syn as ent, so there are |syns| entities. This is used in the syn mode for BM25 where the --cand_pool_path is in syn as attr format and used for output only.')
    parser.add_argument('--save_all_predictions',action="store_true", help='turn on if saving all predictions instead of saving only those having gold in top-k candidates.')
    parser.add_argument('--edge_cand_enrich',action="store_true", help='whether to enrich the edge candidates.')
    parser.add_argument('--edge_catalogue_fn', type=str, 
                        help='filepath to entities to encode (.jsonl file)',
                        default="ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl")
    parser.add_argument('--top_k_cand_seed', type=int, 
                        help='number of seeded candidates (before any candidate enrichment steps) for each mention',
                        default="10")
    parser.add_argument("--use_leaf_edge_score",action="store_true",help="Whether to use the LEAF_EDGE_SCORE below")
    parser.add_argument("--LEAF_EDGE_SCORE",type=int,help="A pre-defined score for leaf edges, i.e., <parent, NULL>, for the edge ranking: For SNOMED-CT, this is suggested to be a very high score, e.g. 1000; For DO, this is suggested to be a very low score, e.g. -1000", default=1000)
    parser.add_argument("--edge_ranking_by_score",action="store_true",help="Whether to rank edges by score")
    parser.add_argument("--use_fix_embs",action="store_true",help="Whether to use results from fix embeddings")    
    parser.add_argument("--debug_random_sample",action="store_true",help="Whether to use random sampling of data if debugging")
    #parser.add_argument('--fix_emb_preds_in_KB_fn', type=str, 
                        # help='filepath to the fixed embedding in-KB predictions (.jsonl file)',
                        # default="blink/prompting/edge_preds25to50_enrich_w_o_ctx_train-all-1000.txt")
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    mode_list = params["mode"].split(',') # param["mode"] as 'train,valid'
    for mode in mode_list:
        new_params = params
        new_params["mode"] = mode
        main(new_params)
