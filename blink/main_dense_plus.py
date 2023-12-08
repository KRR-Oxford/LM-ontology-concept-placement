# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json

from tqdm import tqdm
from tqdm.contrib import tzip
import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import load_biencoder
from blink.crossencoder.crossencoder_multi_label import load_crossencoder
from blink.biencoder.data_process import (
    process_mention_for_insertion_data,
    #get_candidate_representation,
)
from blink.common.params import ENT_TITLE_TAG
from blink.biencoder.nn_prediction import add_dict_list, load_edge_catalogue, enrich_edge_cands_w_scores,get_ranking_indices_w_BM25,_aggregating_indices_synonyms,_normalise_local_id,_normalise_to_ori_local_id#,_aggregating_indices_synonyms_ave
import blink.candidate_ranking.utils as utils
from preprocessing.onto_snomed_owl_util import get_entity_graph_info, load_SNOMEDCT_deeponto, deeponto2dict_ids, load_deeponto_verbaliser, get_iri_from_SCTID_id, get_SCTID_id_from_iri, _extract_iris_in_parsed_complex_concept,extract_SNOMEDCT_deeponto_taxonomy,calculate_wu_palmer_sim,is_complex_concept,get_dict_iri_pair_to_lca,filter_out_complex_edges,deeponto2dict_ids_obj_prop
from blink.candidate_ranking.semantic_edge_evaluation import overall_edge_wp_sim,overall_edge_wp_sim_w_comp
from blink.crossencoder.train_cross_multi_label import modify, aggregate_into_multi_label, evaluate_edges#, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_for_insertion_data#,prepare_crossencoder_data
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

from blink.out_of_KB_utils import set_NIL_to_candidates, infer_out_KB_ent_bi_enc, infer_out_KB_ent_BM25, _softmax
import matplotlib.pyplot as plt
import csv
import os
import sys
import pickle

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]

#torch.cuda.set_device(0)

def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences, lowercase=True):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label_concept_str"] = "unknown"
        record["label_concept"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ] #.lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ] #.lower()
        record["mention"] = mention["text"] #.lower()
        if lowercase:
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


# here it load all the candidate entities, which need both the catalogue of the entity and the encoding of the entity
def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding) # load the .t7 model of entity encoding        
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

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

            entity["title"] = (entity["parent"], entity["child"])
            entity["text"] = ""
            id_path = (entity["parent_idx"],entity["child_idx"])
            local_id2wikipedia_id[local_idx] = id_path
            wikipedia_id2local_id[id_path] = local_idx
            
            # if "idx" in entity:
            #     split = entity["idx"].split("curid=")
            #     if len(split) > 1:
            #         wikipedia_id = int(split[-1].strip())
            #     else:
            #         wikipedia_id = entity["idx"].strip()

            #     #assert wikipedia_id not in wikipedia_id2local_id # this does not hold any more with the "entity as synonym" setting
            #     #thus, only record the first idx if the entity is in the 
            #     if not wikipedia_id in wikipedia_id2local_id:
            #         wikipedia_id2local_id[wikipedia_id] = local_idx
            #     else:
            #         # it is a synonym row (in the "syn as ent" setting)
            #         # processing the synonyms
            #         assert wikipedia_id in wikipedia_id2local_id
            #         label_idx_normalised = wikipedia_id2local_id[wikipedia_id]
            #         if not label_idx_normalised in id2synonyms:
            #             id2synonyms[label_idx_normalised] = entity["title"]
            #         else:
            #             id2synonyms[label_idx_normalised] = id2synonyms[label_idx_normalised] + '|' + entity["title"]
            #     local_id2wikipedia_id[local_idx] = wikipedia_id

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            if "synonyms" in entity:
                id2synonyms[local_idx] = entity["synonyms"]
            id2text[local_idx] = entity["text"]
            local_idx += 1

        # transform the local_id into original_local_id in the dicts (there is a difference between local_id and ori_local_id when using the "synonym as entity" setting: ori_local_id is the non-synonym id and local_id is the id in the "synonym as entity" setting)
        wikipedia_id2ori_local_id = {k:ori_id for ori_id, (k,v) in enumerate(wikipedia_id2local_id.items())}
        ori_local_id2wikipedia_id = {v:k for ori_id, (k,v) in enumerate(wikipedia_id2ori_local_id.items())}
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2synonyms,
        id2text,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
        wikipedia_id2ori_local_id,
        ori_local_id2wikipedia_id,
        indexer,
    )


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


# here it processes the .json test file
def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger, lowercase=True):
    #print('wikipedia_id2local_id:',wikipedia_id2local_id)
    test_samples = []
    with open(test_filename, "r", encoding="utf-8-sig") as fin: # encoding adapted to utf-8-sig if needed
        lines = fin.readlines()
        for ind_row_men, line in enumerate(lines):
            record = json.loads(line)
            #record["label_concept_str"] = str(record["label_concept"])
            record["label_concept_str"] = (record["parent_concept"],record["child_concept"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label_concept_str"] in kb2id:
                    record["label_concept"] = kb2id[record["label_concept_str"]]
                else:
                    continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = record["label_concept_str"]
                    #if key.isnumeric():
                    #    key = int(record["label_concept_str"].strip())  
                    if key in wikipedia_id2local_id:
                        record["label_concept"] = wikipedia_id2local_id[key]
                        '''
                        for Share-CLEF_eHealth2013-train/test
                        unfound in entity id list: C0746226
                        unfound in entity id list: C0750125
                        unfound in entity id list: C0085649
                        unfound in entity id list: C0600260
                        '''
                    else:
                        print('unfound in entity id list:',key)
                        continue
                except:
                    continue

            #print('record label:', record['label_id'])
            if lowercase:
                # LOWERCASE EVERYTHING !
                record["context_left"] = record["context_left"].lower()
                record["context_right"] = record["context_right"].lower()
                record["mention"] = record["mention"].lower()
                if "synonyms" in record:
                    record["synonyms"] = record["synonyms"].lower()
            record["row_ind_men"] = ind_row_men
            test_samples.append(record)

    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger, lowercase=True
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger, lowercase)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params,use_NIL_tag=False,use_NIL_desc=False,use_NIL_desc_tag=False):
    _, tensor_data = process_mention_for_insertion_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        use_context=biencoder_params["use_context"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
        #use_NIL_tag=use_NIL_tag,
        #use_NIL_desc=use_NIL_desc,
        #use_NIL_desc_tag=use_NIL_desc_tag,
        #for_inference=True,
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader

# for inference with BM25
def _run_BM25(dataloader, candidate_pool, local_id2wikipedia_id,wikipedia_id2local_id,top_k=100,index_title_special_token=3,aggregating_factor=20):
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        mention_input, _, _, label_ids, _ = batch
        indicies = []
        scores = []
        for i in tqdm(range(mention_input.size(0))):
            # get indicies through BM25
            inds_per_mention, scores_per_mention = get_ranking_indices_w_BM25(mention_input[i].cpu().tolist(),candidate_pool.cpu().tolist(),topn=top_k*aggregating_factor,index_title_special_token=index_title_special_token)
            #inds = torch.tensor(inds)
            scores.append(scores_per_mention)
            indicies.append(inds_per_mention)
        scores = np.array(scores)
        indicies = np.array(indicies)    
        indicies = np.array([_aggregating_indices_synonyms(indicies_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k) for indicies_per_datum in indicies])

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    print('labels:',len(labels),len(labels[0]))    # labels: 99 1
    print('nns:',len(nns),len(nns[0])) # nns: 99 100
    print('all_scores:',len(all_scores),len(all_scores[0])) # all_scores: 99 100
    return labels, nns, all_scores

#output str content to a file
#input: filename and the content (str)
def _output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8") as f_output:
        f_output.write(str)

# for inference (with edge candidate enrich if chosen to)
def _run_biencoder(biencoder, dataloader, candidate_encoding, local_id2wikipedia_id,wikipedia_id2local_id, top_k=100, aggregating_factor=20, indexer=None):#, edge_cand_enrich=False,dict_ind_edge_json_info=None,dict_edge_tuple_to_ind=None,dict_parent_to_child=None,dict_child_to_parent=None, top_k_cand_seed=10, ):
    # '''
    #     The below arguments are for edge_cand_enrich: 
    #         edge_cand_enrich=False, 
    #         dict_ind_edge_json_info=None,
    #         dict_edge_tuple_to_ind=None,
    #         dict_parent_to_child=None,
    #         dict_child_to_parent=None,
    #         top_k_cand_seed=10,
    # '''
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores_sorted = []
    all_scores = []
    for batch in tqdm(dataloader):
        _, context_input, _, _, label_ids, _, _ = batch # here only need context input and label ids, but not candidate input
        context_input = context_input.to(device=biencoder.device)
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding, #.to(device)
                    unit_norm=False, # normalise the score (unit norm for mention and canditate) to cosine similarity
                )
                #print('scores in main_dense._run_biencoder():',scores.shape)
                #      scores in main_dense._run_biencoder(): torch.Size([3, 88150])
                #print('scores:',scores) sample scores below
                #this is a fairly large value - as the vectors were not normalised?
                '''scores: tensor([[73.8414, 72.7153, 74.7849,  ..., 74.1165, 75.0617, 71.7682],
                [70.2872, 69.9775, 70.9030,  ..., 72.0679, 71.9649, 67.5665],
                [78.1441, 75.7104, 77.5939,  ..., 78.3324, 76.8203, 71.4749],
                ...,
                [74.0225, 73.1922, 74.2409,  ..., 74.1694, 75.4666, 68.5208],
                [77.9721, 75.5569, 76.7411,  ..., 78.7790, 75.8169, 71.0363],
                [74.3793, 74.5216, 74.1057,  ..., 74.5519, 74.9505, 71.2634]])
                '''
                scores_sorted, indicies = scores.topk(top_k*aggregating_factor) # indices, this is just the row indices in the cand encoding matrix, corresponding to the local_idx for id2text, see def _load_candidates() in main_dense
                #print('scores:',scores)
                #print('scores.data:',scores.data)
                scores = scores.data.cpu().numpy()
                scores_sorted = scores_sorted.data.cpu().numpy()
                indicies = indicies.data.cpu().numpy()
                #error if .cpu() from cuda device - RuntimeError: CUDA error: an illegal memory access was encountered
                #scores = scores.data.cpu().numpy()
                #indicies = indicies.data.cpu().numpy()

                #aggregating indicies of synonyms
                #print("indicies:",indicies.shape)
                #by maximum pooling
                indicies = np.array([_aggregating_indices_synonyms(indicies_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k) for indicies_per_datum in indicies])
                #by average pooling
                # indicies = np.array([_aggregating_indices_synonyms_ave(indicies_per_datum,scores_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k) for indicies_per_datum,scores_per_datum in zip(indicies,scores)])
                # indicies = [wikipedia_id2local_id[local_id2wikipedia_id[int(indice)]] for indice in indicies]
                # indicies = list(dict.fromkeys(indicies))[:top_k]
                # assert len(indices) == top_k
                # indices = np.array(indices)
                # anything to be done here regarding the classification of NIL?
                # inference of NIL
                # if edge_cand_enrich:
                #     inds_seed = indicies[:top_k_cand_seed]
                #     print('inds_seed.tolist() in _run_biencoder():',inds_seed.tolist())
                #     inds_non_seed = indicies[top_k_cand_seed:]
                #     list_inds_enriched = enrich_edge_cands(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,inds_seed.tolist())
                #     if len(list_inds_enriched) < top_k: # enriched with original non-seeded part (from inds_non_seed) if not enough cands after the enrichment
                #         for ind_ in inds_non_seed:
                #             if not ind_ in list_inds_enriched:
                #                 list_inds_enriched.append(ind_) #list_inds_enriched = list_inds_enriched + [float("nan")] * (top_k - len(list_inds_enriched))
                #                 #set_inds_enriched.add(ind_)
                #                 if len(list_inds_enriched) == top_k:
                #                     break
                #     else:
                #         #list_inds_enriched = list(set_inds_enriched) 
                #         list_inds_enriched = list_inds_enriched[:top_k]
                #         #set_inds_enriched = set(list_inds_enriched)
                #     #list_inds_enriched = list(set_inds_enriched)
                #     indicies = np.array(list_inds_enriched)

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies) # nn is a list of numpy arrays
        all_scores_sorted.extend(scores_sorted)
        all_scores.extend(scores)
    print('labels:',len(labels),len(labels[0]))    # labels: 99 1
    print('nns:',len(nns),len(nns[0])) # nns: 99 100
    print('all_scores_sorted:',len(all_scores_sorted),len(all_scores_sorted[0])) # all_scores: 99 100
    print('all_scores:',len(all_scores),len(all_scores[0])) # all_scores: 99 100
    return labels, nns, all_scores_sorted, all_scores

def _process_crossencoder_dataloader(context_input, label_input, tensor_multi_hot_is_complete, tensor_is_NIL_labels, batch_size):
    tensor_data = TensorDataset(context_input, label_input, tensor_multi_hot_is_complete, tensor_is_NIL_labels) # this sets batch[0] and batch[1] in train_cross.evaluate(), and also adds batch[2], the list of boolean indicating whether each label is a NIL label
    sampler = SequentialSampler(tensor_data) # Samples elements sequentially, always in the same order.
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=batch_size
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda",ks=[1,3,5,10,50,100,150,200,250,300]):
    crossencoder.model.eval()
    ins_any = 0.0
    crossencoder.to(device)

    res = evaluate_edges(crossencoder, dataloader, device, logger, context_len, zeshel=False, silent=False, ks=ks) # here it uses the device - cuda or gpu
    #accuracy = res["normalized_accuracy"]
    #print('accuracy in _run_crossencoder:',accuracy)
    #logits = res["logits"]
    ins_at_1_all = res["ins_at_1_all"]
    ins_at_1_any = res["ins_at_1_any"]
    ins_at_5_all = res["ins_at_5_all"]
    ins_at_5_any = res["ins_at_5_any"]
    ins_at_10_all = res["ins_at_10_all"]
    ins_at_10_any = res["ins_at_10_any"]  
    ins_all = res["ins_all"]
    ins_any = res["ins_any"]    
    yhat_raw = res["yhat_raw"]
    yhat = res["yhat"]
    #num_tp_all = res["tp_all"]
    # num_tp_in_KB = res["tp_in_KB"]
    # num_tp_NIL = res["tp_NIL"]
    nb_eval_examples = res["nb_eval_examples"]
    # nb_eval_examples_in_KB = res["nb_eval_examples_in_KB"]
    # nb_eval_examples_NIL = res["nb_eval_examples_NIL"]
    # prec_in_KB = res["precision_in_KB"]
    # prec_NIL = res["precision_NIL"]
    # rec_in_KB = res["recall_in_KB"]
    # rec_NIL = res["recall_NIL"]
    # f1_in_KB = res["f1_in_KB"]
    # f1_NIL = res["f1_NIL"]

    # the accuracy will be changed if it setting with_NIL_infer as true, which infers NIL entities based on logits, but the output of predictions and logits will not be changed.
    return ins_at_1_all, ins_at_1_any, ins_at_5_all, ins_at_5_any, ins_at_10_all, ins_at_10_any, ins_all, ins_any, yhat, yhat_raw, nb_eval_examples

# save candidate lists for mentions (one list for a mention) to a .jsonl file
def _save_candidates(args, samples, nns, id2title, logger=None):
    if logger:
        logger.info("save candidates to %s" % args.candidate_path)
    
    if not os.path.exists(args.candidate_path):
        os.makedirs(args.candidate_path)    

    if args.use_BM25:
        candidate_fn = os.path.join(args.candidate_path,'%s_BM25_%s_%s_%d%s.jsonl' % (args.dataname, args.biencoder_model_name, args.cross_model_setting.replace('/','-'), args.top_k, ('-' + args.marking) if args.marking != '' else ''))
    else:
        candidate_fn = os.path.join(args.candidate_path,'%s_biencoder_%s_%s_%d%s.jsonl' % (args.dataname, args.biencoder_model_name, args.cross_model_setting.replace('/','-'), args.top_k, ('-' + args.marking) if args.marking != '' else ''))

    assert len(samples) == len(nns)
    list_mention_cand_json_str = []
    for ind_sample, sample in enumerate(samples):
        row_ind_men = sample['row_ind_men']
        mention_lower_cased = sample['mention']
        topk_indicies = nns[ind_sample]
        print('topk_indicies:',topk_indicies)
        #topk_indicies_str = [str(indice) for indice in topk_indicies]
        topk_tits = [id2title[indice] for indice in topk_indicies]

        # create a nested dict to form each json string
        mention_row_id_dict = {}
        cand_dict = {}
        mention_info_dict = {}
        
        for indice in topk_indicies:
            cand_dict[indice.item()] = id2title[indice.item()]
            # the order of indices in the cand_dict is preserved in the output

        mention_info_dict[mention_lower_cased] = cand_dict 
        mention_row_id_dict[row_ind_men] = mention_info_dict
        mention_cand_json_str = json.dumps(mention_row_id_dict)

        list_mention_cand_json_str.append(mention_cand_json_str)

    _output_to_file(candidate_fn,'\n'.join(list_mention_cand_json_str))

def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["no_cuda"] = args.no_cuda # default as False, i.e. w/ GPU
        biencoder_params["path_to_model"] = args.biencoder_model
        biencoder_params["bert_model"] = args.biencoder_bert_model # this updates the bert_model used (overwrites the one in the biencoder_config).
        biencoder_params["lowercase"] = args.lowercase
        biencoder_params["max_cand_length"] = int(args.max_cand_length)
        biencoder_params["max_seq_length"] = biencoder_params["max_context_length"] + biencoder_params["max_cand_length"]
        biencoder_params["use_context"] = args.use_context
        biencoder_params["eval_batch_size"] = int(args.eval_batch_size)
        print("biencoder bert model:", biencoder_params["bert_model"])
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["no_cuda"] = args.no_cuda # default as False, i.e. w/ GPU
            crossencoder_params["path_to_model"] = args.crossencoder_model
            crossencoder_params["bert_model"] = args.crossencoder_bert_model # this updates the bert_model used (overwrites the one in the crossencoder_config).
            crossencoder_params["lowercase"] = args.lowercase
            crossencoder_params["max_cand_length"] = int(args.max_cand_length)
            crossencoder_params["max_seq_length"] = crossencoder_params["max_context_length"] + crossencoder_params["max_cand_length"]
            crossencoder_params["use_context"] = args.use_context
            crossencoder_params["eval_batch_size"] = 1 #int(args.eval_batch_size)
            print("crossencoder bert model:", crossencoder_params["bert_model"])
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2synonyms,
        id2text,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
        wikipedia_id2ori_local_id,
        ori_local_id2wikipedia_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue, 
        args.entity_encoding, 
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2synonyms,
        id2text,
        wikipedia_id2local_id,
        local_id2wikipedia_id,
        wikipedia_id2ori_local_id,
        ori_local_id2wikipedia_id,
        faiss_indexer,
    )

# save prediction scores for mentions (one list for a mention) to a .jsonl file
def _save_predictions(args, samples, predictions, scores, id2title, logger=None):
    '''
    an example of samples - the first element
    samples[0]: {'context_left': 'palmoplantar pustulosis - a cross-sectional analysis in germany palmoplantar pustulosis (ppp) is a recalcitrant chronic ', 'mention': 'inflammatory skin disease', 'context_right': '. data relevant for the medical care of patients with ppp are scarce. thus, the aim of this work was to investigate the disease burden, clinical characteristics, and comorbidity of ppp patients in germany. ppp patients were examined in a crosssectional study at seven specialized psoriasis centers in germany. of the 172 included patients with ppp, 79.1% were female and 69.8% were smokers .in addition, 25.0% suffered from psoriasis vulgaris, 28.2% had documented psoriatic arthritis, and 30.2% had a family history of psoriasis. in 77 patients the mean dermatology life quality index (dlqi) was 12.2 ± 7.7 (mean ± sd). the mean psoriasis palmoplantar pustulosis area and severity index (pppasi) was 12.6 ± 8.6. mean body mass index was above average at 27.1 ± 5.5. the ppp patients', 'label_concept_UMLS': 'C3875321', 'label_concept': 36106, 'label_concept_ori': '703938007', 'entity_label_id': 64900, 'entity_label': '', 'entity_label_title': 'NIL', 'parent_concept': '95320005', 'child_concept': '277408007', 'parent': 'disorder of skin (disorder)', 'child': 'histologic type of inflammatory skin disorder (disorder)', 'edge_label_id': 36106, 'label_concept_str': ('95320005', '277408007'), 'row_ind_men': 0}
    '''
    if logger:
        logger.info("save predictions to %s" % args.prediction_path)
    
    if not os.path.exists(args.prediction_path):
        os.makedirs(args.prediction_path)    

    predictions_fn = os.path.join(args.prediction_path,'%s_%s_%s_%d%s.jsonl' % (args.dataname, args.biencoder_model_name, args.cross_model_setting.replace('/','-'), args.top_k,('-' + args.marking) if args.marking != '' else ''))

    list_ranked_edges_str = []

    # sort the scores and corresponding predictions for each row
    for prediction, score in zip(predictions,scores):
        sorted_ind = np.argsort(np.array(score))[::-1]
        #prediction_sorted = [prediction[ind_] for ind_ in sorted_ind]
        #score_sorted = [score[ind_] for ind_ in sorted_ind]
        #pred_str = ['-'.join(pred_edge) for pred_edge in prediction_sorted]
        list_ranked_edges_str.append(', '.join([' -> '.join(prediction[ind_]) + ' (' + str(score[ind_]) + ')' for ind_ in sorted_ind]))

    _output_to_file(predictions_fn,'\n'.join(list_ranked_edges_str))

def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2synonyms,
    id2text,
    wikipedia_id2local_id,
    local_id2wikipedia_id,
    wikipedia_id2ori_local_id,
    ori_local_id2wikipedia_id,
    faiss_indexer=None,
    test_data=None,
):
    #print('wikipedia_id2local_id in main_dense.run():',wikipedia_id2local_id)

    if not test_data and not args.test_mentions and not args.interactive:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entitied (--test_entities)"
        )
        raise ValueError(msg)

    # id2url = {
    #     v: "https://en.wikipedia.org/wiki?curid=%s" % k
    #     for k, v in wikipedia_id2local_id.items()
    # }

    stopping_condition = False
    while not stopping_condition:

        samples = None

        if args.interactive:
            logger.info("interactive mode")

            # biencoder_params["eval_batch_size"] = 1

            # Load NER model
            ner_model = NER.get_model()

            # Interactive
            text = input("insert text:")

            # Identify mentions
            samples = _annotate(ner_model, [text], lowercase=args.lowercase)

            _print_colorful_text(text, samples)

        else:
            if logger:
                logger.info("test dataset mode")

            if test_data:
                samples = test_data
            else:
                # Load test mentions
                samples = _get_test_samples(
                    args.test_mentions,
                    args.test_entities,
                    title2id,
                    wikipedia_id2local_id,
                    logger,
                    lowercase=args.lowercase,
                )

            stopping_condition = True

        # don't look at labels
        keep_all = (
            args.interactive
            or samples[0]["label_concept_str"] == "unknown"
            #or samples[0]["label_concept"] < 0            
        )
        print("keep_all:", keep_all)  # False
        # prepare the data for biencoder
        if logger:
            logger.info("preparing data for biencoder")
        print('args.use_NIL_tag:',args.use_NIL_tag)
        print('args.use_NIL_desc:',args.use_NIL_desc)
        print('args.use_NIL_desc_tag:',args.use_NIL_desc_tag)
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params,
            use_NIL_tag=args.use_NIL_tag,
            use_NIL_desc=args.use_NIL_desc,
            use_NIL_desc_tag=args.use_NIL_desc_tag,
        )

        if args.edge_cand_enrich:
            dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent = load_edge_catalogue(args.entity_catalogue)
        else:
            dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent = None, None, None, None
        # run biencoder
        if logger:
            logger.info("run biencoder")
        top_k = args.top_k
        if not args.use_BM25:
            labels, nns_ori, scores, scores_over_all_cands = _run_biencoder(
                biencoder, dataloader, candidate_encoding, local_id2wikipedia_id, wikipedia_id2local_id,top_k,aggregating_factor=args.aggregating_factor,indexer=faiss_indexer,#,edge_cand_enrich=args.edge_cand_enrich,dict_ind_edge_json_info=dict_ind_edge_json_info,dict_edge_tuple_to_ind=dict_edge_tuple_to_ind,dict_parent_to_child=dict_parent_to_child,dict_child_to_parent=dict_child_to_parent,top_k_cand_seed=args.top_k_cand_seed,
            )
        else:
            # get candidate ids (candidate_pool)
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(args.entity_ids)
            candidate_pool = torch.load(args.entity_ids)
            print('candidate_pool:',candidate_pool.size())
            index_title_special_token = biencoder.tokenizer.convert_tokens_to_ids(ENT_TITLE_TAG)
            # run BM25
            labels, nns_ori, scores = _run_BM25(
                dataloader,candidate_pool,local_id2wikipedia_id,wikipedia_id2local_id,top_k,
                index_title_special_token=index_title_special_token,
                aggregating_factor=args.aggregating_factor,
            )
        # if args.edge_cand_enrich:
        #     dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent = load_edge_catalogue(args.entity_catalogue)
        if args.edge_cand_enrich:
            logger.info('Enriching edge candidates')
            logger.info('args.top_k_cand_seed: %d' % args.top_k_cand_seed)
            nns = []
            for nn,score_over_all_cands in tzip(nns_ori,scores_over_all_cands):
                inds_seed = nn[:args.top_k_cand_seed]
                inds_non_seed = nn[args.top_k_cand_seed:]

                list_edge_inds_seed_w_scores = [(ind_seed, score_over_all_cands[ind_seed]) for ind_seed in inds_seed]
                list_inds_enriched_w_scores = enrich_edge_cands_w_scores(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds_seed_w_scores,score_over_all_cands=score_over_all_cands,use_leaf_edge_score=args.use_leaf_edge_score,LEAF_EDGE_SCORE=args.LEAF_EDGE_SCORE)
                
                # enriched with original non-seeded part (from inds_non_seed) if not enough cands after the enrichment
                if len(list_inds_enriched_w_scores) < top_k: 
                    list_inds_enriched = [ind for ind,_ in list_inds_enriched_w_scores]
                    for ind_ in inds_non_seed:
                        if not ind_ in list_inds_enriched:
                            list_inds_enriched_w_scores.append((ind_,score_over_all_cands[ind_])) #list_inds_enriched = list_inds_enriched + [float("nan")] * (top_k - len(list_inds_enriched))
                            #set_inds_enriched.add(ind_)
                            if len(list_inds_enriched_w_scores) == top_k:
                                break
                
                if args.edge_ranking_by_score:
                    # rank the list_inds_enriched
                    #list_inds_enriched_w_scores = [(ind, score_over_all_cands[ind]) for ind in list_inds_enriched]
                    #print("list_inds_enriched_w_scores:",list_inds_enriched_w_scores)                
                    list_inds_enriched_w_scores = sorted(list_inds_enriched_w_scores, key=lambda x: x[1],reverse=True) #sort the edges by score
                    #print("list_inds_enriched_w_scores, sorted:",list_inds_enriched_w_scores)
                
                # get the top-k after the ranking
                #list_inds_enriched = list(set_inds_enriched) 
                list_inds_enriched_w_scores_topk = list_inds_enriched_w_scores[:top_k]
                list_inds_enriched_topk = [ind for ind, _ in list_inds_enriched_w_scores_topk]
                #set_inds_enriched = set(list_inds_enriched)
                #list_inds_enriched = list(set_inds_enriched)
                nn = np.array(list_inds_enriched_topk)
                nns.append(nn)
        else:
            nns = nns_ori

        # save candidate generation results to .jsonl file
        print('nns:',len(nns),nns[0].size)
        if args.save_cand:
            _save_candidates(args,samples,nns,id2title,logger=logger)
        if args.cand_only:
            # quit the function after generating the candidates
            return (-1,)*19

        # infer out-of-KB entities from biencoder predicted scores, if existed, adapting the biencoder predicted scores
        local_id_NIL = wikipedia_id2local_id.get(args.NIL_concept,-1) # we added a general out-of-KB / NIL entity to the list - so that all out-of-KB entities share a common ID.
        if args.with_NIL_infer:                        
            #print('args:',args)
            print('th_NIL_bi_enc:',args.th_NIL_bi_enc)
            print('nns-pre-NIL-infer:',nns)
            if not args.use_BM25:
                nns = infer_out_KB_ent_bi_enc(scores,nns,NIL_ent_id=local_id_NIL,th_NIL_bi_enc=float(args.th_NIL_bi_enc))
            else:
                nns = infer_out_KB_ent_BM25(scores,nns,NIL_ent_id=local_id_NIL,th_NIL_BM25=0.0)
            print('nns-post-NIL-infer:',nns)
        
        # set NIL as the last top-k candidate if NIL is not predicted for the mention
        if args.set_NIL_as_cand:
            nns = set_NIL_to_candidates(nns,NIL_ent_id=local_id_NIL)

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            _print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()

            if args.fast:
                # use only biencoder
                continue

        else:

            biencoder_accuracy = -1
            recall_at = -1
            if not keep_all:
                # get recall values for all entities
                top_k = args.top_k
                x = []
                y = []
                print('labels:',len(labels),labels[0],type(labels[0]))
                print('nns:',len(nns),nns[0])
                for i in range(1, top_k+1):
                #for i in range(1, top_k+1-1): # +1 to ensure that the last one in the candidate list is reached (-1 to not count the added NIL when args.with_NIL_infer)
                    temp_y = 0.0
                    for ind_, (label, top) in enumerate(zip(labels, nns)):
                        #if label in top[:i]:
                        top_concept_ids = [local_id2wikipedia_id[local_id_] for local_id_ in top]
                        #label_normalised = _normalise_local_id(int(label),local_id2wikipedia_id,wikipedia_id2local_id)
                        if ori_local_id2wikipedia_id[int(label)] in top_concept_ids[:i]: # check the concept_id (e.g. CUI) instead of row_id (or local_id) as in the "entity as synonym" mode a concept_id usually matches to multiple row_ids.
                            temp_y += 1
                        elif i == top_k:
                            if ind_ < 5:
                                print('missed case (ind %d, within first 5):' % ind_,
                                      ori_local_id2wikipedia_id[int(label)], 'not in', top_concept_ids)
                    if len(labels) > 0:
                        temp_y /= len(labels)
                    x.append(i)
                    y.append(temp_y)
                #plt.plot(x, y) # plot k-recall@k line.
                biencoder_accuracy = y[0] # the top1 accurarcy
                recall_at = y[-1] # the top_k (k as 100) accuracy
                print("biencoder accuracy: %.4f" % biencoder_accuracy)
                print("biencoder recall@%d: %.4f" % (top_k, recall_at))
                #print("biencoder recall@%d: %.4f" % (top_k-1, recall_at))

            if args.fast:

                predictions = []
                for entity_list in nns:
                    sample_prediction = []
                    for e_id in entity_list:
                        e_title = id2title[e_id]
                        sample_prediction.append(e_title)
                    predictions.append(sample_prediction)

                # use only biencoder
                return (
                    biencoder_accuracy,
                    recall_at,
                    -1,
                    -1,
                    len(samples),
                    predictions,
                    scores,
                )
        
        # sys.exit(0)
        # now we reach the territory of crossencoder
        
        # setting to whether filter based on the results of biencoder
        filter_within = False

        # get device for inferencing with crossencoder
        #device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1) for all labels
        # prepare crossencoder data for all labels
        #print('id2synonyms:', str(len(id2synonyms)) + ' ' + str(list(id2synonyms.items())[:5]) if id2synonyms else 'None')#;sys.exit(0)
        context_input, context_input_ori, candidate_input, label_input, nns_filtered, tensor_is_NIL_labels, concept_id_ori_vecs = prepare_crossencoder_for_insertion_data(
            crossencoder.tokenizer, 
            samples, 
            labels, 
            nns, 
            id2title, 
            id2synonyms, 
            id2text, 
            ori_local_id2wikipedia_id, 
            local_id2wikipedia_id, 
            max_cand_length=crossencoder_params["max_cand_length"], 
            topk=top_k, 
            keep_all=keep_all, 
            filter_within=filter_within, 
            use_context=crossencoder_params["use_context"],
            test_NIL_label_only=None,
            #test_NIL_label_only as None here
            #NIL_ent_id=label_id_NIL,
            # use_NIL_tag=args.use_NIL_tag, 
            # use_NIL_desc=args.use_NIL_desc, 
            # use_NIL_desc_tag=args.use_NIL_desc_tag,
            # use_synonyms=args.use_synonyms,            
        )
        print('label_input[0]:',label_input[0])
        print('samples[0]:',samples[0])
        #sys.exit(0)        

        context_input = modify(
            context_input, candidate_input, crossencoder_params["max_seq_length"]
        )

        # make data multi-label
        # aggregating the labels over the context input
        multi_hot_size = context_input.size()[1]
        context_input, label_input, tensor_multi_hot_is_complete, tensor_is_NIL_labels, labels_aggregated, nns_aggregated = aggregate_into_multi_label(context_input, label_input, tensor_is_NIL_labels, multi_hot_size=multi_hot_size, concept_id_ori_vecs=concept_id_ori_vecs, context_input_ori=context_input_ori, labels=labels, nns=nns)
        print('label_input[0]-multi-label:',label_input[0])
        # print('samples[0]:',samples[0])
        #sys.exit(0)
        dataloader = _process_crossencoder_dataloader(
            context_input, label_input, tensor_multi_hot_is_complete, tensor_is_NIL_labels, crossencoder_params["eval_batch_size"]
        )

        # run crossencoder and get accuracy for all candidate labels
        ks = [1,3,5,10,50,100,150,200,250,300] # setting the @k values to be evaluated
        print('evaluate crossencoder on all candidate labels')
        #accuracy, index_array, unsorted_scores, num_corr_pred, nb_eval_examples = _run_crossencoder(
        #accuracy, predictions, logits, num_tp_all, nb_eval_examples
        #ins_at_1_all, ins_at_1_any, ins_all, ins_any, yhat, yhat_raw, nb_eval_examples
        ins_at_1_all, ins_at_1_any, ins_at_5_all, ins_at_5_any, ins_at_10_all, ins_at_10_any, ins_all, ins_any, yhat, yhat_raw, nb_eval_examples = _run_crossencoder(
            crossencoder,
            dataloader,
            logger=logger,
            context_len=biencoder_params["max_context_length"],
            device=device,
            ks=ks,
            # with_NIL_infer=args.with_NIL_infer, # here it sets to infer NIL from the logits
            # nns=nns if not filter_within else nns_filtered,
            # NIL_ent_id=local_id_NIL, # use the local_id_NIL
            # th_NIL_cross_enc=float(args.th_NIL_cross_enc),
            # # below are training params for cross-encoder
            # use_original_classification=args.use_ori_classification,
            # use_NIL_classification=args.use_NIL_classification, 
            # use_NIL_classification_infer=args.use_NIL_classification_infer,
            # lambda_NIL=args.lambda_NIL,
            # use_score_features=args.use_score_features,
            # use_score_pooling=args.use_score_pooling,
            # use_men_only_score_ft=args.use_men_only_score_ft,
            # use_extra_features=args.use_extra_features,
        )
        if not filter_within:
            assert nb_eval_examples == label_input.size(dim=0) # check whether the denominator in cross-encoder is the same as the number of samples/mentions in the data - this should be correct after using all data in the cross-encoder stage (i.e. no filtering based on bi-encoder results, but keeping them as -1 when passed to cross-encoder.)

        #print('accuracy:',accuracy)
        #print('index_array:',len(index_array),len(index_array[0]),type(index_array[0]),index_array)
        #print('unsorted_scores:',len(unsorted_scores),len(unsorted_scores[0]),type(unsorted_scores[0]),unsorted_scores) # these are logits.
        
        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            _print_colorful_text(text, samples)

            # print crossencoder prediction
            # to read this closely - v important
            idx = 0
            for entity_list, index_list, sample in zip(nns, yhat, samples): #index_array is the prediction
                e_id = entity_list[index_list[-1]] 
                # as each index_list in the index_array is sorted ascendingly, we fetch the last element in the index_list; 
                # as each index_list in the index_array is actually index in the corresponding nns (pred lists of indexes in bi-encoder), we use entity_list[index_list[-1]] to get the entity id.
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()
        else:
            # get the evaluation scores
            scores = []
            predictions = []
            pred_idx_tuples = []
            gold_idx_tuples = []
            print("labels_aggregated:",labels_aggregated)
            # loop over contextual mentions
            for gold_label_list, pred_label_list, yhat_one_row, yhat_raw_one_row in zip(
                labels_aggregated, nns_aggregated, yhat, yhat_raw
            ):

                yhat_one_row = yhat_one_row.tolist()

                # descending order
                #index_list.reverse()

                sample_prediction = []
                sample_pred_idx_tuples = []
                sample_scores = []
                for ind, index in enumerate(yhat_one_row):
                    #print('index:',index)
                    #if index == 1:
                    e_local_id_pred = pred_label_list[ind]
                    e_title_tuple_pred = id2title[e_local_id_pred]
                    e_idx_tuple_pred = local_id2wikipedia_id[e_local_id_pred]
                    sample_prediction.append(e_title_tuple_pred)
                    sample_pred_idx_tuples.append(e_idx_tuple_pred)
                    sample_scores.append(yhat_raw_one_row[ind])
                
                sample_gold_idx_tuples = [ori_local_id2wikipedia_id[e_ori_local_id_gold] for e_ori_local_id_gold in gold_label_list]      
                
                predictions.append(sample_prediction)
                pred_idx_tuples.append(sample_pred_idx_tuples)
                gold_idx_tuples.append(sample_gold_idx_tuples)
                scores.append(sample_scores)
            print("predictions[0]:",predictions[0])
            print("pred_idx_tuples[0]:",pred_idx_tuples[0])
            print("scores[0]:",scores[0])
            print("gold_idx_tuples[0]:",gold_idx_tuples[0])
            # crossencoder_normalized_accuracy = -1
            # overall_unormalized_accuracy = -1
            # crossencoder_normalized_in_KB_accuracy = -1
            # overall_unormalized_in_KB_accuracy = -1
            # crossencoder_normalized_NIL_accuracy = -1
            # overall_unormalized_NIL_accuracy = -1
            
            if args.measure_wp:
                ## WP SCORES CALCULATION
                # get the wp scores using pred_edges and gold_edges
                # pred_edges are from pred_idx_tuples
                # gold_edges are from gold_idx_tuples

                #load ontology and the taxonomy part of it for wu & palmer similarity
                onto_old = load_SNOMEDCT_deeponto(args.ontology_fn)
                dict_SCTID_onto_obj_prop = deeponto2dict_ids_obj_prop(onto_old)
                onto_old_taxo = extract_SNOMEDCT_deeponto_taxonomy(onto_old)

                #  embed and load the shortest node depth (snd) and lowest common ancestor (lca) dicts
                if 'SNOMEDCT' in args.ontology_fn:
                    onto_name = 'snomed-ct'
                    name_eles = args.ontology_fn.split('-')
                    onto_ver_subset = name_eles[3]
                elif 'DO' in args.ontology_fn:
                    onto_name = 'doid'
                    ontology_fn = args.ontology_fn
                    ontology_fn = ontology_fn[:ontology_fn.find('_')]
                    name_eles = ontology_fn.split('-')
                    onto_ver_subset = '-'.join(name_eles[1:])
                else:
                    logger.info('onto name cannot be inferred - please set in the code')
                    sys.exit(0)
                    
                snd_fn = "blink/prompting/%s-%s-%s.pkl" % (onto_name, onto_ver_subset, "shortest-node-depth")
                if os.path.exists(snd_fn):
                    with open(snd_fn, 'rb') as data_f:
                        dict_iri_to_snd = pickle.load(data_f)
                        logger.info('dict_iri_to_snd: %d' % len(dict_iri_to_snd))
                else:
                    dict_iri_to_snd = {}

                lca_fn = "blink/prompting/%s-%s-%s.pkl" % (onto_name, onto_ver_subset, "lowest-common-ancestor")
                if os.path.exists(lca_fn):
                    with open(lca_fn, 'rb') as data_f:
                        dict_iri_pair_to_lca = pickle.load(data_f)
                        logger.info('dict_iri_pair_to_lca: %d' % len(dict_iri_pair_to_lca))
                else:
                    dict_iri_pair_to_lca = {}
                    #dict_iri_pair_to_lca = get_dict_iri_pair_to_lca(snomed_ct_old_taxo)
                
                ks_wp = [1,5,10] # setting the @k values to be evaluated for WP metrics
                for top_k_value in ks_wp:
                    ave_overall_wp_sim_min = 0.
                    ave_overall_wp_sim_max = 0.
                    ave_overall_wp_sim_ave = 0.
                    # loop over mentions
                    wp_metric_denominator = 0 # the denominator as number of counted mentions (where the both pred and gold edges have at least one atmoic edges) for wu and palmer metric
                    n_mention = 0 # counting the number of mentions
                    for pred_idx_tuples_one_ment,gold_idx_tuples_one_ment in tqdm(zip(pred_idx_tuples,gold_idx_tuples),total=len(gold_idx_tuples)):
                        n_mention += 1
                        ## filter out complex edges
                        #pred_idx_tuples_one_ment = filter_out_complex_edges(pred_idx_tuples_one_ment)
                        #gold_idx_tuples_one_ment = filter_out_complex_edges(gold_idx_tuples_one_ment)

                        pred_edges = [(get_iri_from_SCTID_id(parent_idx,prefix=args.iri_prefix), get_iri_from_SCTID_id(child_idx,prefix=args.iri_prefix)) for parent_idx,child_idx in pred_idx_tuples_one_ment]
                        pred_edges = pred_edges[:top_k_value]
                        gold_edges = [(get_iri_from_SCTID_id(parent_idx,prefix=args.iri_prefix), get_iri_from_SCTID_id(child_idx,prefix=args.iri_prefix)) for parent_idx,child_idx in gold_idx_tuples_one_ment]

                        #assert len(pred_edges) != 0
                        if len(pred_edges) == 0:
                            logger.info("the mention number %d (starting from 0) does not have atomic pred edges" % n_mention) 
                            print('number of pred edges including complex:',len(pred_idx_tuples_one_ment))
                            print('pred edges including complex:', pred_idx_tuples_one_ment)
                            continue
                        
                        #assert len(gold_edges) != 0
                        if len(gold_edges) == 0:
                            logger.info("the mention number %d (starting from 0) does not have atomic gold edges" % n_mention) 
                            print('number of gold edges including complex:',len(gold_idx_tuples_one_ment))
                            print('gold edges including complex:', gold_idx_tuples_one_ment)
                            continue

                        men_overall_edge_wp_sim_ave,men_overall_edge_wp_sim_min,men_overall_edge_wp_sim_max, dict_iri_to_snd, dict_iri_pair_to_lca = overall_edge_wp_sim(onto_old_taxo,pred_edges,gold_edges,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca)
                        #men_overall_edge_wp_sim_ave,men_overall_edge_wp_sim_min,men_overall_edge_wp_sim_max, dict_iri_to_snd, dict_iri_pair_to_lca = overall_edge_wp_sim_w_comp(onto_old_taxo,dict_SCTID_onto_obj_prop,pred_edges,gold_edges,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca,iri_prefix=args.iri_prefix)
                        ave_overall_wp_sim_ave += men_overall_edge_wp_sim_ave
                        ave_overall_wp_sim_min += men_overall_edge_wp_sim_min
                        ave_overall_wp_sim_max += men_overall_edge_wp_sim_max
                        wp_metric_denominator += 1

                    ave_overall_wp_sim_ave = ave_overall_wp_sim_ave / wp_metric_denominator
                    ave_overall_wp_sim_min = ave_overall_wp_sim_min / wp_metric_denominator
                    ave_overall_wp_sim_max = ave_overall_wp_sim_max / wp_metric_denominator
                    
                    logger.info('wp_at_%d_ave: %.5f' % (top_k_value, ave_overall_wp_sim_ave))
                    logger.info('wp_at_%d_min: %.5f' % (top_k_value, ave_overall_wp_sim_min))
                    logger.info('wp_at_%d_max: %.5f' % (top_k_value, ave_overall_wp_sim_max))
                    logger.info('wp_at_%d_denominator: %d' % (top_k_value, wp_metric_denominator))

                # save shortest node depth (snd) and lowest common ancestor (lca) dicts
                with open(snd_fn, 'wb') as data_f:
                    pickle.dump(dict_iri_to_snd, data_f)        
                    logger.info('shortest node depth dict stored to %s' % snd_fn)
                with open(lca_fn, 'wb') as data_f:
                    pickle.dump(dict_iri_pair_to_lca, data_f)        
                    logger.info('lowest common ancestor dict stored to %s' % lca_fn)

            if not keep_all:
                # 1) for all entities
                # crossencoder_normalized_accuracy = accuracy
                # print(
                #     "crossencoder normalized accuracy: %.4f"
                #     % crossencoder_normalized_accuracy
                # )
                #print('label_input',label_input)
                #label_input tensor([ 0,  0, 25, 44,  0,  1, 42,  0,  6,  0,  0,  1, 27, 20,  1,  1, 24, 94,
                #0,  0,  0,  0,  1,  1,  0,  0,  5,  1,  0,  0,  3,  0, 26,  0,  0,  0,
                #0, 23,  0,  1,  0,  0,  0,  0,  0,  1,  1, 35, 20,  3,  0,  0, 27,  0,
                #0,  0, 62,  0, 16,  0])
                #print('samples',samples)
                print('label_input/samples-aggregated:%d/%d' % (len(label_input),context_input.size()[0]))
                # if len(samples) > 0:
                #     overall_unormalized_accuracy = (
                #         crossencoder_normalized_accuracy * len(label_input) / len(samples)
                #         # unnormalise the crossencoder accuracy to the scale of whole samples (from the scale of retrieved candidates by the bi-encoder), where label_input is the number of input correctly predicted by the candidate generator (bi-encoder). So this is actually just factored by the recall (recall@100) of the bi-encoder.
                #     )
                # print(
                #     "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
                # )
                assert not filter_within
                # if not filter_within:
                #     assert overall_unormalized_accuracy == crossencoder_normalized_accuracy # the unormalised metric should be the same as the cross-encoder normalised metric when we do not filter the data based on biencoder results (keeping them as -1 and passed to cross-encoder).

                # save predictions
                _save_predictions(args, samples, predictions, scores, id2title, logger=logger)

            return (
                biencoder_accuracy,
                recall_at,
                # biencoder_in_KB_accuracy,
                # recall_in_KB_at,
                #biencoder_NIL_accuracy,
                #recall_NIL_at,
                # crossencoder_normalized_accuracy,
                # overall_unormalized_accuracy,
                ins_at_1_all,
                ins_at_1_any,
                ins_at_5_all,
                ins_at_5_any,
                ins_at_10_all,
                ins_at_10_any,
                ins_all,
                ins_any,
                # prec_in_KB, 
                # rec_in_KB, 
                # f1_in_KB, 
                # prec_NIL, 
                # rec_NIL, 
                # f1_NIL, 
                # crossencoder_normalized_in_KB_accuracy,
                # overall_unormalized_in_KB_accuracy,
                # crossencoder_normalized_NIL_accuracy,
                # overall_unormalized_NIL_accuracy,
                len(samples),
                # nb_eval_examples_in_KB,
                # nb_eval_examples_NIL,
                predictions,
                scores,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset."
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities."
    )

    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.", 
    )

    # biencoder
    parser.add_argument(
        "--biencoder_bert_model",
        dest="biencoder_bert_model",
        type=str,
        default="bert_large_uncased",
        help="the type of the bert model, as specified in the huggingface model hub.",
    )
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_bert_model",
        dest="crossencoder_bert_model",
        type=str,
        default="bert_base_uncased",
        help="the type of the bert model, as specified in the huggingface model hub.",
    )
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    models = load_models(args, logger)
    run(args, logger, *models)
