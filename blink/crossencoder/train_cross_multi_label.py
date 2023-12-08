# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
#from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

import blink.candidate_retrieval.utils
#from blink.biencoder.candidate_analysis import enrich_edge_cands
from blink.crossencoder.crossencoder_multi_label import CrossEncoderRanker, load_crossencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.candidate_ranking.multi_label_evaluation as multi_label_eval
#import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from blink.out_of_KB_utils import infer_out_KB_ent_cross_enc, infer_out_KB_ent_cross_enc_classify

logger = None


def modify(context_input, candidate_input, max_seq_length):
    print('max_seq_length:',max_seq_length)
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    # pair each context input with each cand input in the batch
    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate - so the length also minus 1
            sample = cur_input + cur_candidate[j][1:] 
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)

def aggregate_into_multi_label(context_input, label_input, label_is_NIL_input, multi_hot_size=300, concept_id_ori_vecs=None,context_input_ori=None,labels=None, nns=None):
    '''
    aggregating the mention-edge-pair data into multi-label format - 
    
    label_input is the labels for the cross-encoder (i.e. from bi-encoder's top-k)
    labels mean the original labels for contextual mention

    regarding concept_id_ori_vecs: the mention's original concept ID in the KB (before KB versioning) - then to be encoded (here using a tokeniser) - this is used for disambiguate different data rows (same mention w/o context, but matched to two ori concept ids, then later can be inserted into two different set of edges)
    
    regarding context_input_ori: provided to aggregate better, as now we assume that diff context means diff mentions and the same contexts are aggregated, w/ "dict_context_to_label_inputs[cur_input]"
    '''

    # for the predictions from the bi-encoder
    if labels==None:
        labels_aggregated = None
    if nns==None:
        nns_aggregated = None
    
    dict_context_to_doc_ind = {}
    dict_context_to_label_inputs = {}
    dict_context_to_is_NIL = {}
    dict_context_to_labels = {}
    dict_context_to_nns = {}

    for i in range(len(context_input)):
        # get current inputs (w/ format cast into different types)
        # as a tuple, where the first is the context_input (can only contain mention if use_context is False)
        #                   the second is the context_id_ori_vecs (for disambiguation if same mention matched to two concepts in the old KB)
        #                   the third is the context_input_ori (for disambiguation if use_context is False)
        cur_input = (str(context_input[i].tolist()),
                     str(concept_id_ori_vecs[i].tolist()) if concept_id_ori_vecs != None else '',
                     str(context_input_ori[i].tolist()) if context_input_ori != None else '',                     
        )
        cur_label_input = label_input[i].item()
        cur_label_is_NIL_input = label_is_NIL_input[i]
        # save the context (or mention) input into dict_context_to_label_inputs
        if not cur_input in dict_context_to_label_inputs:
            # for doc_inds
            dict_context_to_doc_ind[cur_input] = [str(i)]
            # for label inds
            dict_context_to_label_inputs[cur_input] = [cur_label_input]
            if labels != None:
                # for label aggregation
                dict_context_to_labels[cur_input] = [labels[i].item()]
            if nns != None:
                # for prediction aggregation
                dict_context_to_nns[cur_input] = nns[i]
        else:
            # for doc_inds
            list_doc_inds = dict_context_to_doc_ind[cur_input]
            if not str(i) in list_doc_inds:
                list_doc_inds.append(str(i))
            dict_context_to_doc_ind[cur_input] = list_doc_inds
            # for label inds
            list_label_input = dict_context_to_label_inputs[cur_input]
            if not cur_label_input in list_label_input:
                list_label_input.append(cur_label_input)
            dict_context_to_label_inputs[cur_input] = list_label_input
            # for ori gold labels
            if labels != None:
                list_labels = dict_context_to_labels[cur_input]
                if not labels[i].item() in list_labels:
                    list_labels.append(labels[i].item())
                dict_context_to_labels[cur_input] = list_labels

        # save the context -> is NIL dict    
        dict_context_to_is_NIL[cur_input] = cur_label_is_NIL_input
    # print doc_ind_aggregated_strs
    lst_doc_ind_aggregated_str = ['|'.join(doc_ind_lst) for doc_ind_lst in dict_context_to_doc_ind.values()]
    print('doc_ind_aggregated_strs:\n','\n'.join(lst_doc_ind_aggregated_str))
    print('context_input:',len(context_input))#,context_input)
    print('dict_context_to_label_inputs:',len(dict_context_to_label_inputs))#,dict_context_to_label_inputs)
    context_output = torch.LongTensor([eval(key_tuple[0]) for key_tuple in dict_context_to_label_inputs.keys()])
    # list_label_input as a padded tensor of multiple labels. (this is not used)
    #label_output = [torch.LongTensor(list_label_input_) for list_label_input_ in dict_context_to_label_inputs.values()]
    #label_output = torch.LongTensor(pad_sequence(label_output, batch_first=True,padding_value=-1.0))

    # turn the list_label_input into a multi-hot representation
    label_multi_hot_is_complete = [] # recording whether the multi-hot aggregation is complete, i.e. if the multi-hot output has all the gold labels (due to the top-k constraint).
    label_output = [list_label_input_ for list_label_input_ in dict_context_to_label_inputs.values()]
    print("label_single-label[0]:",label_output[0])
    #multi_hot_size = max_in_2d(label_output)+1
    list_multi_hot = [[0.]*multi_hot_size for _ in range(len(label_output))]
    for row_ind, label_output_row in enumerate(label_output):
        multi_hot_is_complete = True
        for label_ind in label_output_row:
            #print("label_ind:",label_ind)
            if label_ind != -1:
                # print(
                #     "updating list_multi_hot (row, col):", row_ind, label_ind
                # )
                list_multi_hot[row_ind][label_ind] = 1.
            else:
                # this information needs to be stored as well: whether all labels are recorded in the multi-hot representation.
                multi_hot_is_complete = False
        label_multi_hot_is_complete.append(multi_hot_is_complete)
    label_multi_hot_output = torch.Tensor(list_multi_hot)
    print("label_multi-label[0]:",label_multi_hot_output[0])
    label_multi_hot_is_complete = torch.Tensor(label_multi_hot_is_complete).bool() # if the multi-hot output has all the gold labels (due to the top-k constraint)
    print('label_multi_hot_is_complete:',label_multi_hot_is_complete)
    label_is_NIL_output = torch.Tensor(list(dict_context_to_is_NIL.values())).bool()
    # get labels aggregation
    if labels != None:
        labels_aggregated = list(dict_context_to_labels.values())
    # get nns pred aggregation
    if nns != None:
        nns_aggregated = list(dict_context_to_nns.values())
    return context_output, label_multi_hot_output, label_multi_hot_is_complete, label_is_NIL_output, labels_aggregated, nns_aggregated

def max_in_2d(lst):
    # This will flatten the list and find the max value.
    return max(max(sublist) for sublist in lst)

# evaluate the predicted edges for insertion
def evaluate_edges(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True, ks=[1,3,5,10,50,100,150,200,250,300]): # nns_filtered is the nns of mentions which have correct candidates generated by the bi-encoder  

    #assert len(list(eval_dataloader)) == len(nns)
    # 'device' sets whether gpu or cpu is to be used
    reranker.model.eval() # set to eval mode
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")
    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    y, yhat, yhat_raw = [], [], []
    list_multi_hot_is_complete = []
    all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2] # batch[2] seems now occupied by tensor_is_NIL_labels, this one to be fixed later.
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        data_size_batch = context_input.size(0)
        label_input = batch[1] # now my labels contain value -1 or 100 - issue to fix: /pytorch/aten/src/THCUNN/ClassNLLCriterion.cu:108: cunn_ClassNLLCriterion_updateOutput_kernel: block: [0,0,0], thread: [0,0,0] Assertion `t >= 0 && t < n_classes` failed. # it feels like that I am operating a large machine - BLINK is pretty complex. - solved by noy calculating loss in reranker() below
        if len(batch)>3: #TODO: fix this with zeshel case
            tensor_multi_hot_is_complete = batch[2]
            tensor_is_NIL_labels = batch[3]            
        else:
            tensor_is_NIL_labels = None
            # for training - this is not yet included in the batch, so set as None, but we can inlcude this in the batch
        if len(batch)>4: #TODO: fix this with zeshel case
            mention_matchable_fts = batch[3]
        else:
            mention_matchable_fts = None # this is so far for the inference stage, see main_dense._process_cross_encoder_dataloader. TODO
        #print('label input in batch %d' % step, label_input)
        #print('tensor_is_NIL_labels in batch %d:' % step, tensor_is_NIL_labels)
        
        row_multi_hot_is_complete = tensor_multi_hot_is_complete.cpu().numpy()
        label_ids = label_input.cpu().numpy() 
        # here it gets the predictions
        with torch.no_grad():
            logits = reranker(context_input, label_input, context_length, inference_only=True,label_is_NIL_input=tensor_is_NIL_labels) # setting inference_only as True to avoid calculating loss 
            # this finally calls CrossEncoderRanker.forward() in crossencoder.py
            logit_after_sigmoid = F.sigmoid(logits)
        
        #if not silent:
        #   print('logits:',logits.size())
        logits = logits.detach().cpu().numpy()
        #is_NIL_labels = tensor_is_NIL_labels.cpu().numpy() if not tensor_is_NIL_labels is None else None
        #print('is_NIL_labels:',is_NIL_labels)
        # you don't know what the label for the label_id is, as here it only encodes where it is appeared in the nn predictions from the biencoder. so you need tensor_is_NIL_labels.
        
        #ind_out = np.argmax(logits, axis=1) # get the predicted index of the output
        logit_after_sigmoid = logit_after_sigmoid.data.cpu().numpy()
        pred_multi_hot = np.round(logit_after_sigmoid)
        #print('pred_multi_hot:',pred_multi_hot) # an array of the predicted indexes (each is an index of the candidates from the bi-encoder)
            
        # get the batch of nns predictions
        #print('nns:',len(nns))
        #nns_batch = nns[step*data_size_batch:(step+1)*data_size_batch]

        #print('ind_out:',ind_out) # an array of the predicted indexes (each is an index of the candidates
        #assert np.all(ind_out <= 99)
        #print('label_ids in train_cross_multi_label.evaluate_edges():',label_ids)
        
        #tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)
        #tmp_eval_accuracy, eval_result = utils.accuracy_from_ind(ind_out, label_ids)
        # the first argument output, tmp_eval_accuracy, is the num of accurate instances in the batch
        #print('tmp_eval_accuracy:',tmp_eval_accuracy)
        #print('eval_result:',eval_result)
        #print('tmp_eval_accuracy_in_KB:',tmp_eval_accuracy_in_KB, eval_result_in_KB)
        #print('tmp_eval_accuracy_NIL:',tmp_eval_accuracy_NIL, eval_result_NIL)

        #eval_accuracy += tmp_eval_accuracy
        list_multi_hot_is_complete.append(row_multi_hot_is_complete)
        y.append(label_ids)
        yhat_raw.append(logit_after_sigmoid)
        yhat.append(pred_multi_hot)
        all_logits.extend(logits)

        # get the number of eval and pred examples in each category of all, in-KB, NIL/out-of-KB.
        nb_eval_examples += data_size_batch

        if zeshel:
            for i in range(data_size_batch):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1
        
    normalized_eval_accuracy = -1
    # #print('nb_eval_examples:',nb_eval_examples)
    # if nb_eval_examples > 0:
    #     normalized_eval_accuracy = eval_accuracy / nb_eval_examples # so this is actually recall. but what is precision? for all labels (in-KB + out-of-KB/NIL), there is no differences in recall or precision, as the model predicts from all data and will predict it to be a
    
    # if zeshel:
    #     macro = 0.0
    #     num = 0.0 
    #     for i in range(len(WORLDS)):
    #         if acc[i] > 0:
    #             acc[i] /= tot[i]
    #             macro += acc[i]
    #             num += 1
    #     if num > 0:
    #         logger.info("Macro accuracy: %.5f" % (macro / num))
    #         logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    # else:
    #     if logger:
    #         logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)

    np_multi_hot_is_complete = np.squeeze(np.array(list_multi_hot_is_complete))
    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)
    #ks = [1,3,5,10,50,100,150,200,250,300]
    num_labels = len(y[0])
    print("num_labels:",num_labels)
    ks = [k for k in ks if k <= num_labels]
    results = multi_label_eval.all_metrics(yhat, y, k=ks, yhat_raw=yhat_raw, is_multi_hot_complete=np_multi_hot_is_complete)
    
    multi_label_eval.print_metrics(results)

    #results["tp_all"] = eval_accuracy
    #print("tp_all:",eval_accuracy,'tp_in_KB:',eval_tp_in_KB,'tp_NIL:',eval_tp_NIL)
    #print("eval_prec_rec_f1_in_KB:",eval_precision_in_KB,eval_recall_in_KB,results["f1_in_KB"])
    #print("eval_prec_rec_f1_NIL:",eval_precision_NIL,eval_recall_NIL,results["f1_NIL"])
    # print('eval_precision_in_KB:',eval_precision_in_KB)
    # print('eval_recall_in_KB:',eval_recall_in_KB)
    # print('eval_precision_NIL:',eval_precision_NIL)
    # print('eval_recall_NIL:',eval_recall_NIL)
    #results["normalized_accuracy"] = normalized_eval_accuracy # actually this is recall - for the "all labels" setting only.
    # print('normalized_eval_accuracy in train_cross.evaluate():',normalized_eval_accuracy)
    results["nb_eval_examples"] = nb_eval_examples
    #print("nb_eval_all:",nb_eval_examples,"in_KB:",nb_eval_examples_in_KB,"NIL:",nb_eval_examples_NIL)
    #results["logits"] = all_logits
    results["yhat_raw"] = yhat_raw
    results["yhat"] = yhat
    results["y"] = y # also output the gold 
    # if logger:
    #     #logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    #     logger.info("tp_all: %d" % (eval_accuracy))
    #     logger.info("nb_eval_all: %d" % (nb_eval_examples))
    return results

def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
       optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    ) # source code here https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    ''' Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    ''' # so epochs have an effect on warmup_steps and WarmupLinearSchedule. 

    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    # Fix the random seeds (part 1)
    if params["fix_seeds"]:
        seed = params["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # display model training settings
    if params["use_ori_classification"]:
        logger.info("use_ori_classification")
    if params["use_NIL_classification"]:
        logger.info("use_NIL_classification")
        logger.info("lambda_NIL: %s", str(params["lambda_NIL"]))
        if params["use_score_features"]:
            logger.info("use_score_features")
        if params["use_score_pooling"]:
            logger.info("use_score_pooling")    
        if params["use_men_only_score_ft"]:
            logger.info("use_men_only_score_ft")
        if params["use_extra_features"]:
            logger.info("use_extra_features")   
        if params["use_NIL_classification_infer"]:
            logger.info("use_NIL_classification_infer")            
    # if params
    # logger.info('')

    # Init model - and here we also expect extra features 
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu
    if params["fix_seeds"]:
        # Fix the random seeds (part 2)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    
    fname = os.path.join(params["data_path"], "train.t7") ## this file is generated with eval_biencoder.py (see https://github.com/facebookresearch/BLINK/issues/92#issuecomment-1126293605) # using train-all (the concatenation of train, valid, test-in-KB)
    train_data = torch.load(fname)
    context_input = train_data["context_vecs"]
    context_input_ori = train_data["contextual_vecs"]
    candidate_input = train_data["candidate_vecs"]
    label_input = train_data["labels"]
    label_is_NIL_input = train_data["labels_is_NIL"]
    concept_id_ori_vecs = train_data["concept_id_ori_vecs"]
    #mention_matchable_fts = train_data["mention_matchable_fts"]
    #nns = train_data["entity_inds"]
    #print('nns:',len(nns),len(nns[0]))
    if params["debug"]:
        max_n = params["debug_max_lines"]
        context_input = context_input[:max_n]
        print('context_input:',context_input[:10])
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
        label_is_NIL_input = label_is_NIL_input[:max_n]
        print('label_is_NIL_input:',label_is_NIL_input[:10])
        #mention_matchable_fts = mention_matchable_fts[:max_n]
        #nns = nns[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    print('context_input',context_input.size()) 
    #context_input torch.Size([200, 100, 159]) debug mode
    #context_input torch.Size([5006, 100, 159])
    print('label_input:',label_input.size(),label_input) 
    #label_input: torch.Size([5006]) tensor([ 0,  3,  0,  ...,  5, 18,  0])

    # make data multi-label
    # aggregating the labels over the context input
    multi_hot_size = context_input.size()[1]
    context_input, label_input, label_multi_hot_is_complete, label_is_NIL_input, _, _ = aggregate_into_multi_label(context_input, label_input, label_is_NIL_input, multi_hot_size=multi_hot_size,concept_id_ori_vecs=concept_id_ori_vecs,context_input_ori=context_input_ori)

    print('context_input[0]:',context_input[0])
    print('label_input[0]:',label_input[0])
    print('label_is_NIL_input[0]:',label_is_NIL_input[0])
    
    if params["zeshel"]:
        src_input = train_data['worlds'][:len(context_input)]
        train_tensor_data = TensorDataset(context_input, label_input, src_input, label_multi_hot_is_complete, label_is_NIL_input) #, mention_matchable_fts
    else:
        train_tensor_data = TensorDataset(context_input, label_input, label_multi_hot_is_complete, label_is_NIL_input) #, mention_matchable_fts
    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )

    fname = os.path.join(params["data_path"], "valid.t7") ## using valid-NIL instead of valid (in-KB)
    valid_data = torch.load(fname)
    context_input = valid_data["context_vecs"]
    context_input_ori = valid_data["contextual_vecs"]
    candidate_input = valid_data["candidate_vecs"]
    label_input = valid_data["labels"]
    label_is_NIL_input = valid_data["labels_is_NIL"]
    concept_id_ori_vecs = valid_data["concept_id_ori_vecs"]
    nns_valid = valid_data["entity_inds"]
    #mention_matchable_fts = valid_data["mention_matchable_fts"]
    #print('nns_valid:',nns_valid,len(nns_valid))#,len(nns_valid[0]))
    if params["debug"]:
        max_n = params["debug_max_lines"]
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
        nns_valid = nns_valid[:max_n]
        #mention_matchable_fts = mention_matchable_fts[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)

    # aggregating the labels over the context input
    context_input, label_input, label_multi_hot_is_complete, label_is_NIL_input, _, _ = aggregate_into_multi_label(context_input, label_input, label_is_NIL_input, multi_hot_size=multi_hot_size,concept_id_ori_vecs=concept_id_ori_vecs,context_input_ori=context_input_ori)

    if params["zeshel"]:
        src_input = valid_data["worlds"][:len(context_input)]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input, label_multi_hot_is_complete, label_is_NIL_input)#, mention_matchable_fts)
    else:
        valid_tensor_data = TensorDataset(context_input, label_input, label_multi_hot_is_complete, label_is_NIL_input)#, mention_matchable_fts)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    # evaluate before training
    results = evaluate_edges(
        reranker,
        valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        zeshel=params["zeshel"],
        silent=params["silent"],
        # nns=nns_valid, 
        # NIL_ent_id=params["NIL_ent_ind"],
        # use_original_classification=params["use_ori_classification"],
        # use_NIL_classification=params["use_NIL_classification"],
        # use_NIL_classification_infer=params["use_NIL_classification_infer"],
        # lambda_NIL=params["lambda_NIL"],
        # use_score_features=params["use_score_features"],
        # use_score_pooling=params["use_score_pooling"],
        # use_men_only_score_ft=params["use_men_only_score_ft"],
        # use_extra_features=params["use_extra_features"],
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = (-1)
    best_score = -1

    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            if params["limit_by_train_steps"]:
                iter_ = tqdm(train_dataloader, 
                         desc="Batch", 
                         total=min(len(train_dataloader),params["max_num_train_steps"]))
            else:
                iter_ = tqdm(train_dataloader, 
                         desc="Batch")
        part = 0
        for step, batch in enumerate(iter_):
            if params["limit_by_train_steps"] and step == params["max_num_train_steps"]:
                break
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0] 
            label_input = batch[1]
            label_multi_hot_is_complete = batch[2]
            label_is_NIL_input = batch[3]
            #mention_matchable_fts = batch[3]
            loss, _ = reranker(context_input, label_input, context_length, label_is_NIL_input=label_is_NIL_input,
                                #   use_original_classification=params["use_ori_classification"],
                                #   use_NIL_classification=params["use_NIL_classification"],lambda_NIL=params["lambda_NIL"],use_score_features=params["use_score_features"],
                                #   use_score_pooling=params["use_score_pooling"],
                                #   use_men_only_score_ft=params["use_men_only_score_ft"],
                                #   use_extra_features=params["use_extra_features"],mention_matchable_fts=None,
                                  )

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                results = evaluate_edges(
                    reranker,
                    valid_dataloader,
                    device=device,
                    logger=logger,
                    context_length=context_length,
                    zeshel=params["zeshel"],
                    silent=params["silent"],
                    # nns=nns_valid, 
                    # NIL_ent_id=params["NIL_ent_ind"],
                    # use_original_classification=params["use_ori_classification"],
                    # use_NIL_classification=params["use_NIL_classification"],
                    # use_NIL_classification_infer=params["use_NIL_classification_infer"],
                    # lambda_NIL=params["lambda_NIL"],
                    # use_score_features=params["use_score_features"],
                    # use_score_pooling=params["use_score_pooling"],
                    # use_men_only_score_ft=params["use_men_only_score_ft"],
                    # use_extra_features=params["use_extra_features"],
                )
                ls = [best_score, results["ins_at_10_any"]]
                li = [best_epoch_idx, (epoch_idx, part)]

                best_score = ls[np.argmax(ls)]
                best_epoch_idx = li[np.argmax(ls)]

                if params["save_model_epoch_parts"]:
                    logger.info("***** Saving fine - tuned model *****")
                    epoch_output_folder_path = os.path.join(
                    model_output_path, "epoch_{}_{}".format(epoch_idx, part)
                    )
                    part += 1
                    utils.save_model(model, tokenizer, epoch_output_folder_path)
                    # utils.write_to_file( # also save the training parameters
                    #    os.path.join(epoch_output_folder_path, "training_params.txt"), str(params)
                    # )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        # utils.write_to_file(
        #     os.path.join(epoch_output_folder_path, "training_params.txt"), str(params)
        # )
        # reranker.save(epoch_output_folder_path)

        # this is the evaluation after each epoch run, also on the development set.
        #output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate_edges(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
            # nns=nns_valid, 
            # NIL_ent_id=params["NIL_ent_ind"],
            # use_original_classification=params["use_ori_classification"],
            # use_NIL_classification=params["use_NIL_classification"],
            # use_NIL_classification_infer=params["use_NIL_classification_infer"],
            # lambda_NIL=params["lambda_NIL"],
            # use_score_features=params["use_score_features"],
            # use_score_pooling=params["use_score_pooling"],
            # use_men_only_score_ft=params["use_men_only_score_ft"],
            # use_extra_features=params["use_extra_features"],
        )

        ls = [best_score, results["ins_at_10_any"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    if type(best_epoch_idx)==int:
        logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    else:
        assert type(best_epoch_idx)==tuple and len(best_epoch_idx) == 2
        logger.info("Best performance in epoch: {}_{}".format(best_epoch_idx[0],best_epoch_idx[1]))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx) if type(best_epoch_idx)==int else "epoch_{}_{}".format(best_epoch_idx[0],best_epoch_idx[1]),
        WEIGHTS_NAME,
    )
    reranker = load_crossencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate_edges(params, logger=logger) #TODO: check here
        
if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
