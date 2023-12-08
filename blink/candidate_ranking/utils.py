# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import io
import sys
import json
import torch
import logging

import numpy as np

from collections import OrderedDict
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME
from tqdm import tqdm

from blink.candidate_ranking.bert_reranking import BertReranker
from blink.biencoder.biencoder import BiEncoderRanker
import random

def read_dataset_from_fn(txt_file_path,limit_by_max_lines=False, max_lines=300000, debug=False,debug_max_lines=200, random_sampling=False):
    # get line counts
    line_count = count_lines(txt_file_path)
    
    # sample random data line numbers (for the debugging or sampling mode), if chosen to
    if random_sampling and debug:
        if debug_max_lines <= line_count:
            # randomly sample k line numbers
            list_rand_line_nums = random.Random(1234).sample(range(line_count), debug_max_lines) 
            print("list_rand_line_nums:",list_rand_line_nums)        
        else:
            print("sample size %s larger than line numbers" % debug_max_lines)
            random_sampling = False    

    samples = []

    with io.open(txt_file_path, mode="r", encoding="utf-8-sig") as file: # changed from utf-8 to utf-8-sig when needed
        for ind, line in enumerate(tqdm(file,total=line_count)):
            if not (random_sampling and debug):
                samples.append(json.loads(line.strip()))
            if debug:
                if not random_sampling:
                    if ind == debug_max_lines - 1:
                        break
                else:
                    if ind in list_rand_line_nums:
                        samples.append(json.loads(line.strip()))
            if limit_by_max_lines and ind == max_lines - 1:
                break
    return samples
        
def read_dataset(dataset_name, preprocessed_json_data_parent_folder, limit_by_max_lines=False, max_lines=300000, debug=False,debug_max_lines=200, random_sampling=False):
    '''
    "random" works only when "debug" is set True
    '''
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(preprocessed_json_data_parent_folder, file_name)
    
    samples = read_dataset_from_fn(txt_file_path,limit_by_max_lines=limit_by_max_lines, max_lines=max_lines, debug=debug,debug_max_lines=debug_max_lines, random_sampling=random_sampling)

    return samples

def count_lines(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

def filter_samples(samples, top_k, gold_key="gold_pos"):
    if top_k == None:
        return samples

    filtered_samples = [
        sample
        for sample in samples
        if sample[gold_key] > 0 and sample[gold_key] <= top_k
    ]
    return filtered_samples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def eval_precision_bm45_dataloader(dataloader, ks=[1, 5, 10], number_of_samples=None):
    label_ids = torch.cat([label_ids for _, _, _, label_ids, _ in dataloader])
    label_ids = label_ids + 1
    p = {}

    for k in ks:
        p[k] = 0

    for label in label_ids:
        if label > 0:
            for k in ks:
                if label <= k:
                    p[k] += 1

    for k in ks:
        if number_of_samples is None:
            p[k] /= len(label_ids)
        else:
            p[k] /= number_of_samples

    return p

# single-label metrics
# this is the original function used in crossencoder/train_cross.evaluate() 
# to calculate (i) the number of accurately predicted instances in a batch, 
# and to display (ii) the result of each instance
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels
# this is the one that calculate accuracy from (argmaxed) indexes instead of raw logits of the predictions.
def accuracy_from_ind(ind_out, labels):
    #outputs = np.argmax(out, axis=1)
    return np.sum(ind_out == labels), ind_out == labels
# this is the one that calculate accuracy for out-of-KB (NIL) labels - all inputs are np.arrays
def accuracy_from_ind_is_NIL(ind_out, labels, is_NIL_labels):    
    #outputs = np.argmax(out, axis=1)
    accordance = ind_out == labels
    accordance_NIL = np.logical_and(accordance,is_NIL_labels)
    if not is_NIL_labels is None:
        num_tp_NIL = np.sum(accordance_NIL)
    else:
        num_tp_NIL = 0    
    return num_tp_NIL, accordance_NIL
# this is the one that calculate accuracy for in-KB labels - all inputs are np.arrays
def accuracy_from_ind_is_in_KB(ind_out, labels, is_NIL_labels):
    #outputs = np.argmax(out, axis=1)
    accordance = ind_out == labels
    accordance_in_KB = np.logical_and(accordance,np.logical_not(is_NIL_labels))
    if not is_NIL_labels is None:
        num_tp_in_KB = np.sum(accordance_in_KB)
    else:
        num_tp_in_KB = 0 
    return num_tp_in_KB, accordance_in_KB

# calculate f1 score, only when prec and rec are both valid (not -1, checked by whether no less than 0), otherwise return -1
def f1_valid(prec,rec):
    if prec >= 0 and rec >= 0:
        return 2*prec*rec/(prec+rec)
    else:
        return -1

# multi-label metrics

def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = "".join(key.split(".module"))
        new_state_dict[name] = value
    return new_state_dict


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('Blink')
    logger.setLevel(10)
    return logger


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


def get_reranker(parameters):
    return BertReranker(parameters)


def get_biencoder(parameters):
    return BiEncoderRanker(parameters)
