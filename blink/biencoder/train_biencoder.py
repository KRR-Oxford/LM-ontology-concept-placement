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

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

#from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
#from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None

# The evaluation function for all data (not just in-batch)
# TODO
def evaluate_all_data(
    reranker, eval_dataloader, params, device, logger,
):
    pass    

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
# TODO: try to output the erroneous data?
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        #context_input, candidate_input, _, _, _ = batch # this is probably for zero shot el - to be checked later
        _, context_input, _, candidate_input, _, tensor_is_label_NIL, _ = batch #see data_process.process_mention_for_insertion_data(): tensor_data = TensorDataset(mention_vecs, context_vecs, contextual_vecs, label_idx, is_label_NIL, concept_id_ori_vecs)    
        if step < 3:  
            print('context_input:',context_input,type(context_input))  
            print('candidate_input:',candidate_input,type(context_input))
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input,tensor_is_label_NIL=tensor_is_label_NIL,use_triplet_loss=params['use_triplet_loss_bi_enc'],use_miner=params['use_miner_bi_enc']) # this logit is the dot product (``scores'').

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids were diagonal, but we reform the target so that the repeated labels have the same index
        # label_ids_diagnl = torch.LongTensor(
        #        torch.arange(params["eval_batch_size"])               
        # ).numpy()
        label_ids_repeat = reranker.reform_target(candidate_input).numpy()
        # print('label_ids_diagnl:',label_ids_diagnl)
        #print('label_ids_repeat:',label_ids_repeat)
        #tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids_diagnl)
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids_repeat)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f (%d/%d)" % (normalized_eval_accuracy,eval_accuracy,nb_eval_examples))
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    print('type_optimization:',params["type_optimization"]) # it seems that all encoder layers are used in bi-encoder       
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
    )# source code here https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    ''' Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    ''' # so epochs have an effect on warmup_steps and therefore WarmupLinearSchedule. The less the epoches, the less the warmup steps, then the quickier it learns.
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    if params["fix_seeds"]:
        # Fix the random seeds (part 1) (nice trick!)
        seed = params["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu
    print('n_gpu:',n_gpu)
    if params["fix_seeds"]:
        # Fix the random seeds (part 2)
        if reranker.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    #args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Load train data
    train_samples = utils.read_dataset("train%s%s" % ('_full' if params["use_full_training_data"] else '', '_preprocessed' if params["use_preprocessed_data"] else ''), params["data_path"], limit_by_max_lines=params["limit_by_max_lines"], max_lines=params["max_number_of_train_size"], debug_max_lines=params["debug_max_lines"]) ## using train-all (the concatenation of train, valid, test-in-KB)
    logger.info("Read %d train samples." % len(train_samples))

    train_data, train_tensor_data = data.process_mention_for_insertion_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        #context_key=params["context_key"],
        use_context=params["use_context"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        debug_max_lines=params["debug_max_lines"],
        NIL_ent_id=params["NIL_ent_ind"],
        # use_NIL_tag=params["use_NIL_tag"],
        # use_NIL_desc=params["use_NIL_desc"],
        # use_NIL_desc_tag=params["use_NIL_desc_tag"],
        # use_synonyms=params["use_synonyms"],
        #for_inference=False,
    )
    if params["shuffle"]:
        print('random shuffling')
        train_sampler = RandomSampler(train_tensor_data)
    else:
        print('without shuffling')
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("valid%s" % ('_preprocessed' if params["use_preprocessed_data"] else ''), params["data_path"]) ## here using valid-NIL.json instead of valid.json (in-KB).
    logger.info("Read %d valid samples." % len(valid_samples))

    valid_data, valid_tensor_data = data.process_mention_for_insertion_data(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        #context_key=params["context_key"],
        use_context=params["use_context"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        debug_max_lines=params["debug_max_lines"],
        NIL_ent_id=params["NIL_ent_ind"],
        # use_NIL_tag=params["use_NIL_tag"],
        # use_NIL_desc=params["use_NIL_desc"],
        # use_NIL_desc_tag=params["use_NIL_desc_tag"],
        # use_synonyms=params["use_synonyms"],
        #for_inference=True,
    )
    valid_sampler = SequentialSampler(valid_tensor_data) # no shuffling for validation data
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    results = evaluate(
        reranker, valid_dataloader, params, device=device, logger=logger,
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

    model.train() # setting it to training mode

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"): # trange(N): a convenient shortcut for tqdm(xrange(N)).
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
        for step, batch in enumerate(iter_):
            if params["limit_by_train_steps"] and step == params["max_num_train_steps"]:
                break
            batch = tuple(t.to(device) for t in batch)
            #context_input, candidate_input, _, _, _ = batch # this is probably for zero shot el - to be checked later
            _, context_input, _, candidate_input, _, tensor_is_label_NIL, _ = batch
            #in_batch_target = torch#generate target
            loss, _ = reranker(context_input, candidate_input,tensor_is_label_NIL=tensor_is_label_NIL,use_triplet_loss=params['use_triplet_loss_bi_enc'],use_miner=params['use_miner_bi_enc']) # here it gets the loss
            #print('loss:',loss,type(loss))
            #print('loss.item():',loss.item(),type(loss.item()))
            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item() # Returns the value of this tensor as a standard Python number. This only works for tensors with one element.

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
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                # we do not save the models here (unlike in cross-encoder)
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
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
    logger.info("Best performance in epoch {}: {}".format(best_epoch_idx, best_score))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_blink_args()
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
