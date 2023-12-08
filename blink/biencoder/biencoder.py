# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# from pytorch_transformers.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )

from transformers import AutoModel,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer

# from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer

# for contrastive loss
from pytorch_metric_learning import miners, losses, distances
#torch.cuda.set_device(0)

def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        # ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        # cand_bert = BertModel.from_pretrained(params['bert_model'])
        if not 'LLAMA' in params["bert_model"]:
            ctxt_bert = AutoModel.from_pretrained(params["bert_model"])
            cand_bert = AutoModel.from_pretrained(params['bert_model'])
        else:
            ctxt_bert = LlamaForCausalLM.from_pretrained(params["bert_model"])
            cand_bert = LlamaForCausalLM.from_pretrained(params["bert_model"])

        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            name='context_encoder',
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            name='cand_encoder',
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            #print('token_idx_ctxt:',token_idx_ctxt, '\nsegment_idx_ctxt:',segment_idx_ctxt, '\nmask_ctxt:',mask_ctxt)
            embedding_ctxt = self.context_encoder( # this calls the blink.common.ranker_base.BertEncoder.forward()
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            #print('token_idx_cands:',token_idx_cands, '\nsegment_idx_cands:',segment_idx_cands, '\nmask_cands:',mask_cands)
            embedding_cands = self.cand_encoder( # this calls the blink.common.ranker_base.BertEncoder.forward()
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands

# the biencoder model to be trained
class BiEncoderRanker(torch.nn.Module): 
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__() # call the __init__ of the super method
        self.params = params
        self.device = torch.device(
           "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        print('self.n_gpu:',self.n_gpu)
        print('self.device in BiEncoderRanker:',self.device)
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        # here the tokenizer is set to lowercase the texts.
        print('params[\"lowercase\"]:', params["lowercase"])
        # self.tokenizer = BertTokenizer.from_pretrained(
        if not 'LLAMA' in params["bert_model"]:            
            self.tokenizer = AutoTokenizer.from_pretrained(    
                params["bert_model"], do_lower_case=params["lowercase"]
            )
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(params["bert_model"])   
        print('model tokenizer:',self.tokenizer)
        # init model
        self.build_model()
        print('model built')
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path,cpu=not torch.cuda.is_available())
        print('model loaded')
        print('moving model to',self.device)
        self.model = self.model.to(self.device)
        print('model moved to',self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        
        # init constrastive learning
        
    def load_model(self, fname, cpu=False):
        if cpu:
            #state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False) # load state_dict # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    def build_model(self):
        self.model = BiEncoderModule(self.params) # ranker model is a BiEncoderModule 

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
    
    #def prune_repeated(self,cand_input):

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
        unit_norm=False, # whether to unit-normalise the vecs before dot product
        #th_NIL=0.4,
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        ## Put the contexts tensors into gpu if it is available - not necessary, just always to set the no_cuda as True, the it will use the CPU to do the bi-encoder stage.
        #token_idx_ctxt = token_idx_ctxt.to(self.device)
        #segment_idx_ctxt = segment_idx_ctxt.to(self.device)
        #mask_ctxt = mask_ctxt.to(self.device)

        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        if unit_norm:
            embedding_ctxt = F.normalize(embedding_ctxt)
        #print('embedding_ctxt:',embedding_ctxt)
        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            cand_encs = cand_encs.to(self.device)
            if unit_norm:
                cand_encs = F.normalize(cand_encs)
            #print('cand_encs:',cand_encs)
            scores = embedding_ctxt.mm(cand_encs.t())
            # to look at closely from here - then to set NIL-based threshold - this will be interesting and challenging
            #print('scores:',scores.size()) # scores: torch.Size([8, 5903527])
            return scores #embedding_ctxt.mm(cand_encs.t()).to(self.device)

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if unit_norm:
            embedding_cands = F.normalize(embedding_cands)
        if random_negs: # what's the difference between this and the one below?
            # train on random negatives
            #print('embedding_ctxt:',embedding_ctxt.size()) # torch.Size([32, 768])
            #print('embedding_cands:',embedding_cands.size()) # torch.Size([32, 768])
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores
    
    # prune the scores of size(M,E) to size(M,unique(E))
    def prune_scores(
        self,
        scores,
        cand_input
    ):
        cand_input_unique = cand_input.unique(dim=0)
        cand_input_ind2uni=list(map(lambda k: cand_input.tolist().index(k),cand_input_unique.tolist()))
        scores = scores[:,cand_input_ind2uni]
        return scores

    # reformulate target by treating repeated classes - let them have the same label index as the first one
    # based on https://stackoverflow.com/a/72727869/5319143
    def reform_target(
        self,
        cand_input,
    ):
        # cand_input_as_list = cand_input.tolist()
        # target_as_list = list(map(lambda x: cand_input_as_list.index(x), cand_input_as_list))
        # target = torch.tensor(target_as_list)
        # or another way
        d = {}; target = torch.tensor([d.setdefault(tuple(i.tolist()), e) for e, i in enumerate(cand_input)])
        return target    

    # TODO: do not emb the NIL entities, but embed the NIL mentions
    def emb_for_contras_learn(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            cand_encs = cand_encs.to(self.device)
            #print('cand_encs:',cand_encs)
            #scores = embedding_ctxt.mm(cand_encs.t())
            # to look at closely from here - then to set NIL-based threshold - this will be interesting and challenging
            #print('scores:',scores.size()) # scores: torch.Size([8, 5903527])
            #return scores #embedding_ctxt.mm(cand_encs.t()).to(self.device)
            emb_all = torch.cat([embedding_ctxt,cand_encs],dim=0)
            return emb_all

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        assert random_negs == True
        if random_negs: # what's the difference between this and the one below?
            # train on random negatives
            #print('embedding_ctxt:',embedding_ctxt.size()) # torch.Size([32, 768])
            #print('embedding_cands:',embedding_cands.size()) # torch.Size([32, 768])
            emb_all = torch.cat([embedding_ctxt,embedding_cands],dim=0)
            return emb_all
        # else:
        #     # train on hard negatives
        #     embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
        #     embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
        #     scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
        #     scores = torch.squeeze(scores)
            # return scores
    
    # def inference_out_of_KB(self, scores, th_NIL):
    #     p_scores = F.softmax(scores)
    #     p_scores > th_NIL
        
    # here to look at further, 26 Apr 2022 - also again 21 May 2022
    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None,tensor_is_label_NIL=None,th_NIL=0.4,lambda_NIL=1, use_triplet_loss=False,use_miner=False):
        flag = label_input is None        
        scores = self.score_candidate(context_input, cand_input, random_negs=flag, unit_norm=False)
        # # prune the scores
        # cand_input_unique = cand_input.unique(dim=0)
        # cand_input_ind2uni=list(map(lambda k: cand_input.tolist().index(k),cand_input_unique.tolist()))
        # scores = scores[:,cand_input_ind2uni]
        # scores = self.prune_scores(scores,cand_input)
        # print(scores.size())
        
        #torch.set_printoptions(edgeitems=5)
        #print('cand_input:',cand_input.size(),type(cand_input),cand_input)
        
        #print('scores:',scores.size(),type(scores),scores[0]) 
        # torch.Size([32, 32])
        bs = scores.size(0)
        #num_unique_cand = scores.size(1)

        if label_input is None: # this is used for training
            target = torch.LongTensor(torch.arange(bs))
            #print('target-ori:',target.size(),type(target),target) #target: torch.Size([16]) <class 'torch.
            #why using torch.arange? - this creates a 1D tensor of [0,1,2,...,bs-1] so that each input has target to its index, thus using the others as in-batch negatives. - so this is a |bs|-class classification.
            #address issue that the in-batch can have same labels - need to reflect this in the target - but this does not work on CE loss.
            #target = self.reform_target(cand_input)
            #cand_input_as_list = cand_input.tolist()
            #target_as_list = list(map(lambda x: cand_input_as_list.index(x), cand_input_as_list))
            #target = torch.tensor(target_as_list)
            #target = torch.LongTensor(target)
            #print('target-new:',target.size(),type(target),target) #target: torch.Size([16]) <class 'torch.Tensor'> tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
            target = target.to(self.device)

            #print('scores before softmax:',scores.size(),type(scores),scores[0]) 
            #scores = F.softmax(scores,dim=-1)
            #scores = torch.log(scores)
            #scores = F.log_softmax(scores, dim=-1) # https://pytorch.org/docs/1.6.0/nn.functional.html?highlight=functional%20log_softmax#torch.nn.functional.log_softmax # dim (int) – A dimension along which log_softmax will be computed.
            #print('scores after softmax:',scores.size(),type(scores),scores[0]) 
            
            # the NIL entity definition may interfere the training - but where to mask the NILs? and how? - to do tmr
            # # set s(m,e_NIL) as 0 in scores, this includes s(m_NIL,e_NIL) and s(m_in_KB,e_NIL)
            # print('tensor_is_label_NIL:',tensor_is_label_NIL.size(),type(tensor_is_label_NIL),tensor_is_label_NIL)
            # tensor_is_label_in_KB = 1 - tensor_is_label_NIL.long()
            # tensor_is_label_in_KB = tensor_is_label_in_KB.t().repeat([bs,1])
            # scores = torch.mul(scores,tensor_is_label_in_KB)

            if not use_triplet_loss:
                # contrastive loss - may need to add temperature, so far r as 1.
                #loss = F.nll_loss(scores, target)
                loss = F.cross_entropy(scores, target, reduction="mean")
                # torch.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
                # https://pytorch.org/docs/1.6.0/nn.functional.html#torch.nn.functional.cross_entropy
            
            # to give a further thought about the loss
            # to add an out-of-KB regularisation (there are several versions - norm and hinge)

            # in/out-KB classification loss - needs to be revised , if this is used, the corresponding inference needs to be adapted too. (or here might need a separate classifier)
            # target_NIL = torch.squeeze(tensor_is_label_NIL.long())
            # #print('target_NIL:',target_NIL)
            # loss_NIL = F.cross_entropy(scores,target_NIL)
            # loss = loss + lambda_NIL*loss_NIL
            
            # # metric learning loss - this can be done, but need to revise from score_candidate(). - done 
            # #contras_loss_def = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5)
            # #contras_loss = contras_loss_def(scores, target)
            # #loss = loss + contras_loss
            if use_triplet_loss:
                #scores.detach() # will this affect the results? TODO # Tensor.detach(): It returns a new tensor without requires_grad = True. The gradient with respect to this tensor will no longer be computed.
                #print('using triplet loss')
                emb_all = self.emb_for_contras_learn(context_input, cand_input, random_negs=flag) # comment out this line with train with the CE loss, otherwise it cannot optimise.
                #print('emb_all:',emb_all.size()) # batch_size * (2*hidden_size), e.g. 16*2048

                # contras_loss_def = losses.TripletMarginLoss(margin=0.2,distance=distances.CosineSimilarity()) # as in COMET paper
                contras_loss_def = losses.TripletMarginLoss(margin=0.2,distance=distances.DotProductSimilarity()) # using the same ranking metric for candidate generation
                target = torch.cat([target,target],dim=0)
                if use_miner:
                    print('use miner with triplet loss')
                    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
                    hard_pairs = miner(emb_all, target)
                    loss = contras_loss_def(emb_all, target, hard_pairs)
                else:
                    loss = contras_loss_def(emb_all, target)
            #print('loss:',loss)
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            #https://pytorch.org/docs/1.6.0/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bce#torch.nn.BCEWithLogitsLoss
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores

def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
