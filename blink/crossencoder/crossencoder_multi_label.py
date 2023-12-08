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

from collections import OrderedDict
from tqdm import tqdm
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

# from pytorch_transformers.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )

# from pytorch_transformers.modeling_roberta import (
#     RobertaConfig,
#     RobertaModel,
# )

#from pytorch_transformers.tokenization_bert import BertTokenizer
#from pytorch_transformers.tokenization_roberta import RobertaTokenizer

from transformers import AutoModel,AutoTokenizer

from blink.common.ranker_base import BertEncoderWithFeatures, get_model_obj #BertEncoder
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG, ENT_SYN_TAG, ENT_NIL_TAG


def load_crossencoder(params):
    # Init model
    crossencoder = CrossEncoderRanker(params)
    return crossencoder


class CrossEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer,extra_features=None):
        super(CrossEncoderModule, self).__init__()
        model_path = params["bert_model"]
        if params.get("roberta"):
            #encoder_model = RobertaModel.from_pretrained(model_path)
            encoder_model = AutoModel.from_pretrained(model_path)
        else:
            #encoder_model = BertModel.from_pretrained(model_path)
            encoder_model = AutoModel.from_pretrained(model_path)
        encoder_model.resize_token_embeddings(len(tokenizer))
        self.encoder = BertEncoderWithFeatures(
            encoder_model,
            params["out_dim"], # this is 1 but could use other values (if we modify the loss)
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            extra_features=extra_features, # cross-encoder-level extra features are initiated here - as we need the size of features to initialise the linear layer in the model (this corresponds to the idea in KG-ZESHEL by Ristoski et al, 2021, but not used for NIL detection here).
        )
        self.config = self.encoder.bert_model.config

    def forward(
        self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
    ):
        embedding_ctxt = self.encoder(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        return embedding_ctxt.squeeze(-1) # squezzed

# here is the key class in the crossencoder for inference/ranking
class CrossEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None, extra_features=None): # extra features regarding entities and mentions initiated here - TODO: as params
        super(CrossEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        if params.get("roberta"):
            #self.tokenizer = RobertaTokenizer.from_pretrained(params["bert_model"],)
            self.tokenizer = AutoTokenizer.from_pretrained(    
                params["bert_model"], do_lower_case=params["lowercase"]
            )

        else:
            #self.tokenizer = BertTokenizer.from_pretrained(
            self.tokenizer = AutoTokenizer.from_pretrained(    
                params["bert_model"], do_lower_case=params["lowercase"]
            )
        
        # special_tokens_dict = {
        #     "additional_special_tokens": [
        #         ENT_START_TAG,
        #         ENT_END_TAG,
        #         ENT_TITLE_TAG,
        #         ENT_SYN_TAG,
        #         ENT_NIL_TAG,
        #     ],
        # }
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        self.NULL_IDX = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        
        self.extra_features = extra_features

        # init model
        self.build_model()
        if params["path_to_model"] is not None:
            self.load_model(params["path_to_model"],cpu=not torch.cuda.is_available())

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            #state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def save(self, output_dir):
        self.save_model(output_dir)
        self.tokenizer.save_vocabulary(output_dir)

    def build_model(self):
        self.model = CrossEncoderModule(
            self.params, 
            self.tokenizer, 
            extra_features=self.extra_features, # also feed the extra features when building the model - (Note: this self.extra_features is not used, different from the extra features in self.score_NILs() for NIL)
        ) 
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        #print('type_optimization:',self.params["type_optimization"])
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def score_candidate(self, text_vecs, context_len):
        # Encode contexts first
        #print('text_vecs:',text_vecs.size()) # text_vecs: torch.Size([1, 100, 159])
        num_cand = text_vecs.size(1)
        text_vecs = text_vecs.view(-1, text_vecs.size(-1))
        #print('text_vecs:',text_vecs.size())#,text_vecs) # text_vecs: torch.Size([100, 159])
        #print('context_len:',context_len)
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX, segment_pos=context_len,
        )

        # here it does all the bert process + linear layer (and also concatenating with extra features if chosen too)
        #print('token_idx_ctxt:',token_idx_ctxt.size())
        #print('segment_idx_ctxt:',segment_idx_ctxt.size())
        #print('mask_ctxt:',mask_ctxt.size())
        embedding_ctxt = self.model(token_idx_ctxt, segment_idx_ctxt, mask_ctxt,)
        #print('embedding_ctxt:',embedding_ctxt.size()) # embedding_ctxt: torch.Size([100])
        return embedding_ctxt.view(-1, num_cand)

    #a min, max, and ave pooling of the men-ent scores
    def min_max_ave_pooling(self,scores_all_ents):
        scores_mean = torch.mean(scores_all_ents,dim=1).view(1,-1)
        scores_max = torch.max(scores_all_ents,dim=1).values.view(1,-1)
        scores_min = torch.min(scores_all_ents,dim=1).values.view(1,-1)
        scores_all_ents = scores_all_ents.view(1,-1)
        print('scores_all_ents:',scores_all_ents.size())
        print('scores_mean:',scores_mean.size())
        print('scores_max:',scores_max.size())
        print('scores_all_ents:',scores_min.size())
        scores_all_ents_w_pooling = torch.cat((scores_all_ents,scores_mean,scores_max,scores_min),dim=1)
        #scores_all_ents_w_pooling = scores_all_ents_w_pooling.view(-1,1)
        return scores_all_ents_w_pooling

    #get mention only score from the mention embedding    
    def get_score_from_mention_emb(self, text_vecs, context_len):
        #num_cand = text_vecs.size(1)
        text_vecs = text_vecs.view(-1, text_vecs.size(-1))
        mention_text_vecs = text_vecs[:1,:context_len] # only get the first row element as all rows are the same for the same mention
        #print('mention_text_vecs:',mention_text_vecs.size())
        mention_token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            mention_text_vecs, self.NULL_IDX, segment_pos=-1,
        ) # no segmentation needed here, same as biencoder.to_bert_input(token_idx, null_idx)
        mention_embedding_ctxt = self.model(mention_token_idx_ctxt, segment_idx_ctxt, mask_ctxt,) # same model as the cross-encoder
        #print('mention_embedding_ctxt:',mention_embedding_ctxt)
        return mention_embedding_ctxt.view(-1, 1) # or param["train_batch_size"]

    # get scores for NIL-classification (with linear layer on top of prediction scores)       
    def score_NILs(self,scores_all_ents,use_score_features=True,use_extra_features=False,use_men_only_score_ft=False,mention_matchable_fts=None,men_score=None):
        assert use_men_only_score_ft or use_score_features or (use_extra_features and (not mention_matchable_fts is None)) # should use at least one type of features and the extra features should not be None if using them

        NIL_features = torch.FloatTensor(1,0).to(self.device)
        #update scores by mention-only score if chosen to
        if use_men_only_score_ft and (not men_score is None):
            #scores_all_ents = torch.cat((men_score,scores_all_ents),dim=1)
            NIL_features = torch.cat((NIL_features,men_score),dim=1)
            #print('scores_all_ents:',scores_all_ents,scores_all_ents.size())

        # if use_score_features:
        #     if use_extra_features and (not mention_matchable_fts is None):
        #         NIL_features = torch.cat((scores_all_ents,mention_matchable_fts),dim=1)
        #     else:
        #         NIL_features = scores_all_ents
        # else:
        #     if use_extra_features and (not mention_matchable_fts is None):
        #         NIL_features = mention_matchable_fts
        if use_score_features:
            NIL_features = torch.cat((NIL_features,scores_all_ents),dim=1)

        if use_extra_features and (not mention_matchable_fts is None):
            NIL_features = torch.cat((NIL_features,mention_matchable_fts),dim=1)

        #print('NIL_features:',NIL_features.size())
        ft_size = NIL_features.size(1)
        linear_NIL = nn.Linear(ft_size,2).to(self.device)
        scores_NIL = linear_NIL(NIL_features) # add more features here

        return scores_NIL

    def forward(self, input_idx, label_input, context_len,inference_only=False, label_is_NIL_input=None): # not calculating loss if inference_only
        #print('input_idx:',input_idx)
        #print('label_input:',label_input)
        scores = self.score_candidate(input_idx, context_len)
        #print("scores:",scores)

        label_is_NIL_input = label_is_NIL_input.long().to(self.device)
        #max, ave, max-ave features of non-NIL scores

        #2 features whether it has exact or fuzzy string matching in KB
        if not inference_only:
            #pscores = F.softmax(scores,dim=-1)
            #scores = torch.log(pscores)
            #loss = F.nll_loss(scores, label_input)
            #loss = F.cross_entropy(scores, label_input, reduction="mean")
            #loss = F.cross_entropy(scores, label_input, reduction="mean")
            loss = F.binary_cross_entropy_with_logits(scores, label_input)
            #print('loss_ori:',loss)
            return loss, scores
        else:
            return scores

def to_bert_input(token_idx, null_idx, segment_pos):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    if segment_pos > 0:
        segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] > 0

    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    # token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
