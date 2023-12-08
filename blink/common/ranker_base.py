# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None,name=''):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None
        self.name=name    

    def forward(self, token_ids, segment_ids, attention_mask):
        #print('BertEncoder is encoding things:', self.name)
        # output_bert, output_pooler = self.bert_model(
        #     token_ids, segment_ids, attention_mask
        # )
        output = self.bert_model(
            input_ids=token_ids, 
            token_type_ids=segment_ids, 
            attention_mask=attention_mask
        )
        output_bert, output_pooler = output.last_hidden_state, output.pooler_output

        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        #print('embeddings:',embeddings.size()) # embeddings: torch.Size([100, 768])
        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings)) # here the additional linear layer is over the pooled vector.
            #print('result after additional_linear:',result.size())
        else:
            result = embeddings # here it uses the vector of the [CLS] token

        return result

class BertEncoderWithFeatures(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None,name='',extra_features=None): # extra_features of size [100,k]
        super(BertEncoderWithFeatures, self).__init__()
        self.layer_pulled = layer_pulled
        self.extra_features = extra_features
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)
        
        if self.extra_features is not None:
            extra_features_dim = extra_features.size(1)
        else:
            extra_features_dim = 0

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim+extra_features_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None
        self.name=name    

    def forward(self, token_ids, segment_ids, attention_mask,):
        #print('BertEncoderWithFeatures is encoding things:', self.name)
        # output_bert, output_pooler = self.bert_model(
        #     token_ids, segment_ids, attention_mask
        # )
        output = self.bert_model(
            input_ids=token_ids, 
            token_type_ids=segment_ids, 
            attention_mask=attention_mask
        )
        output_bert, output_pooler = output.last_hidden_state, output.pooler_output

        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        if self.extra_features is not None:
            embeddings = torch.cat([embeddings,self.extra_features],dim=1)

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings)) # here the additional linear layer is over the pooled vector.
            #print('result after additional_linear:',result.size())
        else:
            result = embeddings # here it uses the vector of the [CLS] token

        return result