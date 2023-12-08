# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Provide an argument parser and default command line options for using BLINK.
import argparse
import importlib
import os
import sys
import datetime


ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"
ENT_SYN_TAG = "[unused3]"
ENT_NIL_TAG = "[unused4]" # NIL tag
ENT_PARENT_TAG = "[unused5]"
ENT_CHILD_TAG = "[unused6]"
ENT_TOP_TAG = "[unused7]"
ENT_NULL_TAG = "[unused8]"

class BlinkParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_blink_args:
        (default True) initializes the default arguments for BLINK package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_blink_args=True, add_model_args=False, 
        description='BLINK parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_blink_args,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_blink_args:
            self.add_blink_args()
        if add_model_args:
            self.add_model_args()

    def add_blink_args(self, args=None):
        """
        Add common BLINK args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only --debug_max_lines samples.",
        )
        parser.add_argument(
            "--debug_max_lines",
            default=200,
            type=int,
            help="if --debug, then run only with the first --debug_max_lines samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda", 
            action="store_true", 
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int) 
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )
        parser.add_argument(
            "--zeshel",
            action="store_true",
            #default=True,
            #type=bool,
            help="Whether the dataset is from zeroshot.",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        ) 
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.", # this further loads the model to override the specified --bert_model type below, see BiEncoderRanker.load_model() in biencoder.py
        )
        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            help="Whether to lower case the input text. True for uncased models, False for cased models.", # this is a paramter of BertTokenizer.from_pretrained(), see AutoTokenizer.__init__() in biencoder.py
        )
        parser.add_argument(
            "--context_key", 
            default="context", 
            type=str,
            help="the formatted key value to look at contexts in the data",
        )
        parser.add_argument(
            "--use_context",
            action="store_true", 
            help="whether to use context around the mention, if not choosing, using mention name only")  
        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        ) # only used when --add_linear, this is the output dim of the added linear layer after bert
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additonal linear projection on top of BERT.",
        )
        parser.add_argument(
            "--data_path",
            default="data/zeshel",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )
        parser.add_argument(
            "--use_triplet_loss_bi_enc",
            action="store_true",
            help="Whether to use triplet loss for the bi-encoder",
        )
        parser.add_argument(
            "--use_miner_bi_enc",
            action="store_true",
            help="Whether to use hard positive/negative miner when triplet loss is used for the bi-encoder",
        )
        parser.add_argument(
            "--use_ori_classification",
            action="store_true",
            help="Whether to use the original entity classification in the cross-encoder",
        )
        parser.add_argument(
            "--use_NIL_classification",
            action="store_true",
            help="Whether to use NIL classification in the cross-encoder",
        )
        parser.add_argument(
            "--lambda_NIL",
            default=0.25,
            type=float,
            help="the weight of NIL labels",
        )
        parser.add_argument(
            "--use_score_features",
            action="store_true",
            help="Whether to use mention-entity scores in the cross-encoder for out-of-KB entity detection.",
        )
        parser.add_argument(
            "--use_score_pooling",
            action="store_true",
            help="Whether to use mention-entity score pooling (mean, min, max) in the cross-encoder for out-of-KB entity detection.",
        )
        parser.add_argument(
            "--use_men_only_score_ft",
            action="store_true",
            help="Whether to use mention-only scores with score features in the cross-encoder for out-of-KB entity detection.",
        )
        parser.add_argument(
            "--use_extra_features",
            action="store_true",
            help="Whether to use extra features in the cross-encoder for out-of-KB entity detection. The extra features will be used when the .t7 data contain them, as generated by eval_biencoder.py. In eval_biencoder.py, setting this to True will ensure the string-matching extra features generated",
        )
        parser.add_argument(
            "--use_BM25",
            action="store_true",
            help="Whether to use BM25 instead of bi-encoder",
        )
        parser.add_argument(
            "--aggregating_factor",
            default=20,
            type=int,
            help="number of candidates to be generated by the number of factor's times (i.e. top_k*agg_factor), so that after aggregation there are still top_k candidates",
        )

    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--limit_by_max_lines", action="store_true", help="Whether to limit the training data by a max number of size (see the parameter below)."
        )
        parser.add_argument(
            "--max_number_of_train_size", default=300000, type=int, 
            help="Maximum data size (or number of mentions) for training *for bi-encoder* (this is not available in the cross-encoder)."
        )
        parser.add_argument(
            "--train_batch_size", default=8, type=int, 
            help="Total batch size for training."
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--limit_by_train_steps", action="store_true", help="Whether to limit the training steps by a max number (see the parameter below)."
        )
        parser.add_argument(
            "--max_num_train_steps",
            default=20000,
            type=int,
            help="Maximum number of training steps; this parameter can be set for bi-encoder and cross-encoder.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=10, 
            help="Interval of loss printing",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=100,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=1, 
            help="Interval for model saving"
        )
        parser.add_argument(
            "--save_model_epoch_parts", 
            action="store_true",
            help="Whether to save interval models (or parts) with each epoch in cross-encoder"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", #type=bool, default=False, 
            action="store_true",
            help="Whether to shuffle train data",
        )
        parser.add_argument(
            "--optimize_NIL",
            action="store_true",
            help="Whether to optimize NIL's metric, if not, it will optimize the overall metric (both in-KB and NIL)",
        )
        parser.add_argument(
            "--use_preprocessed_data", 
            action="store_true",
            help="Whether to use preprocessed data",
        )
        parser.add_argument(
            "--use_full_training_data", 
            action="store_true",
            help="Whether to use full training data (i.e. train+valid)",
        )
        parser.add_argument(
            "--fix_seeds",
            action="store_true",
            help="Fixing the seeds for training",
        )
        
    def add_eval_args(self, args=None):
        """
        Add model evaluation args (also for data generation from biencoders - see eval_biencoders.py).
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--eval_batch_size", default=8, type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results. Should be set as true to generate data for cross-encoder training.",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="Batch size for encoding."
        )
        # here is about candidate representation 
        parser.add_argument(
            "--use_synonyms", 
            action="store_true",
            help="Whether to use synonyms for candidate representation",
        )
        parser.add_argument(
            "--use_NIL_tag", 
            action="store_true",
            help="Whether to use NIL tag, an unknown token in word piece tokenizer in BERT, for NIL entity reprentation, instead of using a string of 'NIL' to represent NIL entities",
        )
        parser.add_argument(
            "--use_NIL_desc", 
            action="store_true",
            help="Whether to add NIL desc for NIL entity reprentation, instead of using an empty string",
        )
        parser.add_argument(
            "--use_NIL_desc_tag", 
            action="store_true",
            help="Whether to use special token of NIL in the desc for NIL entity reprentation, instead of using \"NIL\"",
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for cached candidate pool (id tokenization of candidates)",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for cached candidate encoding",
        )
        # infering NILs
        parser.add_argument(
            "--use_NIL_classification_infer",
            action="store_true",
            help="Whether to infer with the multi-task NIL classifier",
        )
        parser.add_argument(
            "--add_NIL_to_bi_enc_pred",
            action="store_true",
            help="Whether to add NIL to the last of the top-k predictions of bi-encoder (this affects eval_biencoder.py to generate data to train cross-encoder)",
        )
        parser.add_argument(
            "--NIL_ent_ind",
            default=88150,
            type=int,
            help="Entity index of NIL in the entity catalogue - default as 88150 for the ShARe/CLEF 2013 dataset"
        )