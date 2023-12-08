# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import prettytable

import blink.main_dense_plus as main_dense_plus
import blink.candidate_ranking.utils as utils

import ast

# get running input params (mostly related to NIL entity discovery)
parser = argparse.ArgumentParser(description="setting thresholds to detect out-of-KB or NIL entities from texts during entity linking")

parser.add_argument('--data', type=str,
                    help="name of the dataset, which is also used in the model names: share_clef or mm",
                    default='share_clef')
parser.add_argument('--onto_name', type=str,
                    help="main name of the ontology, UMLS for share_clef and mm; and WikiData for NILK", 
                    default='UMLS')
parser.add_argument('--onto_ver', type=str,
                    help="UMLS version. For share/clef: 2012AB. For mm, pruned or prev: using 2017AA_pruned0.1, 2017AA_pruned0.2, or 2014AB, 2015AB for \'full\' data_setting; 2015AB/active, 2017AA/active for st21pv", 
                    default='2012AB')
parser.add_argument('--snomed_subset', type=str,
                    help="SNOMED-CT subset mark: Disease, CPP, etc.", default='Disease')
parser.add_argument('--ontology_fn', type=str,
                    help="path to the ontology .owl file",
                    default='')
parser.add_argument('--iri_prefix', type=str,
                    help="http prefix of the iri",
                    default='http://snomed.info/id/')
parser.add_argument('--onto_complex_concept_setting', type=str,
                    help="empty string for DO, and -all-complexEdge for SNOMED-CT", default='-all-complexEdge')
parser.add_argument('--lowercase',action="store_true",
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument('--biencoder_bert_model', type=str,
                    help="the type of the bert model, as specified in the huggingface model hub.",
                    default='bert_large_uncased')
parser.add_argument('--biencoder_model_name', type=str,
                    help="bi-encoder model name",
                    default='share-clef-syn-tl-NIL-tag')
parser.add_argument('--biencoder_model_size',type=str,
                    help="bi-encoder model size, base or large",
                    default='large')
parser.add_argument('--use_context',
                    help="whether using context for mention representation", action='store_true') 
parser.add_argument('--max_cand_length',type=str,
                    help="max candidate length used in training and testing",
                    default='128')
parser.add_argument('--eval_batch_size',type=str,
                    help="evaluation batch size for bi-encoder and cross-encoder",
                    default='32')
parser.add_argument('--use_synonyms',
                    help="whether using synonyms for candidate representation", action='store_true')  
parser.add_argument('--NIL_enc_mark', type=str,
                    help="a part of the entity encoding file name",
                    default='_w_NIL_syn_tag')                    
parser.add_argument('-top_k','--top_k', type=int,
                    help="top_k for candidate generation", default=100)
parser.add_argument('--top_k_cand_seed', type=int, 
                    help='number of seeded candidates (before any candidate enrichment steps) for each mention',
                    default="10")
parser.add_argument('--edge_cand_enrich',action="store_true", 
                    help='whether to enrich the edge candidates.')
parser.add_argument("--edge_ranking_by_score",action="store_true",
                    help="Whether to rank edges by score")
parser.add_argument("--use_leaf_edge_score",action="store_true",
                    help="Whether to use the LEAF_EDGE_SCORE below")
parser.add_argument("--LEAF_EDGE_SCORE",type=int,
                    help="A pre-defined score for leaf edges, i.e., <parent, NULL>, for the edge ranking: For SNOMED-CT, this is suggested to be a very high score, e.g. 1000; For DO, this is suggested to be a very low score, e.g. -1000", default=1000)    
parser.add_argument('--aggregating_factor', type=int,
                    help="aggregating factor for top-k*prediction", default=20)
parser.add_argument("--crossencoder_bert_model",type=str,
                    help="the type of the bert model, as specified in the huggingface model hub.",
                    default='bert_base_uncased')
parser.add_argument('--cross_model_setting',type=str,
                    help="cross-encoder model name",
                    default='original')
parser.add_argument('--cross_model_size',type=str,
                    help="cross-encoder model size, base or large",
                    default='base')
parser.add_argument('--set_NIL_as_cand',
                    help="this allows the model to include NIL as the last top-k candidate if it is not predicted by the bi-encoder", action='store_true')   
parser.add_argument('--use_NIL_tag',
                    help="whether using the NIL special token to represent a NIL entity", action='store_true')  
parser.add_argument('--use_NIL_desc',
                    help="whether adding NIL desc to represent a NIL entity", action='store_true')  
parser.add_argument('--use_NIL_desc_tag',
                    help="whether using special token of NIL in the desc to represent a NIL entity", action='store_true')  
parser.add_argument('--with_NIL_infer',
                    help="this allows the model to infer NIL entities based on the prediction scores/logits in the bi-encoder and the cross-encoder", action='store_true')   
# the two arguments work only when args.with_NIL_infer is set True.
parser.add_argument('-th1','--th_NIL_bi_enc', type=str,
                    help="threshold for bi-encoder", default=0)
parser.add_argument('-th2','--th_NIL_cross_enc', type=str,
                    help="threshold for cross-encoder", default=0)   
parser.add_argument('-BM25','--use_BM25',
                    help="using BM25 as the candidate generator instead of the bi-encoder", 
                    action='store_true')     
parser.add_argument('--save_cand', 
                    help="whether to save candidates",            
                    action='store_true')                       
parser.add_argument('--cand_only', 
                    help="only for candidate generation",            
                    action='store_true')
parser.add_argument('-m','--marking', type=str,
                    help="string to mark the output file name",
                    default='')
parser.add_argument('--debug', 
                    help="debugging with sampled test set",            
                    action='store_true')
parser.add_argument('--no_cuda', 
                    help="whether or not *not* using gpu",            
                    action='store_true')
parser.add_argument("--NIL_concept", type=str, 
                    help="NIL concept ID",
                    default="SCTID-less")
parser.add_argument('--measure_wp', 
                    help="whether or not to calculte wu & palmer similarity",            
                    action='store_true')

                            
args_keyed_in = parser.parse_args()

data = args_keyed_in.data

if data[:3] == 'mm+':
    #use_NIL_tag = False
    #if not args_keyed_in.use_NIL_tag:
    #bienc_model_name = data.replace('_','-') + '-22Jul'
    bienc_model_name = args_keyed_in.biencoder_model_name
    biencoder_model_size = args_keyed_in.biencoder_model_size
    NIL_enc_mark = args_keyed_in.NIL_enc_mark
    #NIL_enc_mark = '_w_NIL'
    #cross_model_setting = 'multi-task-score-only+pool+men'#'original-w-o-NIL' #'multi-task-score-only+pool+men'#'original-w-o-NIL' #'original-BM25-top100'#'original-cand-len64' #'score+extra-NIL-infer'#'multi-task-score+extra-NIL-infer'#'multi-task-score+extra'#'multi-task-score-only' #'original' #'original-cand-len32' #'original'
    cross_model_setting = args_keyed_in.cross_model_setting
#else:
    #    bienc_model_name = data.replace('_','-') + '-13-Sep-NIL-tag'
    #    NIL_enc_mark = '_w_NIL_tag'
    #    cross_model_setting = 'original-NIL-tag'
    #with_NIL_ent_id = not args_keyed_in.use_BM25 #BM25 does not match to 'NIL'
    with_NIL_ent_enc = True
    cross_model_name = data + '/' + cross_model_setting
    crossencoder_model_size = args_keyed_in.cross_model_size
    onto_name = args_keyed_in.onto_name
    onto_ver = args_keyed_in.onto_ver #'2017AA_pruned0.1'
    onto_ver_prefix = onto_ver.split('_')[0]    
    entity_catalogue_postfix = '-'.join(NIL_enc_mark.split('-')[:1]).replace('_','-') # "-edges-all"
    print("entity_catalogue_postfix:",entity_catalogue_postfix)
    if args_keyed_in.debug:
        DATASETS = [
            # {
            #     "name": "mm-%s-dev" % onto_ver, #TODO test data from another onto_ver
            #     "filename": "data/MedMentions-preprocessed/full-%s/valid.jsonl" % onto_ver, 
            # },
            {
                "name": "mm-valid-in-KB-sample",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/valid-sample.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), # the first 1000 rows of the data, e.g. created by using "head -1000 data/../valid.jsonl data/../valid-sample.jsonl"
            },            
            {
                "name": "mm-valid-NIL",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/valid-NIL.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            },
            {
                "name": "mm-test-NIL",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-NIL.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            },
            # {
            #     "name": "mm-test-NIL-all",
            #     "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-NIL-all.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            # },
            # {
            #     "name": "mm-test-NIL-complex",
            #     "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-NIL-complex.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            # },            
        ]    

    else:    
        DATASETS = [
            {
                "name": "mm-valid-in-KB",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/valid.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            },  
            {
                "name": "mm-test-in-KB",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-in-KB.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            },
            {
                "name": "mm-valid-NIL",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/valid-NIL.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            },
            {
                "name": "mm-test-NIL",
                "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-NIL.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            },
            # {
            #     "name": "mm-test-NIL-all",
            #     "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-NIL-all.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            # },
            # {
            #     "name": "mm-test-NIL-all-complex",
            #     "filename": "data/MedMentions-preprocessed+/%s/st21pv_syn_attr%s-edges-final/test-NIL-all-complex.jsonl" % (args_keyed_in.snomed_subset,args_keyed_in.onto_complex_concept_setting), 
            # },
        ]   

#the key parameters here
PARAMETERS = {
    "faiss_index": None,
    "index_path": None,
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    #"biencoder_bert_model": bi_enc_bert_model,
    #"biencoder_model": "models/biencoder_wiki_large.bin", # this one needs to be re-trained
    #"biencoder_config": "models/biencoder_wiki_large.json",
    "biencoder_model": "models/biencoder/%s/pytorch_model.bin" % bienc_model_name, # re-trained, with "NIL" rep
    #"biencoder_model": "models/biencoder/share-clef-tl-prep/pytorch_model.bin",
    #"biencoder_model": "models/biencoder/epoch_2/pytorch_model.bin", # re-trained
    "biencoder_config": "models/biencoder_custom_%s.json" % biencoder_model_size,
    #"biencoder_config": "models/biencoder_custom_base.json",
    "entity_catalogue": "ontologies/%s%s%s.jsonl" % (onto_name,onto_ver,entity_catalogue_postfix), # a four-element entity data structure: text (or definition), idx (or url), title (or name of the entity), entity (canonical name)
    #*here we use the version with_NIL*, which added a general NIL or out-of-KB ("CUI-less") entity in the entity catagolue. The original version is UMLS2012AB.jsonl.
    #"entity_ids": "preprocessing/saved_cand_ids_umls%s_w_NIL_re_tr.pt" % onto_ver,
    "entity_ids": "preprocessing/saved_cand_ids_%s%s%s_re_tr.pt" % (onto_name.lower(),onto_ver, NIL_enc_mark),
    #"entity_encoding": "models/UMLS2012AB_ent_enc/UMLS2012AB_ent_enc.t7", # a torch7 file # get this using script/generate_cand_ids.py and script/generate_candidates_blink.py, derived/pre-computed from UMLS2012AB.jsonl (the one *without* the NIL entity) and the bi-encoder model. # this one needs to be re-generated with new biencoder model
    #"entity_encoding": "models/UMLS%s_ent_enc_re_tr/UMLS%s_ent_enc_re_tr.t7" % (onto_ver,onto_ver), #re-trained
    "entity_encoding": "models/%s%s_ent_enc_re_tr/%s%s%s_ent_enc_re_tr.t7" % (onto_name,onto_ver_prefix,onto_name,onto_ver, NIL_enc_mark if with_NIL_ent_enc else ''), # re-trained, with NIL
    #"crossencoder_bert_model": cross_enc_bert_model,
    #"crossencoder_model": "models/crossencoder_wiki_large.bin", # this one needs to be re-trained
    #"crossencoder_config": "models/crossencoder_wiki_large.json",
    #"crossencoder_model": "models/crossencoder/pytorch_model.bin", # re-trained
    #"crossencoder_model": "models/crossencoder/share_clef/original/pytorch_model.bin", # re-trained with "NIL" rep
    "crossencoder_model": "models/crossencoder/%s/pytorch_model.bin" % cross_model_name, # re-trained w/o "NIL" rep
    #"crossencoder_config": "models/crossencoder_%s_%s.json" % ('custom' if data[:2] != 'mm' else 'medmention',crossencoder_model_size), # use bert-base for cross-encoder 
    "crossencoder_config": "models/crossencoder_custom_%s.json" % crossencoder_model_size, # use bert-base for cross-encoder 
    #"crossencoder_config": "models/crossencoder_custom_large.json",
    "output_path": "output",
    "fast": False,
    #"top_k": 10,
    "candidate_path": "models/candidates",
    "prediction_path": "models/crossencoder/%s" % cross_model_name,
    #"with_NIL_infer": True, # this allows the model to infer NIL entities based on the prediction scores/logits in the bi-encoder and the cross-encoder.
    #"th_NIL_bi_enc": "0.1",
    #"th_NIL_cross_enc": "0.4",
}

# # get NIL-related bi-encoder training params: to get the params below and save them into PARAMETERS
# '''
# use_NIL_tag
# '''
# bi_encoder_model_folder_path = '/'.join(PARAMETERS['biencoder_model'].split('/')[:-1]) # get the folder path from the model path
# bi_encoder_model_params_fn = bi_encoder_model_folder_path + '/training_params.txt'
# with open(bi_encoder_model_params_fn,encoding='utf-8') as f_content:
#     params_dict_str = f_content.readlines()
# assert len(params_dict_str) == 1 # there should be only one row (as a string form dict) in training_params.txt
# print('params_dict_str[0]:',params_dict_str[0])
# params_bi_enc_training = ast.literal_eval(params_dict_str[0])
# list_of_NIL_related_params = ['use_NIL_tag']
# for NIL_related_param_name in list_of_NIL_related_params:                  
#     #assert NIL_related_param_name in params_bi_enc_training # the list of params should be in the cross-encoder training parameters
#     if NIL_related_param_name in params_bi_enc_training:
#         PARAMETERS[NIL_related_param_name] = params_bi_enc_training[NIL_related_param_name]
#     else:
#         PARAMETERS[NIL_related_param_name] = False
#         print(NIL_related_param_name, 'not in params_bi_enc_training')    
# print('PARAMETERS[\'use_NIL_tag\']:', 'use_NIL_tag' in PARAMETERS)

# get NIL-related cross-encoder training params: to get the params below and save them into PARAMETERS
'''
use_ori_classification,
use_NIL_classification, 
use_NIL_classification_infer,
lambda_NIL,
use_score_features,
use_score_pooling
use_men_only_score_ft,
use_extra_features,
'''
list_crossencoder_model_path_split = PARAMETERS['crossencoder_model'].split('/')
if list_crossencoder_model_path_split[-2].startswith("epoch_"):
    # get the folder path from the model path (now does not include the epoch sub-folder, thus -2)
    cross_encoder_model_folder_path = '/'.join(list_crossencoder_model_path_split[:-2])     
else:
    # get the folder path from the model path
    cross_encoder_model_folder_path = '/'.join(list_crossencoder_model_path_split[:-1]) 
cross_encoder_model_params_fn = cross_encoder_model_folder_path + '/training_params.txt'
with open(cross_encoder_model_params_fn,encoding='utf-8') as f_content:
    params_dict_str = f_content.readlines()
assert len(params_dict_str) == 1 # there should be only one row (as a string form dict) in training_params.txt
print('params_dict_str[0]:',params_dict_str[0])
params_cross_enc_training = ast.literal_eval(params_dict_str[0])
list_of_NIL_related_params = ['use_ori_classification',
                              'use_NIL_classification',
                              'use_NIL_classification_infer',
                              'lambda_NIL',
                              'use_score_features',
                              'use_score_pooling',
                              'use_men_only_score_ft',
                              'use_extra_features']
for NIL_related_param_name in list_of_NIL_related_params:                  
    #assert NIL_related_param_name in params_cross_enc_training # the list of params should be in the cross-encoder training parameters
    if NIL_related_param_name in params_cross_enc_training:
        PARAMETERS[NIL_related_param_name] = params_cross_enc_training[NIL_related_param_name]
    else:
        PARAMETERS[NIL_related_param_name] = False
        print(NIL_related_param_name, 'not in params_cross_enc_training')
args = argparse.Namespace(**PARAMETERS,**vars(args_keyed_in)) # combine both key-in parameters with configured parameters
print('args:',args)
logger = utils.get_logger(args.output_path)

models = main_dense_plus.load_models(args, logger) # load biencoder, crossencoder, and candidate entities

table = prettytable.PrettyTable(
    [
        "DATASET",
        "bi-enc acc, micro",
        #"rec at 99", # as the last one is used for NIL, we only compare the first 99.
        "rec at %d, micro" % args.top_k,
        #"bi-enc acc in-KB",
        #"rec at %d in-KB" % args.top_k,
        #"bi-enc acc NIL",
        #"rec at %d NIL" % args.top_k,
        #"cross-enc acc",
        # "acc/rec all",#"overall unorm acc",
        "ins@1, all", # insertion rate at 1, all edges to be satisfied
        "ins@1, any", # insertion rate at 1, any edges to be satisfied
        "ins@5, all",
        "ins@5, any",
        "ins@10, all",
        "ins@10, any",
        "ins, all",
        "ins, any",
        # "prec in-KB",
        # "rec in-KB",
        # "f1 in-KB",
        # "prec NIL",
        # "rec NIL",
        # "f1 NIL",
        # "cross-enc norm acc in-KB",
        # "overall unorm acc in-KB",
        # "cross-enc norm acc NIL",
        # "overall unorm acc NIL",        
        "supp",
        # "supp in-KB",
        # "supp NIL",
    ]
)

for dataset in DATASETS:
    logger.info(dataset["name"])
    PARAMETERS["dataname"] = dataset["name"]
    PARAMETERS["test_mentions"] = dataset["filename"]
    #set the parameter test_mentions as the filename of the dataset

    args = argparse.Namespace(**PARAMETERS,**vars(args_keyed_in))
    
    (
        biencoder_accuracy,
        recall_at,
        #biencoder_in_KB_accuracy,
        #recall_in_KB_at,
        #biencoder_NIL_accuracy,
        #recall_NIL_at,
        #crossencoder_normalized_accuracy,
        #overall_unormalized_accuracy, 
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
        num_datapoints,
        # num_datapoints_in_KB,
        # num_datapoints_NIL,
        predictions,
        scores,
    ) = main_dense_plus.run(args, logger, *models) # here it starts inferencing

    table.add_row(
        [
            dataset["name"],
            round(biencoder_accuracy, 4),
            round(recall_at, 4),
            #round(biencoder_in_KB_accuracy, 4),
            #round(recall_in_KB_at, 4),
            #round(biencoder_NIL_accuracy, 4),
            #round(recall_NIL_at, 4),
            #round(crossencoder_normalized_accuracy, 4),
            #round(overall_unormalized_accuracy, 4),
            round(ins_at_1_all,4),
            round(ins_at_1_any,4),
            round(ins_at_5_all,4),
            round(ins_at_5_any,4),
            round(ins_at_10_all,4),
            round(ins_at_10_any,4),            
            round(ins_all,4),
            round(ins_any,4),
            num_datapoints,
        ]
    )
    # to look at these later
    # print('predictions:',predictions)
    # print('scores:',scores)
    #result_analysis(predictions,scores)

if args.with_NIL_infer:
    logger.info('out_of_KB_thresholds_bi+cross: %.2f,%.2f',float(args.th_NIL_bi_enc),float(args.th_NIL_cross_enc))
else:
    logger.info('out_of_KB_thresholds_bi+cross: not applicable')

table_str = "\n{}".format(table)
print(table_str)
#logger.info(table_str)

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w') as f_output:
        f_output.write(str)

dataset_name = '-'.join(DATASETS[0]["name"].split('-')[0:-1]) # drop the data split mark (train/test) from the first dataset name.
if args.with_NIL_infer:
    output_res_file_name = 'results/%s-results-th1-%.2f-th2-%.2f-%s%s.txt' % (dataset_name,float(args.th_NIL_bi_enc),float(args.th_NIL_cross_enc),cross_model_setting.replace('/','_'),('-' + args.marking) if args.marking != '' else '')   
else: 
    output_res_file_name = 'results/%s-results-%s%s.txt' % (dataset_name,cross_model_setting.replace('/','_'),('-' + args.marking) if args.marking != '' else '')    
output_to_file(output_res_file_name,table_str)