# semantic embeddings for edge candidate generation
# using a BERT-model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import numpy as np

import pandas as pd
from datasets import Dataset

import argparse
from tqdm import tqdm
import json
import math 

import sys
sys.path.append("../..") # add the BLINKout+ main folder
from preprocessing.onto_snomed_owl_util import get_entity_graph_info, load_SNOMEDCT_deeponto, deeponto2dict_ids, load_deeponto_verbaliser, get_iri_from_SCTID_id, get_SCTID_id_from_iri, _extract_iris_in_parsed_complex_concept,extract_SNOMEDCT_deeponto_taxonomy,calculate_wu_palmer_sim,is_complex_concept,get_dict_iri_pair_to_lca,filter_out_complex_edges
from blink.candidate_ranking.semantic_edge_evaluation import overall_edge_wp_sim

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))

from deeponto import init_jvm
init_jvm("32g")
from deeponto.align.bertmap import BERTMapPipeline
from deeponto.utils import Tokenizer

parser = argparse.ArgumentParser(description="format the medmention dataset with different ontology settings")
parser.add_argument('--model_name', type=str,
                    help="BERT embedding model name", default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
parser.add_argument('--snomed_subset', type=str,
                    help="SNOMED-CT subset mark: Disease, CPP, etc.", default='Disease')
parser.add_argument('--data_split', type=str,
                    help="valid-NIL, test-NIL, test-NIL-all", default='test-NIL')
parser.add_argument("--pool_size",type=int,help="number of maximum candidates for each mention", default=300)
parser.add_argument("--pool_size_edge_lvl",type=int,help="number of maximum edges for each mention", default=300)
parser.add_argument("--pool_size_seed",type=int,help="number of seeded candidates (before enrichment by hops) for each mention", default=50)
parser.add_argument("--edge_ranking_by_score",action="store_true",help="Whether to rank edges by score")
parser.add_argument("--enrich_cands",action="store_true",help="Whether to enrich candidates from the seeded candidates (of number --pool_size_seed)")
parser.add_argument("--use_context",action="store_true",help="Whether to use contexts to enrich mentions and generate half of the candidates")
parser.add_argument("--percent_mention_w_ctx",type=float,help="percentage of targets created by using contexts, if use_context is chosen; 0.5 for half mention-based and half mention+ctx-based; 1.0 for pure mention+ctx-based", default=0.5)
parser.add_argument("--context_token_length",type=int,help="window size for contexts, left k tokens and right k tokens", default=5)
parser.add_argument("--enrich_edges",action="store_true",help="Whether to enrich edge candidates by hop")
parser.add_argument("--enrich_targets",action="store_true",help="Whether to enrich targets for a loose evaluation of the edges, default by 1 hop for parents and children")
parser.add_argument("--enrich_targets_2_hops",action="store_true",help="Whether to enrich targets by 2 hops, provided --enrich_targets is selected")
parser.add_argument("--use_idf_score_for_cands",action="store_true",help="Whether to use idf scores instead of embeddings for concept candidate selections")
parser.add_argument("--use_idf_score_for_edges",action="store_true",help="Whether to use idf scores instead of embeddings for edge candidate ranking")
parser.add_argument("--use_leaf_edge_score_always",action="store_true",help="Whether to always use the LEAF_EDGE_SCORE, otherwise, uses the LEAF_EDGE_SCORE only when the most similar seed concept of a mention do not have children")
parser.add_argument("--LEAF_EDGE_SCORE",type=int,help="A pre-defined score for leaf edges, i.e., <parent, NULL>, for the edge ranking: For SNOMED-CT, this is suggested to be a very high score, e.g. 1000; For DO, this is suggested to be a very low score, e.g. -1000", default=1000)
parser.add_argument("--measure_wp",action="store_true",help="Whether to calculte wu & palmer similarity")
parser.add_argument("--output_preds",action="store_true",help="Whether to output predictions, both for the aggregated and the micro-lvl (ment-edge-pair-lvl)")

args = parser.parse_args()
print('args:',args)

# Load pre-trained BERT model
#model_name = "bert-base-uncased"  # Example model, you can use other models from Hugging Face
#model_name = "prajjwal1/bert-tiny"
#model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
model_name = args.model_name
emb_tokenizer = AutoTokenizer.from_pretrained(model_name)
emb_model = AutoModel.from_pretrained(model_name)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
emb_model = emb_model.to(device)

# def get_texts_from_iri(inv_index,iri):
#     inv_index.original_index[iri1]

#output str content to a file
#input: filename and the content (str)
def _output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8") as f_output:
        f_output.write(str)

def load_ment_edge_pair_lvl_data(ment_edge_pair_lvl_data_fn):
    '''
    load mention edge pair lvl data into dict of row number to a tuple of (ment,ctx_l,ctx_r,label_concept_ori)'''
    dict_row_id_to_ment_micro = {}
    with open(ment_edge_pair_lvl_data_fn,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()
    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)  
        mention = mention_info["mention"]
        context_left = mention_info["context_left"]
        context_right = mention_info["context_right"]
        label_concept_ori = mention_info["label_concept_ori"]
        dict_row_id_to_ment_micro[ind] = (mention,context_left,context_right,label_concept_ori)
    return dict_row_id_to_ment_micro

# not used 
def idf_sim(inv_index, iri1, iri2):
    '''
    using the iri1 to query iris and then calculate idf for iri2
    this metric is symmetric for iri1 and iri2, i.e. using iri2 to query iri1 will get the same result
    '''
    idf_sim_score = 0
    concept1 = inv_index.original_index[iri1] # get concept labels from iri
    #concept2 = inv_index.original_index[iri2] 
    c1_tokens = inv_index.tokenizer(concept1)
    D = len(inv_index.original_index)
    for c1_token in c1_tokens:
        potential_candidates = inv_index.constructed_index[c1_token]
        if iri2 in potential_candidates:
            idf_sim_score += math.log10(D / len(potential_candidates))
    #print(iri1,concept1,iri2,concept2,idf_sim_score)
    return idf_sim_score

def idf_sim_iri_to_mention(inv_index, iri1, mention):
    '''
    using the mention to query iris and then calculate idf for *the iri*
    '''
    idf_sim_score = 0
    mention_tokens = inv_index.tokenizer(mention)
    D = len(inv_index.original_index)
    for mention_token in mention_tokens:
        potential_candidates = inv_index.constructed_index[mention_token]
        if iri1 in potential_candidates:
            idf_sim_score += math.log10(D / len(potential_candidates))
    return idf_sim_score

def edge_score_idf_sim(inv_index, iri1, iri2, mention):
    return 0.5 * (idf_sim_iri_to_mention(inv_index, iri1, mention) + idf_sim_iri_to_mention(inv_index, iri2, mention))

# Define a custom comparison function
def compare_tuples(a, b):
    return a[2] - b[2]

def embed_concept(model,tokenizer,text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #text = "Your input text goes here."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    mean_pooling = torch.mean(embeddings, dim=1)
    return mean_pooling.cpu().detach().numpy()[0] # one in a batch during the embedding, thus get the first one in the 2D vec

# def concept_sim(embedding1,embedding2):
#     #embedding1 = embed_concept(model,tokenizer,concept1)
#     #embedding2 = embed_concept(model,tokenizer,concept2)
#     similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
#     return similarity.item()

def cosine_similarity_np(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def edge_score_emb(vector1, vector2, vec_mention):
    return 0.5 * (cosine_similarity_np(vector1,vec_mention) + cosine_similarity_np(vector2,vec_mention))

def embed_whole_ontology(model,tokenizer,inv_index):
    dict_iri_to_vec = {}
    for iri,texts in tqdm(inv_index.original_index.items(),total=len(inv_index.original_index)):
        texts = ' '.join(texts).lower() # also make them in lower case (for uncased BERT models)
        vec = embed_concept(model,tokenizer,texts)
        dict_iri_to_vec[iri] = vec
    return dict_iri_to_vec

# def embed_mentions(model,tokenizer,inv_index):
#     dict_mentions_to_vec = {}
#     for iri,texts in tqdm(inv_index.original_index.items(),total=len(inv_index.original_index)):
#         texts = ' '.join(texts).lower() # also make them in lower case (for uncased BERT models)
#         vec = embed_concept(model,tokenizer,texts)
#         dict_mentions_to_vec[iri] = vec
#     return dict_mentions_to_vec

def rep_iri_atomic_and_complex(dict_iri_to_vec,iri):
    
    if "[EX.]" in iri:
        # average emb of elements if complex concept
        complex_iri_ele_lst = _extract_iris_in_parsed_complex_concept(iri)        
        n=0
        rep=np.zeros_like(list(dict_iri_to_vec.values())[0])
        for complex_iri_ele in complex_iri_ele_lst:
            complex_iri_ele = get_iri_from_SCTID_id(complex_iri_ele,prefix=iri_prefix)
            if complex_iri_ele in dict_iri_to_vec:
                rep += dict_iri_to_vec[complex_iri_ele]
                n += 1
        if n>0:
            rep = rep / n
    else:
        rep = dict_iri_to_vec[iri]
    return rep

def query_top_k(mention,dataset_iri_to_emb,model,tokenizer,top_k=100):
    mention_embedding = embed_concept(model,tokenizer,mention)
    #print(mention_embedding.shape)

    scores, samples = dataset_iri_to_emb.get_nearest_examples(
        "embeddings", mention_embedding, k=top_k
    )
    #print(samples['iri'],scores)
    return [(sample,score) for sample, score in zip(samples["iri"],scores)]

def query_top_k_from_men_embedding(mention_embedding,dataset_iri_to_emb,top_k=100):
    scores, samples = dataset_iri_to_emb.get_nearest_examples(
        "embeddings", mention_embedding, k=top_k
    )
    #print(samples['iri'],scores)
    return [(sample,score) for sample, score in zip(samples["iri"],scores)]

def load_edge_catalogue(edge_catalogue_fn):
    dict_ind_edge_json_info = {}
    dict_edge_tuple_to_ind = {}
    dict_parent_to_child = {}
    dict_child_to_parent = {}
    with open(edge_catalogue_fn,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    for ind, edge_info_json in enumerate(tqdm(doc)):
        edge_info = json.loads(edge_info_json)  
        #edge_str = '%s (%s) -> %s (%s), degree %d' % (edge_info["parent"], edge_info["parent_idx"], edge_info["child"], edge_info["child_idx"], edge_info["degree"])
        parent = edge_info["parent_idx"]
        child =  edge_info["child_idx"]
        edge_tuple = (parent, child)
        dict_ind_edge_json_info[ind] = edge_info
        dict_edge_tuple_to_ind[edge_tuple] = ind

        #if edge_info["degree"] == 0:
        #print('dict_parent_to_child:',dict_parent_to_child)
        dict_parent_to_child = add_dict_list(dict_parent_to_child,parent,child)
        dict_child_to_parent = add_dict_list(dict_child_to_parent,child,parent)            

    return dict_ind_edge_json_info, dict_edge_tuple_to_ind, dict_parent_to_child, dict_child_to_parent

def enrich_edge_cands_leaf(dict_ind_edge_json_info,dict_edge_tuple_to_ind,list_edge_inds):
    '''
    enrich edge candidates: adding leaf edges if the parent is predicted in an edge
    input: (i) dict of ind edges into json
           (ii) a list of edge indices to be enriched
    '''
    list_edge_inds_enriched = list_edge_inds[:]
    for edge_ind in list_edge_inds:
        edge_info = dict_ind_edge_json_info[edge_ind]
        if edge_info["child_idx"] != "SCTID_NULL":
            leaf_edge_tuple = (edge_info["parent_idx"],"SCTID_NULL")
            list_edge_inds_enriched = enrich_edge_list(dict_edge_tuple_to_ind,list_edge_inds_enriched,leaf_edge_tuple)
    return list_edge_inds_enriched

def enrich_edge_cands_hop(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds):
    '''
    enrich edge candidates: adding 2-degree edges by extending 1-degree edges
    input: (i) dict of ind edges into json
           (ii) a list of edge indices to be enriched
    '''
    list_edge_inds_enriched = list_edge_inds[:]
    for edge_ind in list_edge_inds:
        edge_info = dict_ind_edge_json_info[edge_ind]
        #if edge_info["degree"] == 0:
        parent = edge_info["parent_idx"]
        child =  edge_info["child_idx"]
        # enrich <p+,c>
        if parent in dict_child_to_parent:
            list_parents_upper = dict_child_to_parent[parent]
            for parent_upper in list_parents_upper:
                hop_edge_tuple = (parent_upper,child)
                list_edge_inds_enriched = enrich_edge_list(dict_edge_tuple_to_ind,list_edge_inds_enriched,hop_edge_tuple)
                # if hop_edge_tuple in dict_edge_tuple_to_ind:
                #     hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                #     if not hop_edge_ind in list_edge_inds_enriched:
                #         list_edge_inds_enriched.append(hop_edge_ind)
        # enrich <p,c->
        if child in dict_parent_to_child:
            list_children_lower = dict_parent_to_child[child]
            for child_lower in list_children_lower:
                hop_edge_tuple = (parent,child_lower)
                list_edge_inds_enriched = enrich_edge_list(dict_edge_tuple_to_ind,list_edge_inds_enriched,hop_edge_tuple)
                # if hop_edge_tuple in dict_edge_tuple_to_ind:
                #     hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                #     if not hop_edge_ind in list_edge_inds_enriched:
                #         list_edge_inds_enriched.append(hop_edge_ind)
        # enrich <p+,c-> (although usually with degree more than 2)
        if parent in dict_child_to_parent:
            list_parents_upper = dict_child_to_parent[parent]
            if child in dict_parent_to_child:
                list_children_lower = dict_parent_to_child[child]
                for parent_upper in list_parents_upper:
                    for child_lower in list_children_lower:
                        hop_edge_tuple = (parent_upper,child_lower)
                        list_edge_inds_enriched = enrich_edge_list(dict_edge_tuple_to_ind,list_edge_inds_enriched,hop_edge_tuple)
                        # if hop_edge_tuple in dict_edge_tuple_to_ind:
                        #     hop_edge_ind = dict_edge_tuple_to_ind[hop_edge_tuple]
                        #     if not hop_edge_ind in list_edge_inds_enriched:
                        #         list_edge_inds_enriched.append(hop_edge_ind)
        
    return list_edge_inds_enriched 

def enrich_edge_list(dict_edge_tuple_to_ind, list_edge_inds, new_edge_tuple,display=False):
    '''
    enrich the list_edge_inds with a new edge, if it is in the edge catalogue
    '''
    if new_edge_tuple in dict_edge_tuple_to_ind:
        new_edge_ind = dict_edge_tuple_to_ind[new_edge_tuple]
        if not new_edge_ind in list_edge_inds:
            if display:
                print("enrich new edge:",new_edge_ind,new_edge_tuple)
            list_edge_inds.append(new_edge_ind)
    return list_edge_inds

# add an element to a dict of list of elements for the id
def add_dict_list(dict,id,ele):
    if not id in dict:
        dict[id] = [ele] # one-element list
    else:
        list_ele = dict[id]
        list_ele.append(ele)
        dict[id] = list_ele
    return dict

#snomed_subset = "Disease"
snomed_subset = args.snomed_subset
data_split = args.data_split

# load the mention-edge-pair-lvl data
data_file_ment_edge_pair = "../../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-edges-final/%s.jsonl" % (snomed_subset,data_split)
dict_row_id_to_ment_micro = load_ment_edge_pair_lvl_data(data_file_ment_edge_pair)

data_file = "../../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-unsup/%s.jsonl" % (snomed_subset,data_split)

onto_path = "../../ontologies"

iri_prefix = "http://snomed.info/id/"

config = BERTMapPipeline.load_bertmap_config()

snomed_ct_old = load_SNOMEDCT_deeponto(onto_path + "/SNOMEDCT-US-20140901-%s-final.owl" % snomed_subset)
#snomed_ct_new = load_SNOMEDCT_deeponto(onto_path + "/SNOMEDCT-US-20170301-%s-final.owl" % snomed_subset)
dict_SCTID_onto = deeponto2dict_ids(snomed_ct_old)
onto_sno_verbaliser = load_deeponto_verbaliser(snomed_ct_old)
if args.measure_wp:
    snomed_ct_old_taxo = extract_SNOMEDCT_deeponto_taxonomy(snomed_ct_old)

# build annotation index
snomed_ct_old_index, _ = snomed_ct_old.build_annotation_index(config.annotation_property_iris)

# build inverted index
tkz = Tokenizer.from_pretrained("google/flan-t5-xxl")
inv_index_snomed_old = snomed_ct_old.build_inverted_annotation_index(snomed_ct_old_index, tkz)

print("snomed_ct_old_index:",len(snomed_ct_old_index))
for ind, (iri,txt) in enumerate(snomed_ct_old_index.items()):
    if ind < 3:
        print(ind, iri, txt)
print("inv_index_snomed_old:",inv_index_snomed_old)

# embed and load the ontology
embed_fn = "snomed-ct-%s-%s.pkl" % (snomed_subset, model_name.replace('/','-'))
if os.path.exists(embed_fn):
    with open(embed_fn, 'rb') as data_f:
        dict_iri_to_emb = pickle.load(data_f)
else:
    dict_iri_to_emb = embed_whole_ontology(emb_model,emb_tokenizer,inv_index_snomed_old)      
    with open(embed_fn, 'wb') as data_f:
        pickle.dump(dict_iri_to_emb, data_f)        
        print('data stored to',embed_fn)
print("display some embs in %s:" % embed_fn)
print("dict_iri_to_emb:",len(dict_iri_to_emb))
for ind, (iri,emb) in enumerate(dict_iri_to_emb.items()):
    if ind < 3:
        print(ind, iri, emb[:5])
# add faiss index - after storing the dict_iri_to_emb as a dataframe, then using Dataset format
iris_list = list(dict_iri_to_emb.keys())
embs_list = list(dict_iri_to_emb.values())
df_iri_to_emb = pd.DataFrame.from_dict({"iri":iris_list, "embeddings": embs_list})
dataset_iri_to_emb = Dataset.from_pandas(df_iri_to_emb)
dataset_iri_to_emb.add_faiss_index(column="embeddings")

if args.measure_wp:
    #  embed and load the shortest node depth (snd) and lowest common ancestor (lca) dicts
    snd_fn = "snomed-ct-%s-%s.pkl" % (snomed_subset, "shortest-node-depth")
    if os.path.exists(snd_fn):
        with open(snd_fn, 'rb') as data_f:
            dict_iri_to_snd = pickle.load(data_f)
            print('dict_iri_to_snd:',len(dict_iri_to_snd))
    else:
        dict_iri_to_snd = {}

    lca_fn = "snomed-ct-%s-%s.pkl" % (snomed_subset, "lowest-common-ancestor")
    if os.path.exists(lca_fn):
        with open(lca_fn, 'rb') as data_f:
            dict_iri_pair_to_lca = pickle.load(data_f)
            print('dict_iri_pair_to_lca:',len(dict_iri_pair_to_lca))
    else:
        dict_iri_pair_to_lca = {}
        #dict_iri_pair_to_lca = get_dict_iri_pair_to_lca(snomed_ct_old_taxo)

# search w/ inverted index
ave_tgt_cand_len = 0
ave_edge_cand_len = 0
num_edges = 0
num_men_leaf = 0
num_men_non_leaf = 0
hits_p = 0
hits_c = 0
hits_pc = 0
hits_pc_edge_any = 0
hits_pc_edge_all = 0
hits_pc_edge_any_leaf = 0
hits_pc_edge_all_leaf = 0
hits_pc_edge_any_non_leaf = 0
hits_pc_edge_all_non_leaf = 0
ave_overall_wp_sim_min = 0.
ave_overall_wp_sim_max = 0.
ave_overall_wp_sim_ave = 0.

pool_size=args.pool_size # candidate level
pool_size_edge_lvl=args.pool_size_edge_lvl # edge level
edge_ranking_by_score=args.edge_ranking_by_score # edge level, ranking them by score 
enrich_cands=args.enrich_cands
if enrich_cands:
    pool_size_seed = args.pool_size_seed
use_context=args.use_context
percent_mention_w_ctx=args.percent_mention_w_ctx
context_token_length = args.context_token_length
enrich_edges = args.enrich_edges
enrich_targets = args.enrich_targets
enrich_targets_2_hops = args.enrich_targets_2_hops

use_idf_score_for_cands = args.use_idf_score_for_cands
use_idf_score_for_edges = args.use_idf_score_for_edges

use_leaf_edge_score_always = args.use_leaf_edge_score_always
LEAF_EDGE_SCORE = args.LEAF_EDGE_SCORE

# load edge catalogue - only for edge cands enrichment
if enrich_edges:
    edge_catalogue_fn = "../../ontologies/SNOMEDCT-US-20140901-%s-edges-all.jsonl" % snomed_subset
    dict_ind_edge_json_info, dict_edge_tuple_to_ind, dict_parent_to_child, dict_child_to_parent = load_edge_catalogue(edge_catalogue_fn)

wp_metric_denominator = 0 # denominator for wp metric - only considering when pred and gold are not atomic.
list_ranked_edge_w_scores = [] # a 2d-list to store all mentions' (as a list) ranked_edge_w_scores, which itself is a list of tuples of (edge, score) 
dict_men_ctx_id_tuple_to_row_id_aggregated = {}
with open(data_file,encoding='utf-8-sig') as f_content:
    doc = f_content.readlines()
for ind, mention_info_json in enumerate(tqdm(doc)):
    mention_info = json.loads(mention_info_json)  
    mention = mention_info["mention"]
    
    context_left = mention_info["context_left"]
    context_right = mention_info["context_right"]
    label_concept_ori = mention_info["label_concept_ori"]
    dict_men_ctx_id_tuple_to_row_id_aggregated[(mention,context_left,context_right,label_concept_ori)] = ind

    if use_context:
        context_left_tokens = context_left.split(" ")
        context_right_tokens = context_right.split(" ")
        context_left_tokens_len = len(context_left_tokens)
        if context_left_tokens_len > context_token_length:
            # update the context left if it has enough tokens to be truncated by the context token length
            context_left = ' '.join(context_left_tokens[context_left_tokens_len-context_token_length:])        
        context_right = ' '.join(context_right_tokens[:context_left_tokens_len])
        mention_w_context = context_left + mention + context_right

#for ind, iri in enumerate(snomed_ct_old_index.keys()):
    #if ind > 3:
    #    continue
    #iri_label = snomed_ct_old_index[iri]
    #print("iri to search:", iri, iri_label)
    
    # 1. here do the search - to find seed concepts as tgt_cands_w_scores
    if use_idf_score_for_cands:
        tgt_cands_w_scores = [(x,s) for x, s in inv_index_snomed_old.idf_select(mention, pool_size=pool_size)] # by idf
    else:
        # get mention embedding
        mention_emb = embed_concept(emb_model,emb_tokenizer,mention.lower())
        #tgt_cands_w_scores = query_top_k(mention.lower(),dataset_iri_to_emb,emb_model,emb_tokenizer,top_k=pool_size) # by embedding similarity
        tgt_cands_w_scores = query_top_k_from_men_embedding(mention_emb,dataset_iri_to_emb,top_k=pool_size) # by embedding similarity
    if use_context:
        num_men_target_to_keep = pool_size if not enrich_cands else pool_size_seed
        tgt_cands_w_scores = tgt_cands_w_scores[:int(num_men_target_to_keep*(1-percent_mention_w_ctx))] # cut by half if percent_mention_w_ctx as 0.5, using purely contexts if percent_mention_w_ctx as 1.0.
        if use_idf_score_for_cands:
            tgt_cands_w_scores_w_ctx = [(x,s) for x, s in inv_index_snomed_old.idf_select(mention_w_context, pool_size=pool_size)] # by idf
        else:
            # get mention embedding (with contexts)
            mention_emb_w_ctx = embed_concept(emb_model,emb_tokenizer,mention_w_context.lower())
            #tgt_cands_w_scores_w_ctx = query_top_k(mention_w_context.lower(),dataset_iri_to_emb,emb_model,emb_tokenizer,top_k=pool_size) # by embedding similarity
            tgt_cands_w_scores_w_ctx = query_top_k_from_men_embedding(mention_emb_w_ctx,dataset_iri_to_emb,top_k=pool_size) # by embedding similarity
        # union the cand from the mention and the cand from mention_w_context, and make the elements unique
        tgt_cands_w_scores_in_loop = tgt_cands_w_scores[:] # create a copy to be used in the loop, since tgt_cands_w_scores will be updated 
        for tgt_cand_score_w_ctx_tuple in tgt_cands_w_scores_w_ctx:
            x_ctx, s_ctx = tgt_cand_score_w_ctx_tuple
            add_x_ctx = True
            for tgt_cand_w_score_tuple in tgt_cands_w_scores_in_loop:
                x, _ = tgt_cand_w_score_tuple
                if x_ctx == x:
                    add_x_ctx = False
                    break
            if add_x_ctx:
                tgt_cands_w_scores.append(tgt_cand_score_w_ctx_tuple)
                if len(tgt_cands_w_scores) == num_men_target_to_keep:
                    break
        #tgt_cands_w_scores = tgt_cands_w_scores[:num_men_target_to_keep]

    tgt_cands = [x for x, _ in tgt_cands_w_scores]
    tgt_cand_scores = [s for _, s in tgt_cands_w_scores]
    if ind < 3:
        print(ind, mention if not use_context else mention_w_context, "tgt_cands:",len(tgt_cands),tgt_cands)
        print(ind, "tgt_cand_scores:",len(tgt_cand_scores),tgt_cand_scores)
    
    ## here get the labels of the searched outcome
    #tgt_cand_labels = [(x, snomed_ct_old_index[x]) for x in tgt_cands]
    ##print("tgt_cand_labels:",len(tgt_cand_labels), tgt_cand_labels)
    
    # 2. for cand-level results: enrich the search outcome with parent and children iris
    if enrich_cands:
        # only use a subset of the cands, top-k as defined with pool_size_seed
        tgt_cands_enriched = tgt_cands[:pool_size_seed]
        for iri in tgt_cands[:pool_size_seed]:
            sctid = get_SCTID_id_from_iri(iri,prefix=iri_prefix)
            #if sctid == "":
            #    continue
            #print("sctid:",sctid)
            if not sctid.isnumeric():
                print("complex concepts:",sctid)
                continue
            # get direct parents and children
            children_str, parents_str, _, _, _ = get_entity_graph_info(snomed_ct_old,iri,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type="all",prefix=iri_prefix)
            list_children = children_str.split('|') if children_str != "" else []
            list_parents = parents_str.split('|') if parents_str != "" else []
            # enrich direct parents and children
            for child in list_children:
                child_iri = get_iri_from_SCTID_id(child,prefix=iri_prefix)
                if not child_iri in tgt_cands_enriched:
                    # break the loop if there are enough cands
                    if len(tgt_cands_enriched) == pool_size:
                        break
                    tgt_cands_enriched.append(child_iri)
                    
            for parent in list_parents:
                parent_iri = get_iri_from_SCTID_id(parent,prefix=iri_prefix)
                if not parent_iri in tgt_cands_enriched:
                    # break the loop if there are enough cands
                    if len(tgt_cands_enriched) == pool_size:
                        break
                    tgt_cands_enriched.append(parent_iri)
            
            # break the loop if there are enough cands
            if len(tgt_cands_enriched) == pool_size:
                break            
        if ind < 3:
            print(ind, "tgt_cands_enriched:",len(tgt_cands_enriched),tgt_cands_enriched)
    else:
        tgt_cands_enriched = tgt_cands[:]

    tgt_cands_enriched = tgt_cands_enriched + ['http://snomed.info/id/SCTID_NULL'] # enrich with the NULL node for the leaf node edges, <parent, NULL>.
    ave_tgt_cand_len = ave_tgt_cand_len + len(tgt_cands_enriched)

    # 3. for edge-level results
    ranked_edge_w_scores = [] # edge w/ score 3-tuple list: (p, c, score)
    # ranking: top-k cands by idf scores, then for each cand C, generate edges of <C+, C>, edges of <C, C->, and edges of <C+, C->, where C+ and C- are the parents and children of the concept C in the ontology O; the generated edges are ranked by an mention-to-edge score (see below). Then, go to another cand, generate edges and rank them, and append them after the first cand.
    # mention-to-edge score: the generated edges to be ranked by the avearge value of parent-mention similarity score and children-mention simlarity score (both either by idf or fixed embedding similarity).
    # also we add <C,NULL> and put it as the first in the ranking (for SNOMED-CT).

    # edge generation and ranking from tgt_cands
    # only use a subset of the cands, top-k as defined with pool_size_seed
    is_mention_leaf_assumed = False # assumption on whether the mention is to be placed as a leaf node - this is true when the most similar seed cand (tgt_cand) do not have children or len(list_children) == 0; this will affect the score for the enriched leaf edges (as LEAF_EDGE_SCORE if True else -LEAF_EDGE_SCORE)
    tgt_cands_for_edge_gen = tgt_cands[:pool_size_seed]
    for tgt_cand_ind, iri in enumerate(tgt_cands_for_edge_gen):
        #if len(ranked_edge_w_scores) >= pool_size_edge_lvl:
        #    break

        # step 0: get direct parents and children (for atomic cands)        
        sctid = get_SCTID_id_from_iri(iri,prefix=iri_prefix)
        #if sctid == "":
        #    continue
        #print("sctid:",sctid)
        if not sctid.isnumeric():
            print("complex concepts:",sctid)            
            continue
        children_str, parents_str, _, _, _ = get_entity_graph_info(snomed_ct_old,iri,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type="all",prefix=iri_prefix)
        list_children = children_str.split('|') if children_str != "" else []
        list_parents = parents_str.split('|') if parents_str != "" else []
        
        # step 1: append leaf node edge - and set a very high score, e.g. 1000 
        if tgt_cand_ind == 0 and len(list_children) == 0:
            is_mention_leaf_assumed = True
        ranked_edge_w_scores.append((iri,'http://snomed.info/id/SCTID_NULL',LEAF_EDGE_SCORE if is_mention_leaf_assumed or use_leaf_edge_score_always else -LEAF_EDGE_SCORE))

        # step 2: generate edges of <C+, C>, edges of <C, C->, and edges of <C+, C-> from the original concept cands (to note, this is independent of enrich_cands previously)
        # also to enrich these edges like in biencoder/candidate_analysis.py

        for parent in list_parents:
            parent_iri = get_iri_from_SCTID_id(parent,prefix=iri_prefix)
            if use_idf_score_for_edges:
                #score_sim = idf_sim(inv_index_snomed_old,iri,parent_iri)
                if use_context:
                    score_sim_ctx = edge_score_idf_sim(inv_index_snomed_old,iri,parent_iri,mention_w_context)
                    score_sim_non_ctx = edge_score_idf_sim(inv_index_snomed_old,iri,parent_iri,mention)
                    score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                else:
                    score_sim = edge_score_idf_sim(inv_index_snomed_old,iri,parent_iri,mention)
                #ranked_edge_w_scores.append((list_parents,iri,score_idf_sim))
                #print(inv_index_snomed_old.original_index[iri])
                #print(inv_index_snomed_old.original_index[parent_iri])
            else:
                vec_iri1 = rep_iri_atomic_and_complex(dict_iri_to_emb,iri)
                vec_iri2 = rep_iri_atomic_and_complex(dict_iri_to_emb,parent_iri)
                if use_context:
                    score_sim_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb_w_ctx)
                    score_sim_non_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                    score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                else:
                    score_sim = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                #score_sim = cosine_similarity_np(rep_iri_atomic_and_complex(dict_iri_to_emb,iri),rep_iri_atomic_and_complex(dict_iri_to_emb,parent_iri))
            ranked_edge_w_scores.append((parent_iri,iri,score_sim))
            # for child in list_children:
            #     score_idf_sim = idf_sim(parent_iri,child_iri)
            #     ranked_edge_w_scores.append((parent_iri,child_iri,score_idf_sim))

        for child in list_children:
            child_iri = get_iri_from_SCTID_id(child,prefix=iri_prefix)
            if use_idf_score_for_edges:
                if use_context:
                    score_sim_ctx = edge_score_idf_sim(inv_index_snomed_old,iri,child_iri,mention_w_context)
                    score_sim_non_ctx = edge_score_idf_sim(inv_index_snomed_old,iri,child_iri,mention)
                    score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                else:
                    score_sim = edge_score_idf_sim(inv_index_snomed_old,iri,child_iri,mention)
                #ranked_edge_w_scores.append((iri,child_iri,score_idf_sim))
            else:
                vec_iri1 = rep_iri_atomic_and_complex(dict_iri_to_emb,iri)
                vec_iri2 = rep_iri_atomic_and_complex(dict_iri_to_emb,child_iri)
                if use_context:
                    score_sim_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb_w_ctx)
                    score_sim_non_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                    score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                else:
                    score_sim = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                #score_sim = cosine_similarity_np(rep_iri_atomic_and_complex(dict_iri_to_emb,iri),rep_iri_atomic_and_complex(dict_iri_to_emb,child_iri))
            ranked_edge_w_scores.append((iri,child_iri,score_sim))
            
            for parent in list_parents:
                parent_iri = get_iri_from_SCTID_id(parent,prefix=iri_prefix)
                if use_idf_score_for_edges:
                    if use_context:
                        score_sim_ctx = edge_score_idf_sim(inv_index_snomed_old,parent_iri,child_iri,mention_w_context)
                        score_sim_non_ctx = edge_score_idf_sim(inv_index_snomed_old,parent_iri,child_iri,mention)
                        score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                    else:
                        score_sim = edge_score_idf_sim(inv_index_snomed_old,parent_iri,child_iri,mention)
                    #ranked_edge_w_scores.append((parent_iri,child_iri,score_idf_sim))
                else:
                    vec_iri1 = rep_iri_atomic_and_complex(dict_iri_to_emb,parent_iri)
                    vec_iri2 = rep_iri_atomic_and_complex(dict_iri_to_emb,child_iri)
                    if use_context:
                        score_sim_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb_w_ctx)
                        score_sim_non_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                        score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                    else:
                        score_sim = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                    #score_sim = cosine_similarity_np(rep_iri_atomic_and_complex(dict_iri_to_emb,parent_iri),rep_iri_atomic_and_complex(dict_iri_to_emb,child_iri))
                ranked_edge_w_scores.append((parent_iri,child_iri,score_sim))

    # enrich edges
    # using the same procedure as in blink/biencoder/candidate_analysis.py
    # by processing the edge catalogue .jsonl file (instead of .owl)
    if enrich_edges:
        list_edge_inds = []
        for edge_w_score in ranked_edge_w_scores:
            #print("edge_w_score[0] and [1]:", edge_w_score[0],edge_w_score[1])
            parent = get_SCTID_id_from_iri(edge_w_score[0],prefix=iri_prefix)
            child = get_SCTID_id_from_iri(edge_w_score[1],prefix=iri_prefix)
            if (parent,child) in dict_edge_tuple_to_ind:
                edge_ind = dict_edge_tuple_to_ind[(parent,child)]
                list_edge_inds.append(edge_ind)
        list_edge_inds_enriched = enrich_edge_cands_leaf(dict_ind_edge_json_info,dict_edge_tuple_to_ind,list_edge_inds)
        list_edge_inds_enriched = enrich_edge_cands_hop(dict_ind_edge_json_info,dict_edge_tuple_to_ind,dict_parent_to_child,dict_child_to_parent,list_edge_inds)
        
        # form back the format of ranked_edge_w_scores - add the newly enriched edges and calculate their scores
        for edge_ind in list_edge_inds_enriched:
            if not edge_ind in list_edge_inds:
                edge_info = dict_ind_edge_json_info[edge_ind]
                parent_iri = get_iri_from_SCTID_id(edge_info["parent_idx"],prefix=iri_prefix)
                child_iri = get_iri_from_SCTID_id(edge_info["child_idx"],prefix=iri_prefix)
                if parent_iri == iri_prefix+'SCTID_THING':
                    continue # ignore parent as THING
                if child_iri == iri_prefix+'SCTID_NULL':                    
                    score_sim = LEAF_EDGE_SCORE if is_mention_leaf_assumed or use_leaf_edge_score_always else -LEAF_EDGE_SCORE # set a leaf edge score for child as NULL
                else:    
                    if use_idf_score_for_edges:
                        if use_context:
                            score_sim_ctx = edge_score_idf_sim(inv_index_snomed_old,parent_iri,child_iri,mention_w_context)
                            score_sim_non_ctx = edge_score_idf_sim(inv_index_snomed_old,parent_iri,child_iri,mention)
                            score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                        else:
                            score_sim = edge_score_idf_sim(inv_index_snomed_old,parent_iri,child_iri,mention)
                        #score_sim = idf_sim(inv_index_snomed_old,parent_iri,child_iri)
                        #ranked_edge_w_scores.append((parent_iri,child_iri,score_idf_sim))
                    else:
                        vec_iri1 = rep_iri_atomic_and_complex(dict_iri_to_emb,parent_iri)
                        vec_iri2 = rep_iri_atomic_and_complex(dict_iri_to_emb,child_iri)
                        if use_context:
                            score_sim_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb_w_ctx)
                            score_sim_non_ctx = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                            score_sim = score_sim_ctx * percent_mention_w_ctx + score_sim_non_ctx * (1-percent_mention_w_ctx)
                        else:
                            score_sim = edge_score_emb(vec_iri1,vec_iri2,mention_emb)
                        #score_sim = cosine_similarity_np(rep_iri_atomic_and_complex(dict_iri_to_emb,parent_iri),rep_iri_atomic_and_complex(dict_iri_to_emb,child_iri))
                ranked_edge_w_scores.append((parent_iri,child_iri,score_sim))

    if edge_ranking_by_score:
        ranked_edge_w_scores = sorted(ranked_edge_w_scores, key=lambda x: x[2],reverse=True) #sort the edges by score
        #ranked_edge_w_scores = sorted(ranked_edge_w_scores, key=itemgetter(2),reverse=True) #sort the edges by score
        #ranked_edge_w_scores = sorted(ranked_edge_w_scores, key=cmp_to_key(compare_tuples),reverse=True) #sort the edges by score
    print('ranked_edge_w_scores, before cap:',len(ranked_edge_w_scores))
    print('ranked_edge_w_scores, before cap:', ranked_edge_w_scores[:5])
    ranked_edge_w_scores = ranked_edge_w_scores[:pool_size_edge_lvl]
    ave_edge_cand_len = ave_edge_cand_len + len(ranked_edge_w_scores)

    # append ranked_edge_w_scores to list
    list_ranked_edge_w_scores.append(ranked_edge_w_scores)

    # 4. eval
    #parent_concept = mention_info["parent_concept"]
    #child_concept = mention_info["child_concept"]
    parents_concept = mention_info["parents_concept"]
    children_concept = mention_info["children_concept"]
    list_parents_concept = parents_concept.split("|")
    list_children_concept = children_concept.split("|")

    pc_paths = mention_info["parents-children_concept"]
    list_pc_paths = pc_paths.split("|")
    if list_pc_paths == ['']:
        continue    
    num_edges = num_edges + len(list_pc_paths)

    ranked_edges = [(p,c) for p, c, _ in ranked_edge_w_scores]        
    gold_edges = []
    is_mention_leaf = False
    p_caught = False # at least one parent met
    c_caught = False # at least one child met
    pc_caught = False # at least one pc edge met, on the cand-level
    #pc_all_caught = True # all pc edges met, on the cand-level
    pc_edge_caught = False # at least one pc edge met, on the edge-level 
    pc_edge_caught_leaf = False # at least one pc edge met, on the edge-level, for leaf edges
    pc_edge_caught_non_leaf = False # at least one pc edge met, on the edge-level, for non-leaf edges
    pc_all_edge_caught = True # all pc edges met, on the edge-level 
    
    # check if it is a leaf mention (a mention that can be placed to at least a leaf edge) by looping over all the pc_path
    for pc_path in list_pc_paths:
        pc_path_ele = pc_path.split("-")
        parent_concept = pc_path_ele[0]
        child_concept = pc_path_ele[1]
        if child_concept.endswith("_NULL"):
            is_mention_leaf = True
            break

    pc_all_edge_caught_leaf = is_mention_leaf # all pc edges met, on the edge-level, for leaf edges
    pc_all_edge_caught_non_leaf = not is_mention_leaf # all pc edges met, on the edge-level, for non-leaf edges 

    for pc_path in list_pc_paths:
        p_caught_now, c_caught_now = False, False # if parent, child in the *current* target edge (possibly extended by one-hop) is caught 
        pc_path_ele = pc_path.split("-")
        parent_concept = pc_path_ele[0]
        child_concept = pc_path_ele[1]
        target_p_iri_ori = get_iri_from_SCTID_id(parent_concept,prefix=iri_prefix)
        target_c_iri_ori = get_iri_from_SCTID_id(child_concept,prefix=iri_prefix)    
        #print('target_p_iri,target_c_iri:',target_p_iri,target_c_iri)

        list_target_p_iri = [target_p_iri_ori]
        list_target_c_iri = [target_c_iri_ori]

        if enrich_targets:
            # enrich target p and c with upper and lower concepts, resp.
            _, parents_str, _, _, _ = get_entity_graph_info(snomed_ct_old,target_p_iri_ori,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type="all",prefix=iri_prefix)
            list_parents = parents_str.split('|')

            childrens_str, _, _, _, _ = get_entity_graph_info(snomed_ct_old,target_p_iri_ori,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type="all",prefix=iri_prefix)
            list_children = childrens_str.split('|')

            for parent in list_parents:
                parent_iri = get_iri_from_SCTID_id(parent,prefix=iri_prefix)
                if not parent_iri in list_target_p_iri:
                    list_target_p_iri.append(parent_iri)

                if enrich_targets_2_hops:
                    _, parents_2_hop_str, _, _, _ = get_entity_graph_info(snomed_ct_old,parent_iri,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type="all",prefix=iri_prefix)
                    list_parents_2_hops = parents_2_hop_str.split('|')

                    for parent_2_hops in list_parents_2_hops:
                        parent_2_hops_iri = get_iri_from_SCTID_id(parent_2_hops,prefix=iri_prefix)
                        if not parent_2_hops_iri in list_target_p_iri:
                            list_target_p_iri.append(parent_2_hops_iri)

            for child in list_children:
                child_iri = get_iri_from_SCTID_id(child,prefix=iri_prefix)
                if not child_iri in list_target_c_iri:
                    list_target_c_iri.append(child_iri)

                if enrich_targets_2_hops:
                    children_2_hop_str, _, _, _, _ = get_entity_graph_info(snomed_ct_old,child_iri,dict_SCTID_onto,onto_verbaliser=onto_sno_verbaliser,concept_type="all",prefix=iri_prefix)
                    list_children_2_hops = children_2_hop_str.split('|')

                    for child_2_hops in list_children_2_hops:
                        child_2_hops_iri = get_iri_from_SCTID_id(child_2_hops,prefix=iri_prefix)
                        if not child_2_hops_iri in list_target_c_iri:
                            list_target_c_iri.append(child_2_hops_iri)
        
        # 4.1 cand-level evaluation:
        if not pc_caught:
            # if any of the (extended) target parent is met, then the mention is inserted
            for target_p_iri in list_target_p_iri:
                if target_p_iri in tgt_cands_enriched:
                    p_caught_now = True
                    p_caught = True
                    break

            # if any of the (extended) target child is met, then the mention is inserted
            for target_c_iri in list_target_c_iri:
                if target_c_iri in tgt_cands_enriched:
                    c_caught_now = True
                    c_caught = True
                    break
            
            # if any of the pair of (extended) target parent and child is met, then the mention is inserted
            if p_caught_now and c_caught_now:
                pc_caught = True
                #hits_pc += 1
                #break

        # 4.2 edge-level evaluation:
        gold_edges_enriched_from_one = [(p,c) for p in list_target_p_iri for c in list_target_c_iri]

        for gold_edge in gold_edges_enriched_from_one:
            if gold_edge in ranked_edges:
                pc_edge_caught = True
                _, c_ = gold_edge
                if c_.endswith("_NULL"):
                    if is_mention_leaf:
                        pc_edge_caught_leaf = True
                else:
                    if not is_mention_leaf:
                        pc_edge_caught_non_leaf = True
                #break
        # for target_p_iri in list_target_p_iri:
        #     for target_c_iri in list_target_c_iri:
        #         if (target_p_iri,target_c_iri) in ranked_edges:
        #             pc_edge_caught = True
        #             break
        #     if pc_edge_caught:
        #         break
        if not (target_p_iri_ori,target_c_iri_ori) in ranked_edges:
            print(target_p_iri_ori, '->', target_c_iri_ori, 'not predicted')
            pc_all_edge_caught = False   
            if target_c_iri_ori.endswith("_NULL"):
                if is_mention_leaf:
                    pc_all_edge_caught_leaf = False
            else:
                if not is_mention_leaf:
                    pc_all_edge_caught_non_leaf = False
        
        # update all gold edges
        gold_edges = gold_edges + [gold_edge for gold_edge in gold_edges_enriched_from_one if not gold_edge in gold_edges]

    if p_caught:
        hits_p += 1
    if c_caught:
        hits_c += 1
    if pc_caught:
        hits_pc += 1
    if is_mention_leaf:
        num_men_leaf += 1
    if pc_edge_caught:
        hits_pc_edge_any += 1
    if pc_all_edge_caught:
        hits_pc_edge_all += 1
    if pc_edge_caught_leaf:
        hits_pc_edge_any_leaf += 1
    if pc_all_edge_caught_leaf:
        hits_pc_edge_all_leaf += 1
    if pc_edge_caught_non_leaf:
        hits_pc_edge_any_non_leaf += 1
    if pc_all_edge_caught_non_leaf:
        hits_pc_edge_all_non_leaf += 1

    if args.measure_wp:
        ranked_edges_atomic = filter_out_complex_edges(ranked_edges)
        gold_edges_atomic = filter_out_complex_edges(gold_edges)

        if len(ranked_edges_atomic) == 0:
            print('number of pred edges including complex:',len(ranked_edges))
            print('pred edges including complex:', ranked_edges)
            continue

        if len(gold_edges_atomic) == 0:
            print('number of gold edges including complex:',len(gold_edges))
            print('gold edges including complex:', gold_edges)
            continue

        men_overall_edge_wp_sim_ave,men_overall_edge_wp_sim_min,men_overall_edge_wp_sim_max, dict_iri_to_snd, dict_iri_pair_to_lca = overall_edge_wp_sim(snomed_ct_old_taxo,ranked_edges_atomic,gold_edges_atomic,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca)
        ave_overall_wp_sim_ave += men_overall_edge_wp_sim_ave
        ave_overall_wp_sim_min += men_overall_edge_wp_sim_min
        ave_overall_wp_sim_max += men_overall_edge_wp_sim_max
        wp_metric_denominator += 1

    # for parent_concept in list_parents_concept:
    #     target_p_iri = get_iri_from_SCTID_id(parent_concept,prefix=iri_prefix)
    #     if target_p_iri in tgt_cands_enriched:
    #         hits_p += 1

    #     for child_concept in list_children_concept:
    #         target_c_iri = get_iri_from_SCTID_id(child_concept,prefix=iri_prefix)    
    #         if target_c_iri in tgt_cands_enriched:
    #             hits_c += 1

    #     if (target_p_iri in tgt_cands_enriched) and (target_c_iri in tgt_cands_enriched):
    #         hits_pc += 1

if args.output_preds:
    # save list_ranked_edge_w_scores as file
    ranked_edges_output_fn = "edge_preds%sto%s%s%s_%s.jsonl" % (pool_size_seed,pool_size_edge_lvl,'_enrich' if enrich_edges else '','_w_o_ctx' if not use_context else '', data_split)
    list_men_ind_to_cand_dict_str = []
    for ind, ranked_edge_w_scores_ in enumerate(list_ranked_edge_w_scores):
        dict_cand_to_score = {}
        dict_men_ind_to_cand_dict = {}
        for p_, c_, score_ in ranked_edge_w_scores_:
            dict_cand_to_score[p_ + ' -> ' + c_] = score_
        dict_men_ind_to_cand_dict[ind] = dict_cand_to_score
        ind_to_cand_dict_str = json.dumps(dict_men_ind_to_cand_dict)
        list_men_ind_to_cand_dict_str.append(ind_to_cand_dict_str)
    _output_to_file(ranked_edges_output_fn,'\n'.join(list_men_ind_to_cand_dict_str))

    # get micro-lvl (ment-edge-pair-lvl) row ids to aggregated (unsup) row ids.
    ranked_edges_micro_output_fn = "edge_preds_micro%sto%s%s%s_%s.jsonl" % (pool_size_seed,pool_size_edge_lvl,'_enrich' if enrich_edges else '','_w_o_ctx' if not use_context else '', data_split)
    list_men_ind_to_cand_dict_str_micro = []
    for row_id_micro,ment_ctx_id_tuple in dict_row_id_to_ment_micro.items():
        row_id_aggregated = dict_men_ctx_id_tuple_to_row_id_aggregated[ment_ctx_id_tuple]
        pred_row_micro = list_men_ind_to_cand_dict_str[row_id_aggregated]
        list_men_ind_to_cand_dict_str_micro.append(pred_row_micro)
    _output_to_file(ranked_edges_micro_output_fn,'\n'.join(list_men_ind_to_cand_dict_str_micro))

if args.measure_wp:
    # save shortest node depth (snd) and lowest common ancestor (lca) dicts
    with open(snd_fn, 'wb') as data_f:
        pickle.dump(dict_iri_to_snd, data_f)        
        print('shortest node depth dict stored to',snd_fn)
    with open(lca_fn, 'wb') as data_f:
        pickle.dump(dict_iri_pair_to_lca, data_f)        
        print('lowest common ancestor dict stored to',lca_fn)

num_men_non_leaf = len(doc)-num_men_leaf

print('num_gold_edges:', num_edges)
print('num_mentions:', len(doc))
print('num_mentions_leaf:', num_men_leaf)
print('num_mentions_non_leaf:', num_men_non_leaf)

print('ave tgt cands len:', ave_tgt_cand_len / len(doc))
print('parent recall:', hits_p / len(doc), hits_p, 'out of', len(doc))
print('child recall:', hits_c / len(doc), hits_c, 'out of', len(doc))       
print('edge recall:', hits_pc / len(doc), hits_pc, 'out of', len(doc))

print('ave edge cands len:', ave_edge_cand_len / len(doc))
print('edge recall, edge-level, any:', hits_pc_edge_any / len(doc), hits_pc_edge_any, 'out of', len(doc))
print('edge recall, edge-level, all:', hits_pc_edge_all / len(doc), hits_pc_edge_all, 'out of', len(doc))
print('edge recall, edge-level, any, leaf:', hits_pc_edge_any_leaf / num_men_leaf, hits_pc_edge_any_leaf, 'out of', num_men_leaf)
print('edge recall, edge-level, all, leaf:', hits_pc_edge_all_leaf / num_men_leaf, hits_pc_edge_all_leaf, 'out of', num_men_leaf)
print('edge recall, edge-level, any, non leaf:', hits_pc_edge_any_non_leaf / num_men_non_leaf, hits_pc_edge_any_non_leaf, 'out of', num_men_non_leaf)
print('edge recall, edge-level, all, non leaf:', hits_pc_edge_all_non_leaf / num_men_non_leaf, hits_pc_edge_all_non_leaf, 'out of', num_men_non_leaf)
if args.measure_wp:
    print('edge wu & palmer sim ave over pred edges:', ave_overall_wp_sim_ave / wp_metric_denominator)
    print('edge wu & palmer sim min over pred edges:', ave_overall_wp_sim_min / wp_metric_denominator)
    print('edge wu & palmer sim max over pred edges:', ave_overall_wp_sim_max / wp_metric_denominator)
    print('wu & palmer sim denominator (ment w/ pred & gold all atomic ):', wp_metric_denominator)
'''
Recent console outputs:

Disease ( embedding search + ranking)
    w/ direct parents/children, r@10 (cand-lvl & edge-lvl cap at 50)    
    
    sapbert
    (w/o target enrich, uncased)
        num_gold_edges: 1637
        ave tgt cands len: 36.13719008264463
        parent recall: 0.5008264462809917 303 out of 605
        child recall: 0.9570247933884297 579 out of 605
        edge recall: 0.4909090909090909 297 out of 605
        ave edge cands len: 44.38181818181818
        edge recall, edge-level: 0.22975206611570248 139 out of 605
        edge recall, edge-level, all: 0.1256198347107438 76 out of 605

    + target 1 hop
        num_gold_edges: 1637
        ave tgt cands len: 36.5603305785124
        parent recall: 0.509090909090909 308 out of 605
        child recall: 0.9669421487603306 585 out of 605
        edge recall: 0.509090909090909 308 out of 605
        ave edge cands len: 45.183471074380165
        edge recall, edge-level: 0.22644628099173553 137 out of 605

        edge recall, edge-level: 0.06611570247933884 40 out of 605 (if w/o setting leaf edges with a high score, i.e., 1000)
        
        (uncased)
        ave tgt cands len: 36.13719008264463
        parent recall: 0.5057851239669422 306 out of 605
        child recall: 0.9570247933884297 579 out of 605
        edge recall: 0.49586776859504134 300 out of 605
        ave edge cands len: 44.38181818181818
        edge recall, edge-level: 0.22975206611570248 139 out of 605

    bert-tiny

    num_gold_edges: 1637
    ave tgt cands len: 37.40495867768595
    parent recall: 0.2528925619834711 153 out of 605
    child recall: 0.8694214876033057 526 out of 605
    edge recall: 0.228099173553719 138 out of 605
    ave edge cands len: 42.922314049586774
    edge recall, edge-level: 0.05950413223140496 36 out of 605

    + target 1 hop
    w/ direct parents/children, r@50 (cand-lvl cap at 100 & edge-lvl cap at 200)

    num_gold_edges: 1637
    ave tgt cands len: 100.4
    parent recall: 0.6297520661157024 381 out of 605
    child recall: 0.9884297520661157 598 out of 605
    edge recall: 0.6297520661157024 381 out of 605
    ave edge cands len: 197.6710743801653
    edge recall, edge-level: 0.2644628099173554 160 out of 605

    
    w/ direct parents/children, r@50 (cand-lvl cap at 300 & edge-lvl cap at 300)

    (w/o target enrich, uncased)
    num_gold_edges: 1637
    ave tgt cands len: 177.27438016528924
    parent recall: 0.6809917355371901 412 out of 605
    child recall: 0.9884297520661157 598 out of 605
    edge recall: 0.6809917355371901 412 out of 605
    ave edge cands len: 268.3801652892562
    edge recall, edge-level: 0.32727272727272727 198 out of 605

    + target 1 hop
    num_gold_edges: 1637
    ave tgt cands len: 176.87107438016528
    parent recall: 0.6925619834710743 419 out of 605
    child recall: 0.9884297520661157 598 out of 605
    edge recall: 0.6925619834710743 419 out of 605
    ave edge cands len: 268.8826446280992
    edge recall, edge-level: 0.36363636363636365 220 out of 605

    (uncased)
    ave tgt cands len: 177.27438016528924
    parent recall: 0.6892561983471074 417 out of 605
    child recall: 0.9884297520661157 598 out of 605
    edge recall: 0.6892561983471074 417 out of 605
    ave edge cands len: 268.3801652892562
    edge recall, edge-level: 0.3322314049586777 201 out of 605

    (uncased, pubmedbert)
    num_gold_edges: 1637
    ave tgt cands len: 181.36198347107438
    parent recall: 0.7090909090909091 429 out of 605
    child recall: 0.8925619834710744 540 out of 605
    edge recall: 0.6396694214876033 387 out of 605
    ave edge cands len: 256.93388429752065
    edge recall, edge-level: 0.15867768595041323 96 out of 605

    (mention_w_context only)
    num_gold_edges: 1637
    ave tgt cands len: 131.3404958677686
    parent recall: 0.5652892561983471 342 out of 605
    child recall: 0.9520661157024793 576 out of 605
    edge recall: 0.5322314049586777 322 out of 605
    ave edge cands len: 206.15537190082645
    edge recall, edge-level: 0.15041322314049588 91 out of 605

    (mention+mention_w_context, union their candidates)
    num_gold_edges: 1637
    ave tgt cands len: 167.09586776859504
    parent recall: 0.8148760330578513 493 out of 605
    child recall: 0.9603305785123967 581 out of 605
    edge recall: 0.7818181818181819 473 out of 605
    ave edge cands len: 232.2512396694215
    edge recall, edge-level: 0.0859504132231405 52 out of 605

    (half+half, pool_size)
    num_gold_edges: 1637
    ave tgt cands len: 161.5818181818182
    parent recall: 0.6760330578512397 409 out of 605
    child recall: 0.9553719008264463 578 out of 605
    edge recall: 0.6347107438016529 384 out of 605
    ave edge cands len: 226.49421487603306
    edge recall, edge-level: 0.10578512396694215 64 out of 605

    (half+half, pool_size_seed)
    ave tgt cands len: 149.59504132231405
    parent recall: 0.5933884297520661 359 out of 605
    child recall: 0.9768595041322314 591 out of 605
    edge recall: 0.5752066115702479 348 out of 605
    ave edge cands len: 205.4495867768595
    edge recall, edge-level: 0.10082644628099173 61 out of 605

    (pubmedbert)
    ave tgt cands len: 143.6099173553719
    parent recall: 0.4826446280991736 292 out of 605
    child recall: 0.9107438016528926 551 out of 605
    edge recall: 0.43305785123966944 262 out of 605
    ave edge cands len: 193.87603305785123
    edge recall, edge-level: 0.06942148760330578 42 out of 605

    + target 1 hop
    w/ direct parents/children, r@100 (cand-lvl cap at 100 & edge-lvl cap at 200)

    num_gold_edges: 1637
    ave tgt cands len: 101.0
    parent recall: 0.36363636363636365 220 out of 605
    child recall: 0.9669421487603306 585 out of 605
    edge recall: 0.3421487603305785 207 out of 605
    ave edge cands len: 200.0
    edge recall, edge-level: 0.256198347107438 155 out of 605

CPP ( embedding search + ranking)
    + target 1 hop
    w/ direct parents/children, r@50 (cand-lvl cap at 300 & edge-lvl cap at 300)

    num_gold_edges: 2131
    ave tgt cands len: 173.551
    parent recall: 0.501 501 out of 1000
    child recall: 0.974 974 out of 1000
    edge recall: 0.501 501 out of 1000
    ave edge cands len: 259.39
    edge recall, edge-level: 0.304 304 out of 1000

Disease ( idf search + embedding ranking)
    + target 1 hop
    w/ direct parents/children, r@10 (cand-lvl & edge-lvl cap at 50)    
    + edge ranking by embeddings (bert-tiny)    
    num_gold_edges: 1637
    ave tgt cands len: 30.254545454545454
    parent recall: 0.2892561983471074 175 out of 605
    child recall: 0.8743801652892562 529 out of 605
    edge recall: 0.28760330578512394 174 out of 605
    ave edge cands len: 34.22479338842975
    edge recall, edge-level: 0.06611570247933884 40 out of 605

    + edge ranking by embeddings (sapbert)    
    num_gold_edges: 1637
    ave tgt cands len: 30.254545454545454
    parent recall: 0.2892561983471074 175 out of 605
    child recall: 0.8743801652892562 529 out of 605
    edge recall: 0.28760330578512394 174 out of 605
    ave edge cands len: 34.22479338842975
    edge recall, edge-level: 0.06115702479338843 37 out of 605
'''    