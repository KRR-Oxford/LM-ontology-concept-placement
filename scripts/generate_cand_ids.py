# generate_cand_ids based on https://github.com/facebookresearch/BLINK/issues/65 and https://github.com/facebookresearch/BLINK/issues/106#issuecomment-1014507351
# see generate_candidates_blink.py for the guidance to run this script

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    #process_mention_data,
    #get_candidate_representation,
    get_edge_candidate_representation,
)
import json
import sys
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_model_config', type=str, required=True, help='filepath to saved model config')
parser.add_argument('--path_to_model', type=str, required=True, help='filepath to saved model')
parser.add_argument('--bert_model', type=str, required=True, help='the type of the bert model, as specified in the huggingface model hub')
parser.add_argument('--lowercase', action="store_true", help="Whether to lower case the input text. True for uncased models, False for cased models.") # this is a paramter of AutoTokenizer.from_pretrained(), see BiEncoderRanker.__init__() in biencoder.py)
parser.add_argument('--saved_cand_ids_path', type=str, required=True, help='filepath to save the IDs of tokens in candidate entities parsed from the tokeniser')
parser.add_argument('--entity_list_json_file_path', type=str, help='filepath to the entity list as a json/jsonl file', default=None)
parser.add_argument('--use_NIL_tag', action="store_true",help="Whether to use NIL tag, an unknown token in word piece tokenizer in BERT, for NIL entity reprentation, instead of using a string of 'NIL' to represent NIL entities")
parser.add_argument('--use_NIL_desc', action="store_true",help="Whether to add NIL description, instead of using an empty string")
parser.add_argument('--use_NIL_desc_tag', action="store_true",help="Whether to use special token of NIL in the description, instead of using \"NIL\"")
parser.add_argument('--use_synonyms', action="store_true",help="Whether to use synonyms for candidate representation")
parser.add_argument('--max_cand_length',type=str,help="max candidate length",default='128')

args = parser.parse_args()

biencoder_config = args.path_to_model_config #"models/biencoder_wiki_large.json"
biencoder_model_path = args.path_to_model #"models/biencoder_wiki_large.bin"
biencoder_model = args.bert_model
lowercase = args.lowercase
saved_cand_ids_path = args.saved_cand_ids_path #"preprocessing/saved_cand_ids_umls2012AB.pt" or 'models/saved_cand_ids_entity.pt'
entity_list_json_file_path = args.entity_list_json_file_path # #'preprocessing/UMLS2012AB.jsonl' or 'models/entity.jsonl'
use_NIL_tag = args.use_NIL_tag
use_NIL_desc = args.use_NIL_desc
use_NIL_desc_tag = args.use_NIL_desc_tag
use_synonyms = args.use_synonyms

# Load biencoder model and biencoder params just like in main_dense.py
with open(biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["path_to_model"] = biencoder_model_path
    biencoder_params["bert_model"] = biencoder_model # this updates the bert_model used (overwrites the one in the path_to_model_config)
    biencoder_params["lowercase"] = lowercase
    biencoder_params["max_cand_length"] = int(args.max_cand_length)
biencoder = load_biencoder(biencoder_params) # here it loads the biencoder

# Read the first 10 or all entities from entity catalogue, e.g. entity.jsonl
entities = []
#count = 10
with open(entity_list_json_file_path, encoding="utf-8-sig") as f:
    for i, line in tqdm(enumerate(f)):
        entity = json.loads(line)
        entities.append(entity)
        #if i == count-1:
        #    break

# Get token_ids corresponding to candidate title and description
tokenizer = biencoder.tokenizer # see biencoder.BiEncoderRanker.__init__()
#max_context_length, max_cand_length =  biencoder_params["max_context_length"], biencoder_params["max_cand_length"]
#max_seq_length = max_cand_length
max_seq_length=biencoder_params["max_cand_length"]
ids = []

# it can take around 6 hours to process all 5.9M Wikipedia entities.
for entity in tqdm(entities):
    #candidate_desc = entity['text']
    #candidate_title = entity['title']
    parent_title = entity['parent']
    child_title = entity['child']
    # if 'synonyms' in entity:
    #     synonyms = entity['synonyms']
    # else:
    #     synonyms = None
    cand_tokens = get_edge_candidate_representation(
        #candidate_desc, 
        #synonyms,
        tokenizer, 
        max_seq_length, 
        #candidate_title=candidate_title,
        parent_candidate_title=parent_title,
        child_candidate_title=child_title,
        # use_NIL_tag=use_NIL_tag,
        # use_NIL_desc=use_NIL_desc,
        # use_NIL_desc_tag=use_NIL_desc_tag,
        # use_synonyms=use_synonyms,
    )

    token_ids = cand_tokens["ids"]
    ids.append(token_ids)

ids = torch.tensor(ids)
print(ids)
torch.save(ids, saved_cand_ids_path)