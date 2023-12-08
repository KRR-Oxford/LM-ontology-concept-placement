# aggregating .jsonl in st21pv_syn_attr-all-complexEdge-edges-final same as the predictions of Edge-Cross-encoder

import os
import json
from tqdm import tqdm

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

snomed_subset = "Disease"

input_path = "../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl" % snomed_subset

dict_ment_info = {}
with open(input_path,encoding='utf-8-sig') as f_content:
    doc = f_content.readlines()
for ind, ment_edge_info_json in tqdm(enumerate(doc)):
    mention_edge_info = json.loads(ment_edge_info_json)  
    context_left = mention_edge_info["context_left"]
    mention = mention_edge_info["mention"]
    context_right = mention_edge_info["context_right"]
    label_concept_ori = mention_edge_info["label_concept_ori"]

    dict_ment_info[context_left + '*' +mention + '*' + context_right + label_concept_ori] = 1

print(len(dict_ment_info))
#print(dict_ment_info)
output_path = input_path[:-6] + '-aggregated.txt'
output_to_file(output_path,'\n'.join(list(dict_ment_info.keys())))