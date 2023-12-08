# prompting LLAMA 2
import os
import json

import torch
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import AutoPeftModelForCausalLM

from Llama_2_finetuning import prompt_concat_ctx_and_men, prompt_id_removal, prompt_correction

import pandas as pd
# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)
from tqdm import tqdm

# run shell script: export CUDA_VISIBLE_DEVICES=0

def prompting_LLAMA_2(prompt,model,tokenizer,max_new_tokens=100):
    with torch.inference_mode():
        input_ids = tokenizer(prompt, return_tensors="pt",truncation=True).input_ids.cuda()
        outputs = model.generate(input_ids=input_ids,max_new_tokens=max_new_tokens,do_sample=True,top_p=0.9,temperature=0.9)
        outputs = tokenizer.decode(outputs[0])  
        #outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]      
    return outputs

def format_prompt(prompt,cot=False):
    '''
    format the prompt to keep it the same as in the fine-tuning in Llama 2, see format_instruction() in Llama_2_fine_tuning.py

    cot (chain of thoughts) format, gold answer is needed to construct the explanation below - 
    ### Explanation:
    From the parents in the options above, xxx, xxx, xxx, the correct parents of the mention xxx include xxx, xxx, xxx. Thus the options are narrowed down to x, x, and x. From the children in the narrowed options, the correct children of the mention xxx include xxx, xxx, xxx. Thus the final answers are x, x, x.
    '''

    if not cot:
        template = f"""### Input:
{prompt}

### Response:
"""
    else:
        template = f"""### Input:
{prompt}

### Explanation:
"""
        
    return template

snomed_subset='Disease' # 'Disease' or 'CPP'
top_k_value = 50 #100 #10 #20 #50
model_size = "7b" # 7b/13b
concat_ctx_ment = True
use_cot_in_model = True # False for original model (see reset in the 3rd line below), False/True for fine-tuned model.
remove_id_in_model = True
model_name = "./llama-%s-int4-%s-insertion-top%d%s%s%s" % (model_size,snomed_subset,top_k_value,'-concat-men' if concat_ctx_ment else '', '-cot' if use_cot_in_model else '', '-w-o-id' if remove_id_in_model else '')
#model_name = "meta-llama/Llama-2-%s-hf" % model_size; use_cot_in_model=False 
model_name_ori = "meta-llama/Llama-2-%s-hf" % model_size

tokenizer = AutoTokenizer.from_pretrained(model_name_ori)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if model_name[:2] == './':
    # model is a self-trained peft model
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    ) 
else:
    # model is the original huggingface model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to("cuda")

model_name_clean = model_name.replace('/','-').replace('.','')

top_k_base_value = top_k_value #100 #20 # 500
top_k_value = top_k_value #100 #20 # 50
filter_by_degree = False #Trues
is_NIL = True

debug = False
data_debug_limit = 10

use_cot_in_prompt = use_cot_in_model
remove_id_in_prompt = remove_id_in_model
if use_cot_in_prompt:
    max_new_tokens = 1000 # 500 
else:
    max_new_tokens = 100

data_splits = ['valid','test']
#data_splits = ['test']
for data_split_mark in data_splits:
    prompts_fn = "../../models/biencoder/mm+%s2017AA-tl-sapbert-NIL-bs16/top%d_candidates/%s%s-top%d-preds%s-prompts-by-edges.csv" % (snomed_subset,top_k_base_value,data_split_mark,'-NIL' if is_NIL else '',top_k_value, "-degree-1" if filter_by_degree else "")

    saving_step=100

    prompts_df = pd.read_csv(prompts_fn,index_col=0)
    if debug:
        prompts_df = prompts_df[:data_debug_limit]
    print('prompts_df:',prompts_df.head())

    prompts_df[model_name_clean] = ""
    for i, row in tqdm(prompts_df.iterrows(),total=len(prompts_df)):
        if i != 0 and i % saving_step == 0:
            prompts_df.to_csv(prompts_fn[:len(prompts_fn)-4] + '-%s-step%d.csv' % (model_name_clean,i),index=True)
        if str(row["answer"]) == "-1":
            continue
        prompt = row["prompt"]        
        prompt = prompt_correction(prompt) # correct prompt
        if concat_ctx_ment:
            prompt = prompt_concat_ctx_and_men(prompt) # concatenate context and mention in the prompt (i.e. use the non-separated paragraph with mention marked as *mention*.)
        if remove_id_in_prompt:
            prompt = prompt_id_removal(prompt) # remove ids (and parentheses) in prompt
        prompt_formatted = format_prompt(prompt,cot=use_cot_in_prompt) # format prompt for Llama 2 (same as in fine-tuning)
        # prompt now
        answer = prompting_LLAMA_2(prompt_formatted,model=model,tokenizer=tokenizer,max_new_tokens=max_new_tokens)
        #answer = answer[5:][:-4].strip()
        prompts_df.at[i,"prompt"] = prompt_formatted
        prompts_df.at[i,model_name_clean] = answer
        print(answer)

    filename_to_save = prompts_fn[:len(prompts_fn)-4] + '%s%s-run2.csv' % (model_name_clean,'-w-o-id' if remove_id_in_prompt else '')
    prompts_df.to_csv(filename_to_save,index=True)
    print('file saved to:', filename_to_save)
