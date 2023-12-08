# prompting FLAN-T5 
import os
import json

import torch
import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from Llama_2_finetuning import prompt_concat_ctx_and_men, prompt_id_removal, prompt_correction

# run shell script: export CUDA_VISIBLE_DEVICES=0

def prompting_FLAN_T5(prompt,model,tokenizer,max_new_tokens=10):
    with torch.inference_mode():
        input_ids = tokenizer(prompt, return_tensors="pt",).input_ids.to("cuda")
        outputs = model.generate(input_ids,max_new_tokens=max_new_tokens)
        outputs = tokenizer.decode(outputs[0])        
    return outputs


#model_name = "./flan-t5-small";checkpoint_name = "checkpoint-48"
#model_name = "./flan-t5-large";checkpoint_name = "checkpoint-1494"
#model_name = "google/flan-t5-xxl";checkpoint_name=""
model_name = "google/flan-t5-xl";checkpoint_name=""
#model_name = "google/flan-t5-large";checkpoint_name=""
#model_name = "google/flan-t5-small";checkpoint_name=""
tokenizer = T5Tokenizer.from_pretrained(model_name)
if checkpoint_name != "":
    model_name = os.path.join(model_name,checkpoint_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto",)

model_name_clean = model_name.replace('/','-').replace('.','')

snomed_subset='CPP'
top_k_base_value = 10 #50 # 20 # 500
top_k_value = 10 #50 # 20 # 50
concat_ctx_ment = True
remove_id_in_prompt = True
filter_by_degree = False #Trues
is_NIL = True

data_splits = ['valid','test']
for data_split_mark in data_splits:
    prompts_fn = "../../models/biencoder/mm+%s2017AA-tl-sapbert-NIL-bs16/top%d_candidates/%s%s-top%d-preds%s-prompts-by-edges.csv" % (snomed_subset,top_k_base_value,data_split_mark,'-NIL' if is_NIL else '',top_k_value, "-degree-1" if filter_by_degree else "")

    import pandas as pd
    # avoid ... in showing long sequence
    pd.set_option("display.max_colwidth", 10000)
    from tqdm import tqdm

    saving_step=100

    prompts_df = pd.read_csv(prompts_fn,index_col=0)
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
        # prompt now
        answer = prompting_FLAN_T5(prompt,model=model,tokenizer=tokenizer)
        #answer = answer[5:][:-4].strip()
        prompts_df.at[i,"prompt"] = prompt        
        prompts_df.at[i,model_name_clean] = answer
        print(answer)

    filename_to_save = prompts_fn[:len(prompts_fn)-4] + '%s.csv' % model_name_clean
    prompts_df.to_csv(filename_to_save,index=True)
    print('file saved to:', filename_to_save)
