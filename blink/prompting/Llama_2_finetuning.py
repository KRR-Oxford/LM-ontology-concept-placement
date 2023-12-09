# fine-tune LLAMA 2 with prompts for edge candidate re-ranking (top-k -> top-1).

import pandas as pd
import re
from datasets import Dataset
# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)
from tqdm import tqdm
import os
from subprocess import Popen, STDOUT

CONCAT_CTX_MENT = True
REMOVE_ID_IN_PROMPT = True
USE_COT = False

def prompt_concat_ctx_and_men(prompt):
    '''
    concatenate the ctx left, mention, and ctx right together and revise the prompt
    '''
    # revise task description
    if '*' in prompt: # remove existing * first. 
        print('asterix * in prompt:', prompt)
        prompt = prompt.replace('*','')
    prompt = prompt.replace("mention based on the provided context (context_left and context_right)?", "mention (marked with *) based on the context?")
    # revise mention
    prompt = prompt.replace("context_left:","mention in context:")
    prompt = prompt.replace("\n\nmention:\n","*")
    prompt = prompt.replace("\n\ncontext_right:\n","*")
    return prompt

def prompt_id_removal(prompt):
    '''
    remove the id number with parenthesis, e.g., "(55342001)", and also the "(SCTID_NULL)" in the prompts. 
    '''
    return re.sub(r'\s*\(\d+\)|\s*\(SCTID_NULL\)', '', prompt)

def prompt_correction(prompt):
    '''
    correct the prompt if there are errors, which were not fixed previously, e.g., in blink/biencoder/candidate_analysis.py
    '''
    return prompt.replace('separated by a colon', 'separated by commas')

def find_mention_in_text(ment_in_ctx):
    '''
    find the mention (the first occurrence of marked between two asterix, *mention*) in context
    '''
    # Define the regex pattern
    pattern = r'\*([^*]+)\*'  # Same pattern as before

    # Search for the first match in the input text
    match = re.search(pattern, ment_in_ctx)

    # Check if a match was found
    if match:
        # Extract the matched string from the capture group
        matched_string = match.group(1)
        return matched_string
    else:
        return ""
    
def prompt_processing(prompt,answer_str="",concat_ctx_and_ment=CONCAT_CTX_MENT):
    '''
    input: prompt string and answer string
    output: mention, list of all parent options, list of correct parent options, list of narrowed option numbers, list of narrowed child options, list of correct narrowed child options, gold option numbers
    '''
    if not concat_ctx_and_ment:
        mention = prompt[prompt.find("mention:")+len("mention:"):prompt.find("context_right:")]
    else:
        mention_w_ctx = mention = prompt[prompt.find("mention in context:")+len("mention in context:"):prompt.find("options:")]
        mention = find_mention_in_text(mention_w_ctx)
    mention = mention.strip()
    #print('mention:',mention)

    options_str = prompt[prompt.find("options:")+len("options:"):].strip()
    options = options_str.split('\n')
    gold_option_nums = answer_str.split(',')

    # extract all parents' and all correct parents' options
    dict_parents_to_num = {}
    gold_parents = []
    dict_num_to_children = {}
    for option in options:
        option_num = option[:option.find(".")]
        option = option[option.find(".")+1:]

        #print('option:',option)
        option_eles = option.split(' -> ')
        parent_op = option_eles[0]
        child_op = option_eles[1]
        dict_num_to_children[option_num] = child_op
        if not parent_op in dict_parents_to_num:
            dict_parents_to_num[parent_op] = [option_num]
        else:    
            # update dict_parents_to_num's option_nums
            option_nums = dict_parents_to_num[parent_op]
            if not option_num in option_nums:
                option_nums.append(option_num)
                dict_parents_to_num[parent_op] = option_nums                    
        if option_num in gold_option_nums:
            if not parent_op in gold_parents:
                gold_parents.append(parent_op)
    parent_options = list(dict_parents_to_num.keys())

    # get narrowed options by parents (if correctly predicted)
    gold_parent_op_nums = []
    for gold_parent in gold_parents:
        op_nums_ = dict_parents_to_num[gold_parent]
        gold_parent_op_nums.extend(op_nums_)

    # extract narrowed children, from the narrowed options, and
    # extract correct chilren from the narrowed children
    children_narrowed = []
    gold_children_narrowed = []
    for narrowed_op_num in gold_parent_op_nums:
        child_op = dict_num_to_children[narrowed_op_num]
        if not child_op in children_narrowed:
            children_narrowed.append(child_op)
        if narrowed_op_num in gold_option_nums:
            if not child_op in gold_children_narrowed:
                gold_children_narrowed.append(child_op)

    return mention, parent_options, gold_parents, gold_parent_op_nums, children_narrowed, gold_children_narrowed, gold_option_nums

def answer_ordering(answer_str=''):
    '''
    reorder the answer sequence ascendingly and output the string format (nums with comma in between)
    '''
    option_nums_str = answer_str.split(',')
    option_nums = [int(op_num_str) for op_num_str in option_nums_str]
    option_nums_sorted = sorted(option_nums) # by ascending order
    option_nums_sorted_str = [str(op_num) for op_num in option_nums_sorted]
    return ','.join(option_nums_sorted_str)

def format_instruction(sample,remove_id_in_prompt=REMOVE_ID_IN_PROMPT,cot=USE_COT,concat_ctx_ment=CONCAT_CTX_MENT):
    '''
    format the prompt to keep it the same as in the fine-tuning in Llama 2, see format_instruction() in Llama_2_fine_tuning.py

    cot (chain of thoughts) format, gold answer is needed to construct the explanation below - 
    ### Explanation:
    From the parents in the options above, xxx, xxx, xxx, the correct parents of the mention xxx include xxx, xxx, xxx. Thus the options are narrowed down to x, x, and x. From the children in the narrowed options, the correct children of the mention xxx include xxx, xxx, xxx. Thus the final answers are x, x, x.
    '''

    corrected_prompt_form = prompt_correction(sample['prompt'])
    if concat_ctx_ment:
        corrected_prompt_form = prompt_concat_ctx_and_men(corrected_prompt_form)
    if remove_id_in_prompt:
        corrected_prompt_form = prompt_id_removal(corrected_prompt_form)
    answer_sorted_form = answer_ordering(sample['answer'])
    
    if not cot:
        template = f"""### Input:
{corrected_prompt_form}

### Response:
{answer_sorted_form}
"""
    else:
        mention, parent_options, gold_parents, gold_parent_op_nums, children_narrowed, gold_children_narrowed, gold_option_nums = prompt_processing(prompt=corrected_prompt_form,answer_str=answer_sorted_form)
        
        explaination_str = "From the parents in the options above, including %s, the correct parents of the mention, %s, include %s. Thus the options are narrowed down to %s. From the children in the narrowed options, including %s, the correct children of the mention, %s, include %s. Thus, the final answers are %s." % (', '.join(parent_options), mention, ', '.join(gold_parents), ', '.join(gold_parent_op_nums), ', '.join(children_narrowed), mention, ', '.join(gold_children_narrowed), ', '.join(gold_option_nums))

        template = f"""### Input:
{corrected_prompt_form}

### Explanation:
{explaination_str}

### Response:
{answer_sorted_form}
"""
    print(template)
    return template

if __name__ == "__main__":
    cmd = "huggingface-cli login --token [YOUR HUGGINGFACE TOKEN]"
    p = Popen(cmd, shell=True, stderr=STDOUT)
    p.wait()

    if 0 != p.returncode:
        print('Command %s wrong!' % cmd)
    else:
        print('Command %s completed successfully!' % cmd)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # Hugging Face model id
    model_size = "7b" # "7b" # "13b"
    #model_id = "NousResearch/Llama-2-%s-hf" % model_size # non-gated
    model_id = "meta-llama/Llama-2-%s-hf" % model_size # gated
    #model_id = "axiong/PMC_LLaMA_13B" if model_size == "13b" else "chaoyi-wu/PMC_LLAMA_7B"

    # BitsAndBytesConfig int-4 config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map="auto")
    model.config.pretraining_tp = 1 

    from transformers import AutoTokenizer
    if "PMC_LLaMA" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, unk_token="<unk>")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM", 
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)

    # prepare dataset
    print('tokenizer.model_max_length:',tokenizer.model_max_length)
    #model_name_no_slash = model_name.replace('/','-')

    snomed_subset='CPP' # Disease, CPP
    is_NIL = False
    top_k_base_value = 10 #200 #10 #20 #500
    top_k_value = 10 #200 #10 #20 #50
    filter_by_degree = False
    #data_splits=['train','valid']
    data_splits=['train']

    debug = False; data_debug_limit = 100

    model_name = "llama-%s-int4-%s-insertion-top%d%s%s%s" % (model_size,snomed_subset,top_k_value,'-concat-men' if CONCAT_CTX_MENT else '', '-cot' if USE_COT else '','-w-o-id' if REMOVE_ID_IN_PROMPT else '')

    for data_split_mark in data_splits:
        prompts_fn = "../../models/biencoder/mm+%s2017AA-tl-sapbert-NIL-bs16/top%d_candidates/%s%s-top%d-preds%s-prompts-by-edges.csv" % (snomed_subset,top_k_base_value,data_split_mark,'-NIL' if is_NIL else '',top_k_value, "-degree-1" if filter_by_degree else "")

        saving_step=100

        prompts_df = pd.read_csv(prompts_fn,index_col=0)
        print('prompts_df:',prompts_df.head())

        # select a subset of columns (prompt and answer)
        prompts_df = prompts_df[["prompt","answer"]]

        # transform prompts_df before making it a huggingface dataset
        prompts_df = prompts_df[prompts_df["answer"] != "-1"]

        if debug:
            prompts_df = prompts_df[:data_debug_limit]

        for i, row in tqdm(prompts_df.iterrows(),total=len(prompts_df)):
            answer = row["answer"]
            # remove -1 in answer
            answer = answer.replace(',-1','').replace('-1,','')
            prompts_df.at[i, "answer"] = answer
            #print(i, answer)

        print('prompts_df:',prompts_df.head())
        print('prompts_df length:',len(prompts_df))

        prompts_answer_dataset = Dataset.from_pandas(prompts_df)#.train_test_split(test_size=0.2)

    # setting training arguments and training
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=False,
        fp16=True,
        tf32=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,  # disable tqdm since with packing values are in correct
    )

    model = get_peft_model(model, peft_config)

    from trl import SFTTrainer

    max_seq_length = 2048 # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=prompts_answer_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction, 
        args=args,
    )

    # train
    trainer.train() # there will not be a progress bar since tqdm is disabled

    # save model
    trainer.save_model()

    # merge and save the whole model
    # from peft import AutoPeftModelForCausalLM

    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     model_name,
    #     low_cpu_mem_usage=True,
    # ) 

    # # Merge LoRA and base model
    # merged_model = model.merge_and_unload()

    # # Save the merged model
    # merged_model.save_pretrained("merged_model",safe_serialization=True)
    # tokenizer.save_pretrained("merged_model")