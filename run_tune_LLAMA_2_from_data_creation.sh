# an example of running Llama-2 fine-tuning and prompting.

# first, generate prompts and instructions
./Edge-Bi-enc+prompt-generation.sh Disease true 10 5 16 # > results_log_Disease_edge_biencoder_top10_bs16_final.txt 
# need to run the above twice if instruction tuning:
#   1. first, by setting eval_set=train in the script above for instruction prompts (for instruction tuning), 
#   2. second, by setting eval_set=valid-NIL,test-NIL in the script above for the evaluation prompting only (for evaluation).
# or by setting eval_set=train,valid-NIL,test-NIL to generate both instruction prompts and evaluation prompts.

# go to the prompting folder
cd blink/prompting/
source activate ontollm38 # the conda environment for LLMs
# change the top-k setting, model and prompt format settings before running 
python Llama_2_finetuning.py
python Llama_2_prompting.py
python Llama_2_results_eval.py
