# an example of running Llama-2 fine-tuning and prompting.

# first, generate prompts and instructions
eval_set=valid-NIL,test-NIL
./Edge-Bi-enc+prompt-generation.sh Disease true 10 5 16 > results_log_Disease_edge_biencoder_top10_bs16_final.txt 
# setting eval_set=train in the script above for instructions, setting eval_set=valid-NIL,test-NIL in the script above for prompting only

# go to the prompting folder
cd blink/prompting/
source activate onto38 # the conda environment for LLMs
# change the top-k setting, model and prompt format settings before running 
python Llama_2_finetuning.py
python Llama_2_prompting.py
python Llama_2_results_eval.py