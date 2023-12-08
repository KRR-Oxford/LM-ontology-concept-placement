export CUDA_VISIBLE_DEVICES=1
./step_all_BLINKout+_eval_bienc_new.sh
conda deavtivate

cd blink/prompting/
source activate onto38
python Llama_2_finetuning.py
python Llama_2_prompting.py
python Llama_2_results_eval.py