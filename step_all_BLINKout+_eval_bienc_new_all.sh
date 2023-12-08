./step_all_BLINKout+_eval_bienc_new.sh Disease true 10 5 16 > results_log_Disease_edge_biencoder_top10_bs16_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_new.sh Disease true 50 25 16 > results_log_Disease_edge_biencoder_top50_bs16_final_w_o_leaf_score_rerun.txt

./step_all_BLINKout+_eval_bienc_new.sh Disease false 50 25 16 > results_log_Disease_edge_biencoder_top50_bs16_final_w_o_leaf_score_w_o_ctx.txt

#./step_all_BLINKout+_eval_bienc_new.sh Disease true 50 25 128 > results_log_Disease_edge_biencoder_top50_bs128.txt

# ./step_all_BLINKout+_eval_bienc_new_w_o_ctx.sh 20 10 > results_log_Disease_edge_biencoder_top20_w_o_ctx_new.txt
# ./step_all_BLINKout+_eval_bienc_new.sh 20 10 > results_log_Disease_edge_biencoder_top20.txt

# ./step_all_BLINKout+_eval_bienc_new.sh Disease false 50 25 16 > results_log_Disease_edge_biencoder_top50_w_o_ctx_bs16_final.txt

# ./step_all_BLINKout+_eval_bienc_new.sh Disease false 300 200 16 > results_log_Disease_edge_biencoder_top300_w_o_ctx_bs16_final.txt
./step_all_BLINKout+_eval_bienc_new.sh Disease true 300 200 16 > results_log_Disease_edge_biencoder_top300_bs16_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_new.sh CPP true 10 5 16 > results_log_CPP_edge_biencoder_top10_bs16_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_new.sh CPP true 50 25 16 > results_log_CPP_edge_biencoder_top50_bs16_final_w_o_leaf_score.txt
# ./step_all_BLINKout+_eval_bienc_new.sh CPP false 50 25 16 > results_log_CPP_edge_biencoder_top50_w_o_ctx_bs16_final.txt
# ./step_all_BLINKout+_eval_bienc_new.sh CPP false 300 200 16 > results_log_CPP_edge_biencoder_top300_w_o_ctx_bs16_final.txt
./step_all_BLINKout+_eval_bienc_new.sh CPP true 300 200 16 > results_log_CPP_edge_biencoder_top300_bs16_final_w_o_leaf_score.txt

# ablation study
./step_all_BLINKout+_eval_bienc_new.sh Disease true 50 25 16 > results_log_Disease_edge_biencoder_top50_bs16_final_w_o_leaf_score_w_o_enrich.txt 

./step_all_BLINKout+_eval_bienc_new.sh CPP true 50 25 16 > results_log_CPP_edge_biencoder_top50_bs16_final_w_o_leaf_score_w_o_enrich.txt