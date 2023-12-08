# ./step_all_BLINKout+_eval_bienc_cross.sh false 20 10 16 > results_log_Disease_edge_biencoder_top20_w_o_ctx_new.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh true 20 10 16 > results_log_Disease_edge_biencoder_top20_new.txt


# ./step_all_BLINKout+_eval_bienc_cross_gpu0.sh Disease true 50 25 16 100 > results_log_Disease_edge_biencoder_top50_new_100_run1.txt
# ./step_all_BLINKout+_eval_bienc_cross_gpu0.sh Disease true 50 25 16 100 > results_log_Disease_edge_biencoder_top50_new_100_run2.txt

# ./step_all_BLINKout+_eval_bienc_cross.sh Disease false 50 25 16 1000000 > results_log_Disease_edge_biencoder_top50_w_o_ctx_new_1000k.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease false 50 25 16 10000000 > results_log_Disease_edge_biencoder_top50_w_o_ctx_new_10000k.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 > results_log_Disease_edge_biencoder_top50_new.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 100 > results_log_Disease_edge_biencoder_top50_new_100_no_rand.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 1000 > results_log_Disease_edge_biencoder_top50_new_1k_no_rand.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 10000 > results_log_Disease_edge_biencoder_top50_new_10k_no_rand.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 100000 > results_log_Disease_edge_biencoder_top50_new_100k_no_rand.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 200000 > results_log_Disease_edge_biencoder_top50_new_200k.txt
# # ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 400000 > results_log_Disease_edge_biencoder_top50_new_400k.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 300 200 16 10000 > results_log_Disease_edge_biencoder_top300_new_100k.txt
#./step_all_BLINKout+_eval_bienc_cross.sh Disease true 300 200 16 100000 > results_log_Disease_edge_biencoder_top300_new_100k_cont.txt
#./step_all_BLINKout+_eval_bienc_cross.sh Disease true 300 200 16 200000 > results_log_Disease_edge_biencoder_top300_new_200k.txt

# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 true 100000 > results_log_Disease_edge_biencoder_top50_100k_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 true 200000 > results_log_Disease_edge_biencoder_top50_200k_final_w_o_leaf_score_rerun.txt
./step_all_BLINKout+_eval_bienc_cross.sh Disease false 50 25 16 true 200000 > results_log_Disease_edge_biencoder_top50_200k_final_w_o_leaf_score_w_o_ctx.txt
./step_all_BLINKout+_eval_bienc_cross_gpu0.sh Disease true 300 200 16 true 200000 > results_log_Disease_edge_biencoder_top300_200k_final_w_o_leaf_score.txt
# ./step_all_BLINKout+_eval_bienc_cross_gpu0.sh Disease false 300 200 16 true 200000 > results_log_Disease_edge_biencoder_top300_200k_final_w_o_ctx.txt

# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 true 1000 > results_log_Disease_edge_biencoder_top50_1k_final.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 true 10000 > results_log_Disease_edge_biencoder_top50_10k_final.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 50 25 16 true 50000 > results_log_Disease_edge_biencoder_top50_50k_final.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease false 300 200 16 > results_log_Disease_edge_biencoder_top300_w_o_ctx_new.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh Disease true 300 200 16 > results_log_Disease_edge_biencoder_top300_new.txt

# ./step_all_BLINKout+_eval_bienc_cross.sh CPP true 20 10 16 > results_log_CPP_edge_biencoder_top20.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh CPP false 20 10 16 > results_log_CPP_edge_biencoder_top20_w_o_ctx_new.txt

# ./step_all_BLINKout+_eval_bienc_cross.sh CPP true 50 25 16 true 200000 > results_log_CPP_edge_biencoder_top50_200k_final_w_o_leaf_score.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh CPP false 50 25 16 > results_log_CPP_edge_biencoder_top50_w_o_ctx_new.txt

# ./step_all_BLINKout+_eval_bienc_cross.sh CPP true 300 200 16 true 200000 > results_log_CPP_edge_biencoder_top300_200k_final_w_o_leaf_score.txt
# ./step_all_BLINKout+_eval_bienc_cross.sh CPP false 300 200 16 > results_log_CPP_edge_biencoder_top300_w_o_ctx_new.txt

./step_all_BLINKout+_eval_bienc_cross.sh CPP true 100 50 16 true 200000 > results_log_Disease_edge_biencoder_top100_200k_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_cross_gpu0.sh CPP true 10 5 16 true 200000 > results_log_Disease_edge_biencoder_top10_200k_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_cross_gpu0.sh CPP true 20 10 16 true 200000 > results_log_Disease_edge_biencoder_top20_200k_final_w_o_leaf_score.txt

./step_all_BLINKout+_eval_bienc_cross_gpu0.sh Disease true 100 50 16 true 200000 > results_log_Disease_edge_biencoder_top100_200k_final_w_o_leaf_score.txt
