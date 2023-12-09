# MM-S14-Disease, using mention contexts (true), top 50 edges from 25 edge seeds, batch size as 16, using the first 200000 mention-edge pairs for training
./Edge-Bi-enc+Cross-enc.sh Disease true 50 25 16 true 200000 > results_log_Disease_edge_biencoder_top50_200k_final.txt

# MM-S14-Disease, using mention only (false), top 50 edges from 25 edge seeds, batch size as 16, using the first 200000 mention-edge pairs for training
./Edge-Bi-enc+Cross-enc.sh Disease false 50 25 16 true 200000 > results_log_Disease_edge_biencoder_top50_200k_final_w_o_ctx.txt

# MM-S14-Disease, using mention contexts (true), top 10 edges from 5 edge seeds, batch size as 16, using the first 200000 mention-edge pairs for training
./Edge-Bi-enc+Cross-enc.sh CPP true 10 5 16 true 200000 > results_log_Disease_edge_biencoder_top10_200k_final.txt
