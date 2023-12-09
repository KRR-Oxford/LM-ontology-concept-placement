# MM-S14-Disease, using mention contexts (true), top 10 edges from 5 edge seeds, batch size as 16
./Edge-Bi-enc+prompt-generation.sh Disease true 10 5 16 > results_log_Disease_edge_biencoder_top10_bs16_final.txt

# MM-S14-Disease, using mention contexts (true), top 50 edges from 25 edge seeds, batch size as 16
./Edge-Bi-enc+prompt-generation.sh Disease true 50 25 16 > results_log_Disease_edge_biencoder_top50_bs16_final.txt

# MM-S14-Disease, using mention only (false), top 50 edges from 25 edge seeds, batch size as 16
./Edge-Bi-enc+prompt-generation.sh Disease false 50 25 16 > results_log_Disease_edge_biencoder_top50_bs16_final_w_o_ctx.txt

# MM-S14-CPP, using using mention contexts (true), top 10 edges from 5 edge seeds, batch size as 16
./Edge-Bi-enc+prompt-generation.sh CPP true 10 5 16 > results_log_CPP_edge_biencoder_top10_bs16_final.txt

# MM-S14-CPP, using using mention contexts (true), top 50 edges from 25 edge seeds, batch size as 16
./Edge-Bi-enc+prompt-generation.sh CPP true 50 25 16 > results_log_CPP_edge_biencoder_top50_bs16_final.txt

# MM-S14-CPP, using mention only (false), top 50 edges from 25 edge seeds, batch size as 16
./Edge-Bi-enc+prompt-generation.sh CPP false 50 25 16 > results_log_CPP_edge_biencoder_top50_w_o_ctx_bs16_final.txt