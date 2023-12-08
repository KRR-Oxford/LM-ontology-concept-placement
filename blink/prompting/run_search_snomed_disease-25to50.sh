# top-50
## embedding-based
# top-50 from 25

# previous results - add --use_leaf_edge_score_always to all commands below

#python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split train > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-train.txt --output_preds # the case of train for cross-encoder training (*sample*)

#python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split valid > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-valid.txt --output_preds # the case of valid for cross-encoder training (*sample*)

#python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split test-in-KB > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-test-in-KB.txt # the case of in-KB

#python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split test-NIL > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL.txt # Getting the results in test-NIL for qualitative analysis

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split valid-NIL > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt #-run2.txt --output_preds

#python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split valid-NIL --LEAF_EDGE_SCORE 0 > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-leaf-score0.txt # the case of leaf-edge-score as 0

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split test-NIL > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

#python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split test-NIL --LEAF_EDGE_SCORE 0 > log-results-25to50-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-leaf-score0.txt # the case of leaf-edge-score as 0

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-25to50-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-25to50-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

# top-1 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split valid-NIL > log-results-50to1-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split test-NIL > log-results-50to1-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to1-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to1-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

# top-5 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split valid-NIL > log-results-50to5-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split test-NIL > log-results-50to5-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to5-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to5-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

# top-10 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split valid-NIL > log-results-50to10-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split test-NIL > log-results-50to10-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to10-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to10-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

## idf-based
# top-50 from 25
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-results-25to50-idf-ranking-enrich-valid-NIL-new.txt #-run2.txt --output_preds

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-results-25to50-idf-ranking-enrich-test-NIL-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-25to50-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-25to50-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# top-1 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-results-50to1-idf-ranking-enrich-valid-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-results-50to1-idf-ranking-enrich-test-NIL-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to1-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to1-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# top-5 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-results-50to5-idf-ranking-enrich-valid-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-results-50to5-idf-ranking-enrich-test-NIL-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to5-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to5-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# top-10 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-results-50to10-idf-ranking-enrich-valid-NIL-new.txt

python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-results-50to10-idf-ranking-enrich-test-NIL-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to10-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-results-50to10-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# top-20 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 20 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-results-50to20-idf-ranking-enrich-valid-NIL-new.txt

# top-30 from top-50 (seed 25)
python prompting_embedding_edges.py --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 30 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-results-50to30-idf-ranking-enrich-valid-NIL-new.txt
