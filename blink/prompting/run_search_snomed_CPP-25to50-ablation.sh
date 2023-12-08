# top-50
## embedding-based
# top-50 from 25

python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --data_split valid-NIL > log-CPP-results-25to50-fixed-embedding-ranking-ori-target-sapbert-valid-NIL-new.txt

python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --data_split test-NIL > log-CPP-results-25to50-fixed-embedding-ranking-ori-target-sapbert-test-NIL-new.txt

# # top-1 from top-50 (seed 25)
# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split valid-NIL > log-CPP-results-50to1-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt

# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split test-NIL > log-CPP-results-50to1-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to1-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to1-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

# # top-5 from top-50 (seed 25)
# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split valid-NIL > log-CPP-results-50to5-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt

# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split test-NIL > log-CPP-results-50to5-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to5-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to5-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

# # top-10 from top-50 (seed 25)
# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split valid-NIL > log-CPP-results-50to10-fixed-embedding-ranking-enrich-ori-target-sapbert-valid-NIL-new.txt

# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split test-NIL > log-CPP-results-50to10-fixed-embedding-ranking-enrich-ori-target-sapbert-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to10-fixed-embedding-ranking-enrich-ori-target+context-sapbert-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to10-fixed-embedding-ranking-enrich-ori-target+context-sapbert-test-NIL-pure-ctx-new.txt

# ## idf-based
# # top-50 from 25
python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-CPP-results-25to50-idf-ranking-valid-NIL-new.txt

python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-CPP-results-25to50-idf-ranking-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-25to50-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 50 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-25to50-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# # top-1 from top-50 (seed 25)
# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-CPP-results-50to1-idf-ranking-enrich-valid-NIL-new.txt

# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-CPP-results-50to1-idf-ranking-enrich-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to1-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 1 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to1-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# # top-5 from top-50 (seed 25)
# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-CPP-results-50to5-idf-ranking-enrich-valid-NIL-new.txt

# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-CPP-results-50to5-idf-ranking-enrich-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to5-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 5 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to5-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt

# # top-10 from top-50 (seed 25)
# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL > log-CPP-results-50to10-idf-ranking-enrich-valid-NIL-new.txt

# python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL > log-CPP-results-50to10-idf-ranking-enrich-test-NIL-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split valid-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to10-idf-ranking-enrich+context-valid-NIL-pure-ctx-new.txt

# # python prompting_embedding_edges.py --snomed_subset CPP --edge_ranking_by_score --enrich_cands --pool_size 50 --pool_size_seed 25 --pool_size_edge_lvl 10 --enrich_edges --use_idf_score_for_cand --use_idf_score_for_edges --data_split test-NIL --use_context --percent_mention_w_ctx 1.0 > log-CPP-results-50to10-idf-ranking-enrich+context-test-NIL-pure-ctx-new.txt