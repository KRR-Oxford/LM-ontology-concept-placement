#step0: reconstruct SNOMED-CT before removing branches/chapters
# prior to this, follow ontology_script+.sh in ../ontologies 
python reconstruct_snomed_owl_subclases_from_axiom.py # needs to set the ontology version parameter
# then remove branches manually with protege

#step1: get all files for an UMLS onto ver for medmentions dataset - for insertion

python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --onto_ver 20140901-Disease
#python get_all_SNOMED_CT_entities.py --add_synonyms --synonym_as_entity --add_direct_hyps --onto_ver 20140901-Disease # with synonym as entity
python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --concept_type all --onto_ver 20140901-Disease
python get_all_SNOMED_CT_entities.py --add_synonyms --synonym_as_entity --add_direct_hyps --concept_type all --onto_ver 20140901-Disease # with synonym as entity

python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --onto_ver 20140901-CPP
#python get_all_SNOMED_CT_entities.py --add_synonyms --synonym_as_entity --add_direct_hyps --onto_ver 20140901-CPP # with synonym as entity
python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --concept_type all --onto_ver 20140901-CPP
python get_all_SNOMED_CT_entities.py --add_synonyms --synonym_as_entity --add_direct_hyps --concept_type all --onto_ver 20140901-CPP # with synonym as entity

# for sieve
python get_all_SNOMED_CT_entities.py -f Sieve --add_synonyms --add_direct_hyps --concept_type all --onto_ver 20140901-Disease 
python get_all_SNOMED_CT_entities.py -f Sieve --add_synonyms --add_direct_hyps --concept_type all --onto_ver 20140901-CPP 

# for newer ontology 20170301 (only for statistics)
# python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --onto_ver 20170301-Disease
# python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --concept_type all --onto_ver 20170301-Disease

python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --onto_ver 20170301-CPP
python get_all_SNOMED_CT_entities.py --add_synonyms --add_direct_hyps --concept_type all --onto_ver 20170301-CPP

python get_all_SNOMED_CT_edges.py --onto_ver 20140901-Disease --concept_type atomic # atomic
python get_all_SNOMED_CT_edges.py --onto_ver 20140901-Disease --concept_type all # all 
python get_all_SNOMED_CT_edges.py --onto_ver 20140901-CPP --concept_type atomic  # atomic
python get_all_SNOMED_CT_edges.py --onto_ver 20140901-CPP --concept_type all # all

python get_all_SNOMED_CT_attributes.py --add_synonyms --onto_ver 20140901-Disease
#python get_all_SNOMED_CT_attributes.py --add_synonyms --synonym_as_entity --onto_ver 20140901-Disease
python get_all_SNOMED_CT_attributes.py --add_synonyms --onto_ver 20140901-CPP

## for DO
python get_all_DO_entities.py --onto_ver 2012-1-4-disease --add_synonyms --add_direct_hyps
python get_all_DO_entities.py --onto_ver 2012-1-4-disease --add_synonyms --synonym_as_entity --add_direct_hyps # with synonym as entity
python get_all_DO_entities.py --onto_ver 2014-6-16-disease --add_synonyms --add_direct_hyps
python get_all_DO_entities.py --onto_ver 2014-6-16-disease --add_synonyms --synonym_as_entity --add_direct_hyps # with synonym as entity
python get_all_DO_entities.py --onto_ver 2017-11-28-disease --add_synonyms --add_direct_hyps
python get_all_DO_entities.py --onto_ver 2022-9-29-disease --add_synonyms --add_direct_hyps
python get_all_DO_entities.py --onto_ver 2023-9-28-disease --add_synonyms --add_direct_hyps

python get_all_DO_entities.py --onto_ver 2022-9-29-disease --add_synonyms --add_direct_hyps --concept_type all
python get_all_DO_entities.py --onto_ver 2022-9-29-disease --add_synonyms --synonym_as_entity --add_direct_hyps --concept_type all # with synonym as entity

python get_all_DO_entities.py --onto_ver 2023-9-28-disease --add_synonyms --add_direct_hyps --concept_type all
python get_all_DO_entities.py --onto_ver 2023-9-28-disease --add_synonyms --synonym_as_entity --add_direct_hyps --concept_type all # with synonym as entity

python get_all_DO_edges.py --DO_ver_older 2014-6-16-disease
python get_all_DO_edges.py --DO_ver_older 2012-1-4-disease
python get_all_DO_edges.py --DO_ver_older 2023-9-28-disease
python get_all_DO_edges.py --DO_ver_older 2012-1-4-disease --concept_type all
python get_all_DO_edges.py --DO_ver_older 2022-9-29-disease --concept_type all
python get_all_DO_edges.py --DO_ver_older 2023-9-28-disease --concept_type all

#step2 get UMLS to SNOMED-CT mapping and,
#step3 get mention-level data.
# mention-level data, for Disease: disorders (in clinical findings)
#python format_trans_medmentions2blink+new.py > log-mm-insertion-data-2014vs2017-syn-attr.txt
# python format_trans_medmentions2blink+new.py --add_synonyms_as_ents > log-mm-insertion-data-2014vs2017-syn-full.txt

# python format_trans_medmentions2blink+new.py --concept_type complex > log-mm-insertion-data-2014vs2017-syn-attr-complex.txt
# python format_trans_medmentions2blink+new.py --concept_type all > log-mm-insertion-data-2014vs2017-syn-attr-all.txt

python format_trans_medmentions2blink+new.py --concept_type all --allow_complex_edge > log-mm-insertion-data-2014vs2017-syn-attr-all-compEdge-compFilt.txt

python format_trans_medmentions2blink+new.py --add_synonyms_as_ents --concept_type all --allow_complex_edge > log-mm-insertion-data-2014vs2017-full-syn-attr-all-compEdge-compFilt.txt

# mention-level data, for CPP: clinical findings + procedures + phamarceutical
# python format_trans_medmentions2blink+new.py --snomed_subset CPP --concept_type complex > log-mm-insertion-data-2014vs2017-CPP-syn-attr-complex.txt

# python format_trans_medmentions2blink+new.py --data_setting full --snomed_subset CPP --concept_type complex > log-mm-full-insertion-data-2014vs2017-CPP-syn-attr-complex.txt

python format_trans_medmentions2blink+new.py --snomed_subset CPP --concept_type all --allow_complex_edge > log-mm-insertion-data-2014vs2017-CPP-syn-attr-all-compEdge-compFilt.txt

python format_trans_medmentions2blink+new.py --snomed_subset CPP --add_synonyms_as_ents --concept_type all --allow_complex_edge > log-mm-insertion-data-2014vs2017-CPP-full-syn-attr-all-compEdge-compFilt.txt


## for DO
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2017-11-28-disease --DO_ver_older 2014-6-16-disease > log-mm-insertion-data-2014vs2017-syn-attr-DO.txt
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2017-11-28-disease --DO_ver_older 2014-6-16-disease --add_synonyms_as_ents > log-mm-insertion-data-2014vs2017-syn-full-DO.txt

python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2014-6-16-disease > log-mm-insertion-data-2014vs2023-syn-attr-DO.txt
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2014-6-16-disease --add_synonyms_as_ents > log-mm-insertion-data-2014vs2023-syn-full-DO.txt

python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2017-11-28-disease --DO_ver_older 2012-1-4-disease > log-mm-insertion-data-2012vs2017-syn-attr-DO.txt
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2017-11-28-disease --DO_ver_older 2012-1-4-disease --add_synonyms_as_ents > log-mm-insertion-data-2012vs2017-syn-full-DO.txt

python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2012-1-4-disease > log-mm-insertion-data-2012vs2023-syn-attr-DO.txt
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2012-1-4-disease --add_synonyms_as_ents > log-mm-insertion-data-2012vs2023-syn-full-DO.txt

python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2022-9-29-disease --concept_type all --allow_complex_edge > log-mm-insertion-data-2022vs2023-syn-attr-all-compEdge-compFilt-DO.txt
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2022-9-29-disease --add_synonyms_as_ents --concept_type all --allow_complex_edge > log-mm-insertion-data-2022vs2023-syn-full-all-compEdge-compFilt-DO.txt

python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2023-9-28-disease --concept_type all --allow_complex_edge > log-mm-insertion-data-2023vs2023-syn-attr-all-compEdge-compFilt-DO.txt
python format_trans_medmentions2blink+new-DO.py --DO_ver_newer 2023-9-28-disease --DO_ver_older 2023-9-28-disease --add_synonyms_as_ents --concept_type all --allow_complex_edge > log-mm-insertion-data-2023vs2023-syn-full-all-compEdge-compFilt-DO.txt

#sieve-based data creation
python format_trans_medmentions2sieve+new.py --concept_type all --allow_complex_edge
python format_trans_medmentions2sieve+new.py --snomed_subset CPP --concept_type all --allow_complex_edge

#step4 get edge-level data from mention-level data
# edge-level data: Disease
python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset Disease --concept_type all --allow_complex_edge > log_mm_data_for_edge_insertion-2014vs2017_Disease-all.txt # all
#python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset Disease --concept_type atomic  > log_mm_data_for_edge_insertion-2014vs2017_Disease.txt # atomic
#python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset Disease --concept_type complex --allow_complex_edge # complex
# for sieve data update
python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset Disease --concept_type all --allow_complex_edge --update_sieve_data > log_mm_data_for_edge_insertion-2014vs2017_Disease-all-sieve.txt
# get complex edge test set
# grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl

# edge-level data: CPP
python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset CPP --concept_type all --allow_complex_edge > log_mm_data_for_edge_insertion-2014vs2017_CPP-all.txt # all
#python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset CPP --concept_type atomic > log_mm_data_for_edge_insertion-2014vs2017_CPP.txt # atomic
# for sieve data update
python format_mm_data_for_edge_insertion.py --onto_ver 20140901 --snomed_subset CPP --concept_type all --allow_complex_edge --update_sieve_data > log_mm_data_for_edge_insertion-2014vs2017_CPP-all-sieve.txt
## for DO
python format_mm_data_for_edge_insertion-DO.py --DO_ver_older 2014-6-16-disease > log_mm_data_for_edge_insertion-2014vs2017-DO.txt

python format_mm_data_for_edge_insertion-DO.py --DO_ver_older 2014-6-16-disease > log_mm_data_for_edge_insertion-2014vs2023-DO.txt

python format_mm_data_for_edge_insertion-DO.py --DO_ver_older 2012-1-4-disease > log_mm_data_for_edge_insertion-2012vs2017-DO.txt

python format_mm_data_for_edge_insertion-DO.py --DO_ver_older 2012-1-4-disease > log_mm_data_for_edge_insertion-2012vs2023-DO.txt

python format_mm_data_for_edge_insertion-DO.py --DO_ver_older 2022-9-29-disease --concept_type all --allow_complex_edge > log_mm_data_for_edge_insertion-2022vs2023-all-DO.txt

python format_mm_data_for_edge_insertion-DO.py --DO_ver_older 2023-9-28-disease --concept_type all --allow_complex_edge > log_mm_data_for_edge_insertion-2023vs2023-all-DO.txt

# get complex edge test set
# Disease NIL
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl
# Disease In-KB
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-in-KB.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-in-KB-complex.jsonl
# CPP NIL
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl
# CPP In-KB
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-in-KB.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-in-KB-complex.jsonl
# DO23 NIL
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl
# DO23 In-KB
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-in-KB.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-in-KB-complex.jsonl

# get complex edge train/valid set
# Disease
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/train.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/train-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/valid.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/valid-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/valid-NIL.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/valid-NIL-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-all.jsonl > ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-all-complex.jsonl
# CPP
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/train.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/train-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/valid.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/valid-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/valid-NIL.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/valid-NIL-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-all.jsonl > ../data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-all-complex.jsonl
# DO23
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/train.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/train-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/valid.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/valid-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/valid-NIL.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/valid-NIL-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-complex.jsonl
grep -i "\[EX" ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-all.jsonl > ../data/MedMentions-preprocessed+/2023-9-28-disease/st21pv_syn_attr-all-complexEdge-edges-final/test-NIL-all-complex.jsonl
# get statistics 
# counting the number of NIL mentions in the Disease test set(an example below)
grep -o "\"NIL\"" ../data/MedMentions-preprocessed+/Disease/st21pv_syn_attr-all-complexEdge-filt/test.jsonl | wc -l