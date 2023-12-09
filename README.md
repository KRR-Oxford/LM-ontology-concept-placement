# LM-ontology-concept-placement
Language Model based ontology concept placement

The study provides a Language Model based framework (including pre-trained and large language models) for new concept placement in ontologies, where the input includes a mention in a text corpus together with an ontology, and the outputs are the predicted edges in the ontology to place the mention.

The methods combines LMs together with ontology structure, and includes three steps:
* (concept or) edge search, 
* edge formation and enrichment, and 
* edge selection. 

This repository provides the implementation of the methods above, running scripts, and the dataset scripts for research-based reproducibility. 

# Requirements
The repository is based on `Python 3.8`. 

See `requirements.txt`, for running Edge-Bi-encoder, Edge-Cross-encoder, Inverted Index, Fixed Embedding. 

See `requirements-LLM.txt`, for running instruction tuning LLMs.

# Model Training and Inference
See `Edge-Bi-enc+prompt-generation.sh` for the steps of running Edge-Bi-encoder, edge enrichment, and prompt generation, with running examples in `Edge-Bi-enc+prompt-gen-run-example.sh`.

See `Edge-Bi-enc+Cross-enc.sh` for the steps of running Edge-Bi-encoder, edge enrichment, and Edge-Cross-encoder, with running examples in `Edge-Bi-enc+Cross-enc-run-example.sh`.

See `run_tune_LLAMA_2_from_data_creation.sh` a running example for data generation, instruction-tuning, and prompting of LLAMA-2.

See `blink/prompt/run_search_snomed_disease-5to10.sh` and similar files for the examples of running Inverted Index and fixed embedding based approarches.

See other files in `blink/prompt` for the prompting of GPT-3.5-turbo, FLAN-T5, and Llama-2.

For all Edge-Bi-enc and Edge-Cross-enc scripts above:
* setting `train_bi` (train Bi-encoder), `rep_ents` (pre-calculate edge embeddings), `eval_biencoder` (inference with Bi-encoder and get data for cross encoder), `train_cross` (train Cross-encoder), `inference` (whole inference) to `true` to select to perform (or not perform) each step. 
* setting `eval_set` to `train`,`valid`,`valid-NIL`,`test-NIL` with comma separated for the `eval_biencoder` step to generate data for each data split.

For Edge-Bi-enc:
* setting `use_cand_analysis` (evaluate Bi-encoder results and generating initial instructions and prompts for LLMs) to true to perform the step.

# Datasets
Our work uses the datasets at [Zenodo](https://zenodo.org/record/8228005) and its JSON keys are described in the `dataset` folder. 

# Data and processing sources
Before data creation, the below sources need to be downloaded.
* SNOMED CT https://www.nlm.nih.gov/healthit/snomedct/archive.html (and use snomed-owl-toolkit to form .owl files)
* UMLS https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html (and mainly use MRCONSO for mapping UMLS to SNOMED CT)
* MedMentions https://github.com/chanzuckerberg/MedMentions (source of entity linking)

The below tools and libraries are used.
* Protege http://protegeproject.github.io/protege/
* snomed-owl-toolkit https://github.com/IHTSDO/snomed-owl-toolkit
* DeepOnto https://github.com/KRR-Oxford/DeepOnto (based on OWLAPI https://owlapi.sourceforge.net/) for ontology processing and complex concept verbalisation

# Data creation scripts
Based on [OET](https://github.com/KRR-Oxford/OET), the data creation scripts are available at `data-construction` folder, where `run_preprocess_ents_and_data+new.sh` provides an overall shell script that calls the other `.py` files.

# Acknowledgement
* Our dataset is based on [OET](https://github.com/KRR-Oxford/OET) and [zenodo link](https://zenodo.org/record/8228005).
* The baseline implementations are based on [BLINKout paper](https://arxiv.org/abs/2302.07189) and [BLINK repository](https://github.com/facebookresearch/BLINK) under the MIT liscence. 
* Acknowledgement to all [data and processing sources](https://github.com/KRR-Oxford/OET#data-and-processing-sources) listed above.