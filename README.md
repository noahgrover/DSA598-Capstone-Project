# DSA598-Capstone-Project
This repository contains Noah Grover's DSA 598 capstone project, submitted on July 22nd, 2026, as required for completion of the Master of Science, Data Science and Analytics program at SUNY Polytechnic Institute.

## DESCRIPTION
This project provides an equity-informed, open-source Python pipeline designed to automate the enrichment of legacy metadata. By integrating domain-customized Named Entity Recognition (NER) and a Large Language Model (LLM) candidate disambiguation layer, the pipeline translates flat-text archival descriptions into discoverable, machine-readable Linked Open Data (LOD). The system extracts key historical entities and grounds them to persistent global identifiers (Wikidata QIDs) while avoiding the algorithmic bias and high hallucination rates common in out-of-the-box generative models.

The pipeline's effectiveness is validated independently across three historically marginalized cohorts to ensure equitable representation:
- Cohort A: Racial/Ethnic Minorities
- Cohort B: LGBTQIA+ Histories
- Cohort C: Indigenous Populations

Full project documentation, including justification, methodology, and discussion of results is located at (pipeline/`DSA598_capstoneProjectDocumentation_Grover.pdf`). Video presentation at ("insert_yt_link_here").

All data comes directly from the Digital Public Library of America (DPLA) API endpoint: (https://api.dp.la/v2/items) 

## DEPENDENCIES & REPRODUCIBILITY
The pipeline is designed to run within Google Colab (Python 3) utilizing its cloud-hosted environment for access to GPU hardware and persistent, version-controlled serialization. All necessary libraries and the versions employed in this project can be found at (pipeline/`requirements.txt`).

The entire engineered pipeline is entirely contained within `DSA598_capstoneProjectNotebook_Grover.ipynb` which is located at (pipeline/`DSA598_capstoneProjectNotebook_Grover.ipynb`). This file relies on .txt files located at (pipeline/promptFiles) for LLM prompting and API keys loaded into Colab Secrets (DPLA_API_KEY and OPENAI_API_KEY). These need to be present in your Colab instance prior to running the notebook. Most cells serialize their output into .json files for human verification and to reduce computational strain. Each serialized file from the latest production run are located at (pipeline/productionRunFiles). These can be used to replicate results, or to test specific cells without re-running the entire pipeline. The first section of the notebook, titled, *Requirements & Reproducibility*, handles installation, import, version verification, and random seed control for the entire notebook. This section must be run first. DPLA is constantly updated, so extracted records change over time. In order to replicate exact results, you must use the provided master file `rawIngestedMetadata.json`. Otherwise, results are reproducible by running cells sequentially. The provided evaluation file, `manualAnnotationFinal.json`, was generated manually, must be present in your Colab instance before modeling, and must match records that are present in `rawIngestedMetadata.json` in order for the evaluation scripts to produce accurate results. The final output, `enriched.jsonld`, is located at (dashboard/`enriched.jsonld`).
