# DSA598-Capstone-Project
This repository contains Noah Grover's DSA 598 capstone project, submitted on July 22nd, 2026, as required for completion of the Master of Science, Data Science and Analytics program at SUNY Polytechnic Institute.

## DESCRIPTION:
Public, academic, and special libraries hold millions of digitized archival records detailing the vital contributions of underrepresented groups, including racial minorities, LGBTQIA+ communities, and Indigenous populations. However, these materials frequently suffer from "historical metadata debt," where legacy cataloging structures treat information as flat, static text strings rather than interconnected concepts. This results in isolated records that remain functionally hidden from researchers and community members using traditional keyword searches.

This project provides an equity-informed, open-source Python data science pipeline designed to automate the enrichment of legacy metadata. By integrating domain-customized Named Entity Recognition (NER) and a Large Language Model (LLM) candidate disambiguation layer, the pipeline translates flat-text archival descriptions into discoverable, machine-readable Linked Open Data (LOD). The system extracts key historical entities and grounds them to persistent global identifiers (Wikidata QIDs) while avoiding the algorithmic bias and high hallucination rates common in out-of-the-box generative models.

The pipeline's effectiveness is validated independently across three historically marginalized cohorts to ensure equitable representation:
- Cohort A: Racial/Ethnic Minorities
- Cohort B: LGBTQIA+ Histories
- Cohort C: Indigenous Populations

## Dependencies
The pipeline is designed to run within Google Colab (Python 3) utilizing its cloud-hosted environment for access to GPU hardware and persistent, version-controlled serialization.

The Jupyter notebook containing the pipeline relies on .txt files () for its LLM prompts and API keys loaded into Colab Secrets. These will need to be loaded into your Colab environment for the notebook to complete.
