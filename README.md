# FairHICON: Fairness-aware Hierarchical Contrastive Learning

Official implementation of **FairHICON**, a framework for interpretable and fairness-aware machine learning in bioinformatics (e.g., Cancer Genomics and Asthma research).

## Overview
FairHICON integrates gene-pathway biological priors with hierarchical contrastive learning to separate common biological signals from sex-specific variations. This ensures more robust and fair representations for downstream clinical tasks like Long-Term Survival (LTS) prediction.

## Key Features
* **Hierarchical Contrastive Loss:** Includes group-wise normalization and adaptive prototype-based hard negative mining.
* **Biological Priors:** Incorporates Sparse Gene-Pathway masks to ensure model interpretability.
* **Fairness-aware:** Explicitly models common and sex-specific (Male/Female) embeddings to reduce bias.
