# Distilled Labels & Consensus Datasets

## Overview
This directory stores reproducible consensus labels extracted from competition metadata and radiology reports.

## Files
- `SUPER_CONSENSUS_labels.parquet`: Distilled multi-label probability targets (0.9172 CV Macro-AUC) across 4,349 studies for all 12 targets.
- `qwen_extracted_soft_labels.csv`: Raw soft probabilities extracted via local Qwen-2.5 72B report parsing.
- `HYBRID_labels_scores.csv`: Cross-validated label reliability scores used for sample re-weighting.
