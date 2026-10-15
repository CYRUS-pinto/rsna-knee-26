# Label Extraction Pipeline: Qwen-2.5 72B Report Distillation

## Overview
Generates high-confidence soft probability targets for all 12 abnormalities by distilling radiology reports with local open-weights LLMs.

## Rules Compliance
- **Rule 4.b Safe:** Never calls external hosted APIs (e.g. OpenAI / Claude / Gemini endpoints).
- Runs 100% locally or inside isolated GPU environments (A100 / L4) using open-weights models (`Qwen/Qwen2.5-72B-Instruct` or `Qwen/Qwen2.5-14B-Instruct-AWQ`).
- Outputs soft probabilities in $[0, 1]$ capturing radiologist diagnostic certainty ('subtle', 'mild', 'complete tear', 'negative').
- Formatted into `SUPER_CONSENSUS_labels.parquet` for direct loss supervision during training.
