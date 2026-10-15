# Validation Framework (`cv_framework.py`)

## 1. Executive Summary & Purpose
The RSNA Knee Abnormality Detection challenge evaluated on **Macro-Averaged ROC-AUC** across 12 pathologies. The competition public dataset contains only **58 fully-annotated "Gold Anchor" studies** with ground truth across all 12 classes, alongside 4,349 studies with uncurated or LLM-extracted soft labels ("Silver" cohort).

A major failure mode in previous submissions (e.g. V7-V12) was evaluating models solely on the 58 gold studies or applying heuristic rules tuned to this tiny sample ($N=58$). This caused massive variance and severe public leaderboard degradation (-0.003 on V12).

`cv_framework.py` establishes a **reproducible, leak-free 5-fold Stratified Group Cross-Validation partition** across the entire 4,407-study universe.

---

## 2. Theoretical & Mathematical Foundation

### 2.1 Multi-Label Rarity Weighting
Standard K-Fold stratification operates on single-label classes. In multi-label pathology detection, rare conditions (such as Fracture $\sim 1.7\%$ prevalence, or Lateral OA $\sim 5\%$) must not concentrate in a single fold.

The rarity score $R_i$ for study $i$ across labels $j \in \{1, \dots, 12\}$ is formulated as:
$$R_i = \sum_{j=1}^{12} \frac{y_{i, j}}{\max(1, N_j)}$$
where $N_j = \sum_{k=1}^N y_{k, j}$ is the total marginal frequency of pathology $j$.

Studies are ordered by descending rarity score $R_i + \epsilon$ (with uniform jitter $\epsilon \sim U(0, 10^{-6})$ for non-deterministic tie-breaking across runs), and greedily assigned to the fold with the largest marginal deficit for the active pathologies.

### 2.2 Gold Anchor Stratification Isolation
To ensure every fold contains an unbiased, representative sample of both ground truth gold data and pseudo-labeled consensus data:
1. Gold studies ($N=58$) are partitioned independently using iterative multi-label stratification ($11-12$ gold studies per fold).
2. Silver studies ($N=4,349$) are partitioned independently into the same 5 folds ($869-870$ studies per fold).
3. The union guarantees exactly 20% of every pathology and 20% of ground-truth gold evidence in each fold.

---

## 3. Interfaces & Data Contract

### Input Requirements
- `SUPER_CONSENSUS_labels.parquet`: Master table of 4,407 studies with continuous consensus probabilities $[0, 1]$ across all 12 pathologies.
- `train.csv`: Competition ground-truth table identifying the 58 Gold Anchor studies where all 12 target columns are non-null.

### Output Artifact
- `data/train_folds.parquet`:
  - `StudyInstanceUID`: String identifier.
  - `fold`: Integer $\in \{0, 1, 2, 3, 4\}$.
  - `is_gold`: Boolean flag.
  - Columns 1-12: Ground truth and consensus target probabilities.

---

## 4. Verification & Testing
The partition logic is 100% dependency-free (pure NumPy and Pandas). It is verified via [`validation/test_cv_framework.py`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/validation/test_cv_framework.py), testing:
- Exact 5-fold distribution ($\pm 1$ study maximum imbalance).
- Exact Gold partition balance ($11-12$ per fold).
- Complete disjoint sets ($\bigcap_{k=0}^4 \text{Val}_k = \emptyset$).
- Zero missing values.
