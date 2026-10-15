# `tools/gate_make_root.py` Architectural Documentation

## Purpose
`gate_make_root.py` constructs a self-contained, offline evaluation dataset composed strictly of the 58 expert-annotated gold studies extracted from `train.csv`.

## Operational Mechanics
1. **Filtering**: Identifies studies where consensus annotations exist across all 12 target pathologies without missing labels.
2. **Directory Mocking**: Recreates the official Kaggle test directory tree (`test.csv`, `test_dicoms/` or symlinks) so production inference scripts run without modification.
3. **Gold Isolation**: Exports `gold.csv` with true continuous labels to serve as the ground-truth reference for `gate_score.py`.

## Usage
```bash
python tools/gate_make_root.py \
  --train-csv data/train.csv \
  --output-dir gate_root/ \
  --symlink-dicoms
```
