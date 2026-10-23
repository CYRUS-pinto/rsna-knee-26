# Reproducibility

## Environment

| Item | Value |
|---|---|
| Platform | Kaggle Notebook |
| Accelerator | 2x Tesla T4 (15 GB each) |
| Internet | Off (all weights and wheels come from attached datasets) |
| torch | 2.11.0+cu128 (as seen in the logs) |
| Pinned in Stage 2 | `timm 1.0.25`, `pydicom 3.0.1` (installed offline from wheels) |
| Competition | `rsna-knee-abnormality-detection` |

## Attached inputs

| Purpose | Dataset / slug | Integrity check in notebook |
|---|---|---|
| Competition data | `rsna-knee-abnormality-detection` | must contain `test.csv` |
| Raptor-384 weights | `nartaa/rsna-knee-publication-swa-weights-20261007` (fallback: `nartaa/rsna-knee-oai-original384-s42-pair`) | SHA-256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` |
| Raptor-224 weights | same publication dataset (fallback: `nartaa/rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224`) | SHA-256 `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6` |
| Mainline code | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | 8 module SHA-256 values asserted |
| Model bundle | `heliosli/rsna-knee-r6-frozen-fe055406-w1` | `manifest.json` `bundle_id == fe055406fad6a12f`, per-file SHA-256 |
| OFL frozen encoder | `rsna-knee-ofl-ssl-weights` | only read if the manifest's backbone is `ofl_l` |

The notebooks refuse to substitute a different checkpoint: a missing file or hash mismatch stops the run.

## Running

1. Create a Kaggle notebook from `submissions/v20_power_rank_blend/` or `submissions/v45_sovereign_rank_master/`.
2. Select 2x T4, disable internet, attach the inputs above.
3. Run all cells in order. Cells 2 and 4 write `/tmp/raptor_run.py` and `/tmp/raptor224_run.py`; cells 3, 5, 7 execute the three stages; cell 8 blends.
4. Output: `/kaggle/working/submission.csv` (12 columns plus `StudyInstanceUID`).

V20 additionally writes `blend_receipt.json` and `submission_receipt.json`. V45 does not write a blend receipt.

## Recorded runtimes (3-study dry run)

Times are seconds. "Assets ready" is cumulative from notebook start.

| Stage | V20 (measured) | V45 (measured) |
|---|---|---|
| Raptor-384 | 50 | 79 |
| Raptor-224 | 23 | 33 |
| Assets ready (cumulative) | 90.0 | 181.4 |
| Heliosli controller | 360.1 | 335.1 |
| Notebook total (internal) | 451.3 (`blend_receipt.json`) | no printed value |
| **Kaggle run time** | **469.4** | **533.9** |

Two things to note:

- My earlier estimate for V45 (about 516 s from its logs) was close; Kaggle's own figure is 533.9 s. The extra time over the internal total is notebook start-up and conversion.
- The extra ~65 s in V45 is cold-start I/O (model loads took 24 s and 7 s versus 3 s and 2 s in V20's run), not extra compute: the blend step added no meaningful work. Still, if an Efficiency-Prize-style ceiling of 500 s applied, V20's Kaggle run (469.4 s) is under it and V45's (533.9 s) is not.

A 3-study run also says little about the hidden-test runtime, which scales with the number of studies.

## Verification checklist after a run

- Log shows `[weights] sha256 ...` for both Raptor checkpoints.
- Log shows `ASSETS_READY ... flip_tta True`.
- `CONTROLLER_DONE` reports `"status": "PASS"` and `"fallback_studies": 0`.
- V20: `NOTEBOOK_COMPLETE` printed. V45: `Wrote verified Candidate V45 submission` printed.
- `submission.csv` has one row per `test.csv` study, 13 columns, no NaN.
