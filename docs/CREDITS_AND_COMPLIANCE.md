# Credits and Compliance

## Credits

This work stands on public contributions from the Kaggle community. In order of how much of the final pipeline depends on them:

| Contributor | Contribution used |
|---|---|
| **dreaddevelopment** | Raptor CoAtNet / MIL inference recipe and public baseline weights ([dataset](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)). Our checkpoints are retrained, not these baseline weights. |
| **nartaa** | Published Raptor checkpoints (384 px and 224 px) and both inference scripts, in the `rsna-knee-publication-swa-weights-20261007` dataset |
| **heliosli** | Compact-model inference code, R6 frozen bundle and mainline code dataset used as the second engine |
| **varshithakolupuri** | The 224 px stage and its 35% share of the Raptor half ([notebook](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956)) |
| **stevenleehans** | [LLM report labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) (weak labels) |
| **pilkwang** | [LLM labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) (weak labels) |
| **riadmohamed42** | [JEV labels and verification](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification) |
| **timm** | CoAtNet backbone implementation |

Source radiology reports and study-level label tables are **not** redistributed in the notebooks.

## Required acknowledgement (OAI / NDA)

Reproduced from the notebooks, as required for use of the Osteoarthritis Initiative data:

> Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.

[Shared NDA Study](https://nda.nih.gov/study.html?id=3407) | [DOI 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676) | [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation)

## External-data inventory

What the two final candidates depend on, as far as the notebooks show:

| Component | External data involved | Evidence in notebooks |
|---|---|---|
| Raptor-384 / 224 checkpoints (Kaggle input is named "RSNA Knee selected SWA weights 0949 and 0945") | Trained with OAI-derived data (the checkpoint filenames include `oai`, and the OAI acknowledgement is required) | Filenames, acknowledgement text |
| Weak labels | Public Kaggle label datasets listed above, plus our own report-derived labels | Credits cell |
| Heliosli bundle | Depends on the bundle manifest. If its views are `v3k`/`v4`, an SKM-TEA anatomy checkpoint (`anatomy.pt`) is used; if the backbone is `ofl_l`, frozen OFL encoder weights are used | Conditional branches in cell 6; the manifest itself was not in the files provided |
| Competition DICOMs | Provided under the competition rules | n/a |

**Action items**

1. Open the R6 bundle's `manifest.json` and record here which of the conditional branches were actually active.
2. The earlier repository README carried a badge reading "Zero External Data Track / 100% Competition Clean". The final candidates above are the *frontier* (Track 1) system and **do** use external-data-derived weights, so that badge should not describe them. It has been removed from the rewritten README. The 100%-clean claim applies only to the separate in-house Track 2 model.
3. Add a `LICENSE` file, and re-check the competition's prize-winner obligations (the earlier README cites Section 2.8) before publishing weights or labels.
