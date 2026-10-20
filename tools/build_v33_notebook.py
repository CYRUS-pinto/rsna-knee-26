"""
Builder for RSNA Knee Candidate V33: Grandmaster Tri-Hybrid Rank Fusion
Platform: Kaggle GPU (T4 / P100)
Architecture:
- Stage 1A: Raptor CoAtNet-384 SWA (0.65) + Cyrus Super-CoAtNet-384 (0.35, Gold AUC 0.9139)
- Stage 1B: Nartaa CoAtNet-224 SWA (0.8 center crop, 0.30 of Raptor half)
- Stage 2:  Heliosli ConvNeXt-Small R7 (4-fold ensemble + Stride-2 capacity pruning)
- Stage 3:  Scale-Invariant Calibrated Specialist Rank Fusion
"""
import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v31_stride2_capacity_master/rsna-knee-v31-stride2-capacity-master.ipynb")
DST_DIR = Path("submissions/v33_grandmaster_rank_fusion")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v33-grandmaster-fusion.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title / Header Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V33: Grandmaster Tri-Hybrid Rank Fusion (LB 0.960+ Target)\n",
    "\n",
    "**Tri-Hybrid Grandmaster Architecture with Native Super-Consensus Injection:**\n",
    "1. **Stage 1A (CoAtNet-384 Dual-Arm):** Dreaddevelopment CoAtNet-384 SWA (0.65) + Cyrus Super-CoAtNet-384 (0.35, Gold AUC 0.9139, Synovitis 0.8554, Meniscus 0.9627).\n",
    "2. **Stage 1B (CoAtNet-224):** Nartaa CoAtNet-224 SWA (0.8 center crop, 0.30 share of Raptor arm).\n",
    "3. **Stage 2 (ConvNeXt-R7):** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (`s0_f0..f3`) with Flip TTA and Stride-2 Window Pruning.\n",
    "4. **Stage 3 (Quantile Fusion):** Scale-Invariant Calibrated Specialist Rank Fusion across all 12 target pathologies.\n"
]

# 2. Update Cell 2 (/tmp/raptor_run.py)
cell2_src = "".join(nb['cells'][2]['source'])

# Update Checkpoint check at top of Cell 2
old_top_check = """_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
_candidates = [
    Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/rsna-knee-oai-original384-s42-pair') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-oai-original384-s42-pair') / _weight_name,
]
_weights = next((p for p in _candidates if p.is_file()), None)
if _weights is None:
    raise FileNotFoundError("Attach the exact selected weights; no alternative model is substituted.")
with _weights.open('rb') as _f:
    assert hashlib.file_digest(_f, 'sha256').hexdigest() == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Checkpoint hash differs"
"""

new_top_check = """_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
_candidates = [
    Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/rsna-knee-oai-original384-s42-pair') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-oai-original384-s42-pair') / _weight_name,
]
_weights = next((p for p in _candidates if p.is_file()), None)
if _weights is None:
    raise FileNotFoundError("Attach the exact selected weights; no alternative model is substituted.")
with _weights.open('rb') as _f:
    assert hashlib.file_digest(_f, 'sha256').hexdigest() == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Checkpoint hash differs"

# Verify Cyrus Super-CoAtNet Checkpoint presence
_super_name = 'colab_super_coatnet_0917_best.pt'
_super_candidates = [
    Path('/kaggle/input/rsna-knee-colab-super-weights') / _super_name,
    Path('/kaggle/input/datasets/cyruspinto/rsna-knee-colab-super-weights') / _super_name,
]
_super_weights = next((p for p in _super_candidates if p.is_file()), None)
if _super_weights is not None:
    with _super_weights.open('rb') as _f:
        _sh = hashlib.file_digest(_f, 'sha256').hexdigest()
        assert _sh == 'b289852d1eeb48c4b9c588bd52148f429166c4b1a483b6c261c0d992ab063630', f"Super-CoAtNet hash differs: {_sh}"
        print(f"[auth] Cyrus Super-CoAtNet verified: {_super_weights} (SHA: {_sh[:16]}...)", flush=True)
"""

assert old_top_check in cell2_src, "Failed to find old_top_check in Cell 2!"
cell2_src = cell2_src.replace(old_top_check, new_top_check)

# Update direct paths in find_weight_file
old_direct = """def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}"]"""

new_direct = """def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}",
              f"/kaggle/input/rsna-knee-colab-super-weights/{fname}",
              f"/kaggle/input/datasets/cyruspinto/rsna-knee-colab-super-weights/{fname}"]"""

assert old_direct in cell2_src, "Failed to find old_direct in Cell 2!"
cell2_src = cell2_src.replace(old_direct, new_direct)

# Update ARMS definition in Cell 2
old_arms = """ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]"""

new_arms = """ARMS = [
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.65},
    {"file": "colab_super_coatnet_0917_best.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.35},
]"""

assert old_arms in cell2_src, "Failed to find old_arms in Cell 2!"
cell2_src = cell2_src.replace(old_arms, new_arms)

# Update load_model sha assertion in Cell 2
old_load_sha = """def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    import hashlib
    with open(pt_path, "rb") as stream:
        observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"
    print(f"[weights] sha256 {observed_sha}", flush=True)"""

new_load_sha = """def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    import hashlib
    with open(pt_path, "rb") as stream:
        observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    VALID_SHAS = {
        '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef': 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa',
        'b289852d1eeb48c4b9c588bd52148f429166c4b1a483b6c261c0d992ab063630': 'colab_super_coatnet_0917_best'
    }
    assert observed_sha in VALID_SHAS, f"Bound checkpoint bytes differ: {observed_sha}"
    print(f"[weights] {VALID_SHAS[observed_sha]} sha256 {observed_sha}", flush=True)"""

assert old_load_sha in cell2_src, "Failed to find old_load_sha in Cell 2!"
cell2_src = cell2_src.replace(old_load_sha, new_load_sha)

nb['cells'][2]['source'] = cell2_src.splitlines(keepends=True)

# 3. Update Cell 8 (Blend Receipt & Logging)
cell8_src = "".join(nb['cells'][8]['source'])
cell8_src = cell8_src.replace(
    "'mode': 'Candidate V31 Stride-2 Four-Fold Capacity Master (4-Fold R7 Stride-2 + MultiScale Raptor)'",
    "'mode': 'Candidate V33 Grandmaster Tri-Hybrid Fusion (Raptor-384 + Cyrus Super-CoAtNet-384 + Raptor-224 + ConvNeXt-R7)'"
)
nb['cells'][8]['source'] = cell8_src.splitlines(keepends=True)

# 4. Save Notebook
json.dump(nb, open(DST_NOTEBOOK, "w", encoding="utf-8"), indent=1)
print(f"SUCCESS: Generated Candidate V33 notebook at {DST_NOTEBOOK}")

# 5. Create kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v33-grandmaster-fusion",
    "title": "RSNA Knee V33 Grandmaster Fusion",
    "code_file": "rsna-knee-v33-grandmaster-fusion.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "heliosli/rsna-knee-compact-r7cnxfsss",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        "cyruspinto/rsna-knee-colab-super-weights"
    ],
    "kernel_sources": [],
    "competition_sources": [
        "rsna-knee-abnormality-detection"
    ],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"
}

with open(DST_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"SUCCESS: Generated kernel-metadata.json at {DST_META}")
