import json
import os
from pathlib import Path

src_nb_path = 'scratch/heliosli_r7_nb/this-one-is-old-version-0-9573-newone-you-can-use.ipynb'
dst_dir = 'benchmarks/arm_audit_120_studies'
dst_nb_path = os.path.join(dst_dir, 'rsna-knee-arm-audit-120-studies.ipynb')
dst_meta_path = os.path.join(dst_dir, 'kernel-metadata.json')

os.makedirs(dst_dir, exist_ok=True)

with open(src_nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Insert Cell 1.5: Setup 120 studies from train_series into /kaggle/working/audit_data and extract Laterality
setup_code = """# Cell 1.5: Setup 120 training studies for cross-arm audit and extract DICOM Laterality
import os, sys, glob, re, shutil
from pathlib import Path
import pandas as pd
import pydicom

print("[*] Setting up 120 training studies for cross-arm audit...")
TRAIN_CSV = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv')
if not TRAIN_CSV.exists():
    TRAIN_CSV = Path('/kaggle/input/rsna-knee-abnormality-detection/train.csv')

train_df = pd.read_csv(TRAIN_CSV)
uids = train_df['StudyInstanceUID'].drop_duplicates().iloc[:120].tolist()

audit_dir = Path('/kaggle/working/audit_data')
audit_dir.mkdir(parents=True, exist_ok=True)
audit_series_dir = audit_dir / 'test_series'
audit_series_dir.mkdir(parents=True, exist_ok=True)

TRAIN_SERIES_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series')
if not TRAIN_SERIES_ROOT.exists():
    TRAIN_SERIES_ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection/train_series')

def extract_side(dcm_path):
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        # Direct DICOM tags
        for tag in ['ImageLaterality', 'Laterality', 'FrameLaterality']:
            val = getattr(ds, tag, None)
            if val in ('L', 'R'):
                return val
        # Text descriptions
        for field in ['SeriesDescription', 'StudyDescription', 'ProtocolName']:
            txt = str(getattr(ds, field, '')).lower()
            if re.search(r'\\b(rt|right)\\b', txt):
                return 'R'
            if re.search(r'\\b(lt|left)\\b', txt):
                return 'L'
    except Exception:
        pass
    return 'UNKNOWN'

lat_records = []
linked_uids = []

for uid in uids:
    src_p = TRAIN_SERIES_ROOT / uid
    dst_p = audit_series_dir / uid
    if src_p.is_dir() and not dst_p.exists():
        os.symlink(src_p, dst_p)
        linked_uids.append(uid)
        
        # Search across all series in this study to resolve laterality
        side = 'UNKNOWN'
        for s_dir in src_p.glob('*'):
            if s_dir.is_dir():
                dcm_files = list(s_dir.glob('*.dcm'))
                if dcm_files:
                    side = extract_side(dcm_files[0])
                    if side != 'UNKNOWN':
                        break
        lat_records.append({'StudyInstanceUID': uid, 'Laterality': side})

# Write mock test.csv
mock_test_df = pd.DataFrame({'StudyInstanceUID': linked_uids})
mock_test_df.to_csv(audit_dir / 'test.csv', index=False)

# Write mock sample_submission.csv (required by probe_dual.py)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
mock_ss_df = pd.DataFrame([[u] + [0.5]*12 for u in linked_uids], columns=["StudyInstanceUID"] + LABELS)
mock_ss_df.to_csv(audit_dir / 'sample_submission.csv', index=False)

# Write mock test_series.csv (required by Raptor)
train_series_csv = TRAIN_CSV.parent / 'train_series.csv'
if train_series_csv.exists():
    ts_df = pd.read_csv(train_series_csv)
    mock_ts_df = ts_df[ts_df['StudyInstanceUID'].isin(linked_uids)].copy()
    mock_ts_df.to_csv(audit_dir / 'test_series.csv', index=False)
    print(f"[+] Successfully generated test_series.csv with {len(mock_ts_df)} series entries.")

# Write laterality.csv
lat_df = pd.DataFrame(lat_records)
lat_df.to_csv('/kaggle/working/laterality.csv', index=False)

print(f"[+] Successfully linked {len(linked_uids)} studies into {audit_dir}")
print(f"[+] Laterality breakdown:\\n{lat_df['Laterality'].value_counts()}")
"""

setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [setup_code]
}

# Insert setup cell right before Raptor cell (which was index 2)
nb['cells'].insert(2, setup_cell)

# Helper to redirect find_test_root in cells
def patch_find_test_root(src_str):
    target = "def find_test_root():\n"
    replacement = "def find_test_root():\n    if os.path.exists('/kaggle/working/audit_data/test.csv'):\n        return '/kaggle/working/audit_data'\n"
    assert target in src_str, "def find_test_root not found"
    return src_str.replace(target, replacement, 1)

# Cell 3 (was 2): Raptor 384
c3 = ''.join(nb['cells'][3]['source'])
c3 = patch_find_test_root(c3)
nb['cells'][3]['source'] = [c3]

# Cell 5 (was 4): Raptor 224
c5 = ''.join(nb['cells'][5]['source'])
c5 = patch_find_test_root(c5)
nb['cells'][5]['source'] = [c5]

# Cell 7 (was 6): Asset mounting and data path
c7 = ''.join(nb['cells'][7]['source'])

old_find_mount = """def find_mount(slug, required):
    cands = [INPUT / slug, INPUT / 'competitions' / slug]
    cands += list(INPUT.glob('datasets/*/' + slug)) + list(INPUT.glob('datasets/*/' + slug + '/*'))
    cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))
    found = []
    for p in cands:
        if (p / required).is_file() and p not in found:
            found.append(p)
    if len(found) != 1:
        raise RuntimeError('Cannot resolve %s: %s' % (slug, found))
    return found[0]"""

new_find_mount = """def find_mount(slug, required):
    slug_clean = slug.replace('_', '-').lower()
    for root, dirs, files in os.walk('/kaggle/input'):
        if required in files:
            root_clean = root.replace('_', '-').lower()
            if slug_clean in root_clean:
                return Path(root)
    for root, dirs, files in os.walk('/kaggle/input'):
        if required in files:
            return Path(root)
    raise RuntimeError(f'Cannot resolve {slug} with {required}')"""

assert old_find_mount in c7, "old_find_mount not found in cell 7"
c7 = c7.replace(old_find_mount, new_find_mount)

# Replace DATA path in cell 7
old_data_line = "DATA = find_mount(COMPETITION, 'test.csv')"
new_data_line = "DATA = Path('/kaggle/working/audit_data') if (Path('/kaggle/working/audit_data') / 'test.csv').is_file() else find_mount(COMPETITION, 'test.csv')"
assert old_data_line in c7, "DATA line not found in cell 7"
c7 = c7.replace(old_data_line, new_data_line)

# Replace C_SLUG with robust R7 finder
old_c_slug = "C_SLUG = 'rsna-knee-r6-frozen-fe055406'   # [d957] the frozen bundle dataset; its version 2 = the R7 bundle (manifest sha256 c4bb0f7399ada9a0.. = heliosli/rsna-knee-compact-r7cnxfsss's)"
new_c_slug = "C_SLUGS = ['rsna-knee-compact-r7cnxfsss', 'rsna-knee-r6-frozen-fe055406']"
assert old_c_slug in c7, "old_c_slug not found in cell 7"
c7 = c7.replace(old_c_slug, new_c_slug)

old_m_src = "M_SRC = find_mount(C_SLUG, 'manifest.json')"
new_m_src = """# Robust resolution of R7 bundle from mounted datasets
def find_r7_mount(required='manifest.json'):
    for root, dirs, files in os.walk('/kaggle/input'):
        if required in files:
            p = Path(root)
            try:
                data = json.loads((p / required).read_text())
                if data.get('bundle_id') == BUNDLE_ID:
                    return p
            except Exception:
                pass
            if 'r7' in root.lower() or 'r6' in root.lower():
                return p
    return find_mount('rsna-knee-compact-r7cnxfsss', required)
M_SRC = find_r7_mount('manifest.json')"""

assert old_m_src in c7, "old_m_src not found in cell 7"
c7 = c7.replace(old_m_src, new_m_src)

nb['cells'][7]['source'] = [c7]

# Cell 10: Append Cross-Arm Check with Laterality Split
eval_code = """# Cell 10: Empirical Cross-Arm Agreement & Swap Check on 120 Studies
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr

L = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
a = pd.read_csv('/kaggle/working/raptor_sub.csv')
b = pd.read_csv('/kaggle/working/ours_sub.csv')
lat = pd.read_csv('/kaggle/working/laterality.csv')

m = a.merge(b, on='StudyInstanceUID', suffixes=('_a', '_b'))
n = len(m)
print(f"=== EMPIRICAL CROSS-ARM CHECK RESULTS (n={n}) ===")
if n < 100:
    print("WARNING: <100 studies; too noisy to act on.")

def rho(d, x, y):
    return spearmanr(d[x], d[y])[0]

rng = np.random.default_rng(0)
def boot(x, y, k=300):
    v = [spearmanr(m[x].values[i], m[y].values[i])[0] for i in (rng.integers(0, n, n) for _ in range(k))]
    return np.percentile(v, [2.5, 97.5])

print(f"\\n{'finding':<18}{'rho(A,B)':>9}  95% CI")
for c in L:
    lo, hi = boot(c+'_a', c+'_b')
    print(f"{c:<18}{rho(m, c+'_a', c+'_b'):>9.3f}  [{lo:.2f}, {hi:.2f}]")

pairs = [('Medial Meniscus','Lateral Meniscus'), ('Medial OA','Lateral OA')]
print("\\nGLOBAL SWAP TEST (diag = same finding; swapped = medial<->lateral):")
for mm, ll in pairs:
    d = (rho(m, mm+'_a', mm+'_b') + rho(m, ll+'_a', ll+'_b')) / 2
    s = (rho(m, mm+'_a', ll+'_b') + rho(m, ll+'_a', mm+'_b')) / 2
    print(f"{mm}/{ll}: diag={d:.3f} swapped={s:.3f}" + ("  <-- POSSIBLE GLOBAL SWAP" if s > d + 0.10 else ""))

print("\\nNOTE: if a swap happens for only SOME studies (e.g. one knee side), the global test can miss it.")
print("Signature then: these 4 columns agree much less than the other 8, with diag AND swapped both low.")

mm_ = m.merge(lat, on='StudyInstanceUID')
print(f"\\nBY LATERALITY (n={len(mm_)}): diag rho per side, medial and lateral columns")
for side, g in mm_.groupby('Laterality'):
    if len(g) < 15:
        print(f"side {side}: n={len(g)} too small")
        continue
    out = [f"{c}={rho(g, c+'_a', c+'_b'):.2f}" for p in pairs for c in p]
    sw = [f"{p[0]}~{p[1]}={rho(g, p[0]+'_a', p[1]+'_b'):.2f}" for p in pairs]
    print(f"side {side} (n={len(g)}): " + "  ".join(out) + "  | cross: " + "  ".join(sw))
print("A side whose diag is low/negative while the other side is high => laterality-dependent swap.")
"""

eval_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [eval_code]
}
nb['cells'].append(eval_cell)

with open(dst_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

meta = {
  "id": "cyruspinto/rsna-knee-arm-audit-120-studies",
  "title": "RSNA Knee Arm Audit 120 Studies",
  "code_file": "rsna-knee-arm-audit-120-studies.ipynb",
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
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "rsna-knee-abnormality-detection"
  ],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}

with open(dst_meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print("[+] Successfully generated 120-study cross-arm audit notebook and metadata!")
