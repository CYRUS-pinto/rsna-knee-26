import json
import os

os.makedirs('scratch/v7_asymmetric_kernel', exist_ok=True)

with open('scratch/clean_0956_safe.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])

old_weights = """W_RAPTOR, W_OURS = 0.5, 0.5
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}   # per-finding Raptor weight (ours = 1 - it); every other finding uses W_RAPTOR"""

new_weights = """W_RAPTOR, W_OURS = 0.5, 0.5
W_RAPTOR_C = {
    'ACL': 0.65, 'MCL': 0.60, 'Contusion': 0.60, 'Fracture': 0.55,
    'Medial OA': 0.30, 'Lateral OA': 0.30, 'PF OA': 0.30,
    'Effusion': 0.35, "Baker's": 0.35,
    'Medial Meniscus': 0.35, 'Lateral Meniscus': 0.35, 'Synovitis': 0.40
}"""

assert old_weights in c6, 'old_weights target string not matched in c6'
c6_v7 = c6.replace(old_weights, new_weights)
nb['cells'][6]['source'] = [c6_v7]

out_nb = 'scratch/v7_asymmetric_kernel/rsna-knee-quint-ensemble-r7-colab.ipynb'
with open(out_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

meta = {
    'id': 'cyruspinto/rsna-knee-quint-ensemble-r7-colab-swa-0960',
    'title': 'RSNA Knee Quint Ensemble R7 Colab SWA 0960',
    'code_file': 'rsna-knee-quint-ensemble-r7-colab.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': False,
    'dataset_sources': [
        'heliosli/rsna-knee-compact-r7cnxfsss',
        'nartaa/rsna-knee-publication-swa-weights-20261007',
        'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'
    ],
    'competition_sources': [
        'rsna-knee-abnormality-detection'
    ],
    'kernel_sources': []
}

with open('scratch/v7_asymmetric_kernel/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print('V7 kernel built and ready in scratch/v7_asymmetric_kernel/')
