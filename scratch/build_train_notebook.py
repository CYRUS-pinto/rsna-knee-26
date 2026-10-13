import json
import os

os.makedirs('scratch/super_train_kernel', exist_ok=True)

with open('scratch/kernels/dread_train/knee-mri-training-the-twelve-finding-model.py', encoding='utf-8') as f:
    train_code = f.read()

# We can put the train code directly into cells of the notebook
# Cell 1: Description
c1 = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# RSNA Knee Super-Consensus Model Training\n',
        '### Supervision: 0.9159 Macro-AUC Super-Consensus Labels\n',
        '### Validation Anchor: 58 Gold Adjudicated Studies\n',
        '### Backbone: CoAtNet RMLP 2 RW 384 with Multi-Instance Learning (MIL) Attention'
    ]
}

# Cell 2: Dependencies
c2 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '!pip install -q timm albumentations\n',
        'import timm\n',
        'print("timm version:", timm.__version__)'
    ]
}

# Cell 3: The training script runner written to file
c3 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Write training script to train_knee.py\n',
        'with open("train_knee.py", "w", encoding="utf-8") as f:\n',
        '    f.write(' + json.dumps(train_code) + ')\n',
        'print("train_knee.py ready.")'
    ]
}

# Cell 4: Execute training
c4 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import sys\n',
        '!python train_knee.py \\\n',
        '    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \\\n',
        '    --res 384 \\\n',
        '    --epochs 12 \\\n',
        '    --bs 4 \\\n',
        '    --k 12 \\\n',
        '    --k_eval 24 \\\n',
        '    --grad_ckpt \\\n',
        '    --tag super_coatnet_0916 \\\n',
        '    --labels /kaggle/input/rsna-knee-super-consensus-labels-0916/SUPER_CONSENSUS_labels.parquet \\\n',
        '    --workers 2'
    ]
}

# Cell 5: Report results
c5 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '!ls -lh raptor_ft_super_coatnet_0916*\n',
        'import json, glob\n',
        'for js in glob.glob("raptor_ft_*.json"):\n',
        '    print("=== " + js + " ===")\n',
        '    with open(js) as f:\n',
        '        print(json.dumps(json.load(f), indent=2))'
    ]
}

nb = {
    'cells': [c1, c2, c3, c4, c5],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('scratch/super_train_kernel/rsna-knee-super-train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

meta = {
    'id': 'cyruspinto/rsna-knee-super-train',
    'title': 'RSNA Knee Super Train 0916',
    'code_file': 'rsna-knee-super-train.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_gpu': True,
    'enable_gpu': True,
    'enable_internet': True,
    'dataset_sources': [
        'dreaddevelopment/knee-raptor-corpus',
        'dreaddevelopment/knee-raptor-corpus-ext',
        'cyruspinto/rsna-knee-super-consensus-labels-0916'
    ],
    'competition_sources': [
        'rsna-knee-abnormality-detection'
    ]
}

with open('scratch/super_train_kernel/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print('Done creating training kernel files.')
