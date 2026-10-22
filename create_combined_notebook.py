#!/usr/bin/env python3
"""
Create a combined Colab notebook for Engine 12 + Engine 13 multi-fold training.
Merges both engines into one sequential notebook.
"""

import json


def read_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_cell_source(nb, cell_idx):
    return ''.join(nb['cells'][cell_idx]['source'])


def create_combined_notebook():
    # Read both notebooks
    nb12 = read_notebook('colab_engine12_multi_fold_master.ipynb')
    nb13 = read_notebook('colab_engine13_multi_fold_master.ipynb')

    # Build combined cells
    combined_cells = []

    # Cell 0: Markdown title
    combined_cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "combined-title"},
        "source": [
            "# RSNA Knee Abnormality Detection — Engine 12 + Engine 13 Multi-Fold Training\n",
            "### Platform: Google Colab Pro (A100 GPU Recommended)\n",
            "**This notebook trains both engines sequentially:**\n",
            "- **Engine 12**: ConvNeXt-Large-384 Gated MIL (198M params) — Folds 1,2,3,4\n",
            "- **Engine 13**: Swin-Base-384 Asymmetric Focal MIL (88M params) — Folds 1,2,3,4\n",
            "- Each fold: 8 epochs, gradient checkpointing, chunk_size=4, k=8 triplets\n",
            "- Auto-publishes weights to Kaggle Datasets after each fold\n",
            "- Estimated total time: ~2.5 hours on A100, ~8 hours on T4\n"
        ]
    })

    # Cell 1: Environment & GPU Diagnostic (from Engine 12, cell 1)
    combined_cells.append(nb12['cells'][1])

    # Cell 2: Authenticate Kaggle & Download Training Assets (from Engine 12, cell 2)
    combined_cells.append(nb12['cells'][2])

    # Cell 3: Write Engine 12 Training Script (parameterized for any fold)
    train_e12_script = get_cell_source(nb12, 3)
    # Modify to be a function that takes fold_num
    train_e12_script = train_e12_script.replace(
        '# 3. Write Engine 12: ConvNeXt-Large Fold 1 Master',
        '# 3. Write Engine 12 Training Script (parameterized for folds 1-4)'
    ).replace(
        "with open('train_engine12_fold1.py', 'w', encoding='utf-8') as f:",
        "with open('train_engine12.py', 'w', encoding='utf-8') as f:"
    ).replace(
        'train_engine12_fold1.py',
        'train_engine12.py'
    ).replace(
        'engine12_convnext_large_fold1_best.pt',
        'engine12_convnext_large_fold{FOLD}.pt'
    ).replace(
        'engine12_fold1_oof.csv',
        'engine12_fold{FOLD}_oof.csv'
    ).replace(
        'engine12_fold1_upload',
        'engine12_fold{FOLD}_upload'
    ).replace(
        'fold1-weights',
        'fold{FOLD}-weights'
    ).replace(
        'Fold 1',
        'Fold {FOLD}'
    )

    combined_cells.append({
        "cell_type": "code",
        "metadata": nb12['cells'][3].get('metadata', {}),
        "execution_count": None,
        "outputs": [],
        "source": [train_e12_script]
    })

    # Cell 4: Write Engine 13 Training Script (parameterized for any fold)
    train_e13_script = get_cell_source(nb13, 3)
    train_e13_script = train_e13_script.replace(
        '# 3. Write Engine 13: Swin-Base Fold 1 Master',
        '# 4. Write Engine 13 Training Script (parameterized for folds 1-4)'
    ).replace(
        "with open('train_engine13_fold1.py', 'w', encoding='utf-8') as f:",
        "with open('train_engine13.py', 'w', encoding='utf-8') as f:"
    ).replace(
        'train_engine13_fold1.py',
        'train_engine13.py'
    ).replace(
        'engine13_swin_base_fold1_best.pt',
        'engine13_swin_base_fold{FOLD}.pt'
    ).replace(
        'engine13_fold1_oof.csv',
        'engine13_fold{FOLD}_oof.csv'
    ).replace(
        'engine13_fold1_upload',
        'engine13_fold{FOLD}_upload'
    ).replace(
        'fold1-weights',
        'fold{FOLD}-weights'
    ).replace(
        'Fold 1',
        'Fold {FOLD}'
    )

    combined_cells.append({
        "cell_type": "code",
        "metadata": nb13['cells'][3].get('metadata', {}),
        "execution_count": None,
        "outputs": [],
        "source": [train_e13_script]
    })

    # Cell 5: Training loop for Engine 12 (Folds 1-4)
    e12_loop = '''# 5. Train Engine 12: ConvNeXt-Large-384 (Folds 1-4)
import os, subprocess, sys, time, gc
import torch

print("=" * 60)
print("ENGINE 12: ConvNeXt-Large-384 Gated MIL")
print("=" * 60)

for FOLD in [1, 2, 3, 4]:
    print(f"\\n{'='*60}")
    print(f"STARTING ENGINE 12 FOLD {FOLD}")
    print(f"{'='*60}\\n")
    
    # Write fold-specific training script
    with open('train_engine12.py', 'r') as f:
        script = f.read()
    script = script.replace('{FOLD}', str(FOLD))
    with open(f'train_engine12_fold{FOLD}.py', 'w') as f:
        f.write(script)
    
    # Run training
    start = time.time()
    result = subprocess.run([sys.executable, f'train_engine12_fold{FOLD}.py'], 
                           capture_output=False, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"[!] Fold {FOLD} training FAILED with return code {result.returncode}")
    else:
        print(f"[+] Fold {FOLD} completed in {elapsed/60:.1f} minutes")
    
    # Publish weights to Kaggle
    print(f"[+] Publishing Engine 12 Fold {FOLD} weights to Kaggle...")
    upload_dir = f'/content/engine12_fold{FOLD}_upload'
    os.makedirs(upload_dir, exist_ok=True)
    weight_file = f'/content/engine12_convnext_large_fold{FOLD}_best.pt'
    oof_file = f'/content/engine12_fold{FOLD}_oof.csv'
    
    if os.path.exists(weight_file):
        subprocess.run(['cp', weight_file, upload_dir + '/'], check=True)
    if os.path.exists(oof_file):
        subprocess.run(['cp', oof_file, upload_dir + '/'], check=True)
    
    meta = {
        'title': f'RSNA Knee Engine 12 ConvNeXt-Large Fold {FOLD} Weights',
        'id': f'cyruspinto/rsna-knee-engine12-fold{FOLD}-weights',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    with open(os.path.join(upload_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    subprocess.run(['kaggle', 'datasets', 'version', '-p', upload_dir, 
                   '-m', f'Engine 12 Fold {FOLD} Weights', '-r', 'zip'], check=True)
    print(f"[+] Published Engine 12 Fold {FOLD} weights to Kaggle!")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"=== Engine 12 Fold {FOLD} Complete ===")

print("\\n" + "="*60)
print("ENGINE 12 ALL FOLDS COMPLETE")
print("="*60)'''

    combined_cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [e12_loop]
    })

    # Cell 6: Training loop for Engine 13 (Folds 1-4)
    e13_loop = '''# 6. Train Engine 13: Swin-Base-384 Asymmetric Focal MIL (Folds 1-4)
import os, subprocess, sys, time, gc
import torch

print("=" * 60)
print("ENGINE 13: Swin-Base-384 Asymmetric Focal MIL")
print("=" * 60)

for FOLD in [1, 2, 3, 4]:
    print(f"\\n{'='*60}")
    print(f"STARTING ENGINE 13 FOLD {FOLD}")
    print(f"{'='*60}\\n")
    
    # Write fold-specific training script
    with open('train_engine13.py', 'r') as f:
        script = f.read()
    script = script.replace('{FOLD}', str(FOLD))
    with open(f'train_engine13_fold{FOLD}.py', 'w') as f:
        f.write(script)
    
    # Run training
    start = time.time()
    result = subprocess.run([sys.executable, f'train_engine13_fold{FOLD}.py'], 
                           capture_output=False, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"[!] Fold {FOLD} training FAILED with return code {result.returncode}")
    else:
        print(f"[+] Fold {FOLD} completed in {elapsed/60:.1f} minutes")
    
    # Publish weights to Kaggle
    print(f"[+] Publishing Engine 13 Fold {FOLD} weights to Kaggle...")
    upload_dir = f'/content/engine13_fold{FOLD}_upload'
    os.makedirs(upload_dir, exist_ok=True)
    weight_file = f'/content/engine13_swin_base_fold{FOLD}_best.pt'
    oof_file = f'/content/engine13_fold{FOLD}_oof.csv'
    
    if os.path.exists(weight_file):
        subprocess.run(['cp', weight_file, upload_dir + '/'], check=True)
    if os.path.exists(oof_file):
        subprocess.run(['cp', oof_file, upload_dir + '/'], check=True)
    
    meta = {
        'title': f'RSNA Knee Engine 13 Swin-Base Fold {FOLD} Weights',
        'id': f'cyruspinto/rsna-knee-engine13-fold{FOLD}-weights',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    with open(os.path.join(upload_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    subprocess.run(['kaggle', 'datasets', 'version', '-p', upload_dir, 
                   '-m', f'Engine 13 Fold {FOLD} Weights', '-r', 'zip'], check=True)
    print(f"[+] Published Engine 13 Fold {FOLD} weights to Kaggle!")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"=== Engine 13 Fold {FOLD} Complete ===")

print("\\n" + "="*60)
print("ENGINE 13 ALL FOLDS COMPLETE")
print("="*60)
print("\\nALL TRAINING COMPLETE! Next steps:")
print("1. Run OOF extraction notebook")
print("2. Run Nelder-Mead optimization")
print("3. Build V50 kernel with optimized weights")'''

    combined_cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [e13_loop]
    })

    # Build final notebook
    combined_nb = {
        "cells": combined_cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "machine_shape": "hm",
                "gpuType": "A100"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU",
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    output_path = 'colab_engine12_13_combined_multi_fold_master.ipynb'
    with open(output_path, 'w') as f:
        json.dump(combined_nb, f, indent=1)
    
    print(f"Created: {output_path}")
    print(f"Total cells: {len(combined_cells)}")
    return output_path


if __name__ == '__main__':
    create_combined_notebook()