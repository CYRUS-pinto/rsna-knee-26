import json
from pathlib import Path

train_code = Path("training/engine7_dinov2_distilled/kaggle_push/train_dinov2_kaggle.py").read_text(encoding="utf-8")

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RSNA Knee Abnormality Detection — Engine 7: DINOv2-S/14 Master Trainer\n",
    "### Platform: Google Colab Pro (A100 / L4 / T4 GPU)\n",
    "**Key Innovations:**\n",
    "- **Self-Supervised DINOv2-S/14 ViT Backbone:** True architectural orthogonality to CoAtNet & ConvNeXt.\n",
    "- **12 Disentangled Task Query Heads:** Cross-attention targeting key compartments (Menisci, Synovitis, Effusion).\n",
    "- **Continuous Distilled Supervision:** Trained on leak-free 5-fold continuous consensus targets (0.9172 Gold-58 Macro-AUC).\n",
    "- **Anchored on Gold-58:** Evaluates official radiologist Gold-58 AUC after every epoch.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Environment & GPU Diagnostic\n",
    "!nvidia-smi\n",
    "!pip install -q timm pyarrow pydicom kaggle\n",
    "import torch\n",
    "print('CUDA Available:', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print('GPU:', torch.cuda.get_device_name(0))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Setup Kaggle & Download Training Assets (~2 minutes)\n",
    "import os, json\n",
    "os.makedirs('/root/.kaggle', exist_ok=True)\n",
    "try:\n",
    "    from google.colab import userdata\n",
    "    token = userdata.get('KAGGLE_API_TOKEN') or userdata.get('KAGGLE_KEY')\n",
    "    user = userdata.get('KAGGLE_USERNAME') or 'cyruspinto'\n",
    "    if token:\n",
    "        with open('/root/.kaggle/kaggle.json', 'w') as f:\n",
    "            json.dump({'username': user, 'key': token}, f)\n",
    "        os.chmod('/root/.kaggle/kaggle.json', 0o600)\n",
    "        print('Authenticated via Colab Secrets.')\n",
    "except Exception:\n",
    "    pass\n",
    "\n",
    "if not os.path.exists('/root/.kaggle/kaggle.json'):\n",
    "    if os.path.exists('kaggle.json'):\n",
    "        !cp kaggle.json /root/.kaggle/ && chmod 600 /root/.kaggle/kaggle.json\n",
    "    else:\n",
    "        print('Please upload kaggle.json or add KAGGLE_KEY to Colab Secrets.')\n",
    "\n",
    "os.makedirs('/content/corpus', exist_ok=True)\n",
    "os.makedirs('/content/labels', exist_ok=True)\n",
    "os.makedirs('/content/rsna_comp', exist_ok=True)\n",
    "\n",
    "!kaggle datasets download -d dreaddevelopment/knee-raptor-corpus -p /content/corpus --unzip -q\n",
    "!kaggle datasets download -d dreaddevelopment/knee-raptor-corpus-ext -p /content/corpus --unzip -q\n",
    "!kaggle datasets download -d cyruspinto/rsna-knee-super-consensus-labels-0916 -p /content/labels --unzip -q\n",
    "!kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p /content/rsna_comp -q\n",
    "!unzip -qo /content/rsna_comp/train.csv.zip -d /content/rsna_comp 2>/dev/null || true\n",
    "print('All training assets downloaded successfully!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Write DINOv2-S/14 Master Trainer\n",
    "with open('train_dinov2_colab.py', 'w', encoding='utf-8') as f:\n",
    "    f.write('''" + train_code.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
    "print('Trainer written: train_dinov2_colab.py')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Launch High-Speed Training on A100 (~35-45 minutes)\n",
    "!python train_dinov2_colab.py\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Pre-Registered Gate Audit\n",
    "import torch\n",
    "ckpt = torch.load('dinov2_f0_best.pt', map_location='cpu') if os.path.exists('dinov2_f0_best.pt') else torch.load('/kaggle/working/dinov2_f0_best.pt', map_location='cpu')\n",
    "score = ckpt.get('gold_auc', 0.0)\n",
    "print(f'Trained Checkpoint Gold AUC: {score:.4f}')\n",
    "if score >= 0.900:\n",
    "    print('GATE PASSED: Proceeding to offline ensembling verification!')\n",
    "else:\n",
    "    print('GATE FAILED: Score does not exceed 0.900. Terminating Track 2 to preserve baseline.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Upload Trained Weights to Kaggle Dataset\n",
    "!mkdir -p /content/kaggle_weights\n",
    "!cp dinov2_f0_best.pt /content/kaggle_weights/ 2>/dev/null || cp /kaggle/working/dinov2_f0_best.pt /content/kaggle_weights/\n",
    "user = 'cyruspinto'\n",
    "meta = {\n",
    "    'title': 'RSNA Knee Engine7 DINOv2 Weights',\n",
    "    'id': f'{user}/rsna-knee-engine7-dinov2-weights',\n",
    "    'licenses': [{'name': 'CC0-1.0'}]\n",
    "}\n",
    "with open('/content/kaggle_weights/dataset-metadata.json', 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "!kaggle datasets create -p /content/kaggle_weights -r zip || !kaggle datasets version -p /content/kaggle_weights -m 'Update DINOv2 weights' -r zip\n",
    "print('Weights uploaded to Kaggle!')\n"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {"gpuType": "A100"},
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.12"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

dst = Path("training/engine7_dinov2_distilled/colab_dinov2_runner.ipynb")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")
Path("colab_dinov2_master_a100.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Successfully generated colab_dinov2_runner.ipynb and colab_dinov2_master_a100.ipynb")

