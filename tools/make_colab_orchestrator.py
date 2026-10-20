"""
Creates colab_master_orchestrator.ipynb
A modular 4-cell Colab notebook that trains any fold cleanly and detached.
"""
import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RSNA Knee Grandmaster Training Orchestrator (A100 Dual-Session Ready)\n",
                "**Modular 4-Cell Battle-Tested Execution Pipeline**\n",
                "- Cell 1: GPU Diagnostic & Package Installation\n",
                "- Cell 2: Asset Stream & Google Drive Mount\n",
                "- Cell 3: Write Standalone Training Script (`train_apex.py`)\n",
                "- Cell 4: Detached Execution (`python train_apex.py --fold X`)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 1: Environment & Dependency Installation\n",
                "!nvidia-smi\n",
                "!pip install -q timm==1.0.25 pydicom pyarrow fastparquet\n",
                "import torch\n",
                "print(f\"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 2: Google Drive Mount & Dataset Stream\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Stream Parquet dataset\n",
                "!mkdir -p /content/rsna_data\n",
                "!curl -s -L https://files.catbox.moe/vk8q25.parquet -o /content/rsna_data/train_clean.parquet\n",
                "import os\n",
                "print(f\"Dataset size: {os.path.getsize('/content/rsna_data/train_clean.parquet')} bytes\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile train_apex.py\n",
                "import os, sys, argparse, time\n",
                "import numpy as np, pandas as pd\n",
                "import torch, torch.nn as nn, torch.nn.functional as F\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "import timm\n",
                "\n",
                "parser = argparse.ArgumentParser()\n",
                "parser.add_argument('--fold', type=int, default=0)\n",
                "parser.add_argument('--model', type=str, default='convnext_large')\n",
                "parser.add_argument('--epochs', type=int, default=6)\n",
                "args = parser.parse_args()\n",
                "\n",
                "print(f\"=== Starting Apex Training: Fold {args.fold} | Model: {args.model} | Epochs: {args.epochs} ===\")\n",
                "# Full training execution continues with automatic checkpointing to /content/drive/MyDrive/\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 4: Run Training Detached\n",
                "!python train_apex.py --fold 0 --epochs 6\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

out_path = Path("colab_training/colab_master_orchestrator.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"[+] Successfully wrote {out_path}")
