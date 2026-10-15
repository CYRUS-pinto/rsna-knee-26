# 🚀 Idiot-Proof Teammate Runbook: Training on Kaggle (Zero-Code Required)

**Send this exact message to your 4 teammates.** They do not need to write a single line of code. They just need to click 6 buttons to run a fold on their free Kaggle GPU.

---

### Instructions for Teammates (Copy & Paste to Discord / WhatsApp / Slack):

Hey team, we are currently at **Rank #1070 (Score 0.956)**. We need to train our 5 cross-validation folds in parallel. It takes **45 minutes** and uses your free Kaggle GPU quota. Here is the exact step-by-step:

#### Step 1: Open Kaggle & Create a Notebook
1. Go to: **[https://www.kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)**
2. Click the black **"Code"** tab at the top.
3. Click the blue **"New Notebook"** button on the right.

#### Step 2: Turn on the GPU
1. On the right-hand panel, look for **"Notebook options"** -> **"Accelerator"**.
2. Click the dropdown (default is "None") and select **"GPU T4 x2"** (or "GPU P100").
3. Make sure **"Internet"** is toggled **ON**.

#### Step 3: Add the Competition Dataset
1. On the top right, click **"+ Add Input"** (or the dataset icon).
2. The competition dataset `rsna-knee-abnormality-detection` should already be listed. If not, search for it and click **"Add"**.

#### Step 4: Paste This Code Into Cell 1
*(Copy and paste this exact block into the first notebook cell):*

```python
# =========================================================================
# RSNA KNEE 2026: FOLD TRAINING RUNNER
# =========================================================================
import os

# 1. Clone our team private repo
!git clone https://github.com/CYRUS-pinto/rsna-knee-26.git
%cd rsna-knee-26

# 2. Run verification
!python validation/test_cv_framework.py

# 3. Train your assigned fold (Replace FOLD_NUMBER with 0, 1, 2, 3, or 4)
# Teammate 1: FOLD=1
# Teammate 2: FOLD=2
# Teammate 3: FOLD=3
# Teammate 4: FOLD=4
FOLD_NUMBER = 1

print(f"Starting training for Fold {FOLD_NUMBER} on GPU...")
!python training/engine3_slice_transformer/model_multiplane.py
```

#### Step 5: Start the Run
1. In the top right corner, click the blue **"Save Version"** button.
2. Select **"Save & Run All (Commit)"**.
3. Click **"Save"**.
4. You can now close your browser tab! Kaggle will run it in the cloud. Once it finishes (in ~45 mins), tell Cyrus in chat.

---

### What Cyrus (Team Lead) Can Do Directly from Here
If you don't want to wait for them, **we can push and run kernels directly from this machine right now** using:
```bash
kaggle kernels push -p submissions/v13_tri_hybrid_aligned/
```
We can launch Fold 0 immediately on your Kaggle account without relying on anyone.
