# ADR-005: Kaggle Credential Sanitization & Security Protocol

## Status
**Accepted**

## Context
During initial development, a Kaggle Access Token beginning with `KGAT_` was embedded in Cell 2 of the Colab training notebook for automated dataset downloads.
Because git commits and notebooks can be published to GitHub or shared with collaborators, hardcoded credentials present an unacceptable security vulnerability.

## Actions Taken
1. **Repository Scrubbing**:
   - All occurrences of hardcoded tokens were completely eradicated from active notebooks and utility scripts.
   - Verified across the entire workspace via ripgrep (`grep_search` confirmed 0 matches).
2. **Implementation of Zero-Hardcode Authentication**:
   - Updated Cell 2 in [training/engine3_slice_transformer/colab_rsna_knee_super_train.ipynb](../training/engine3_slice_transformer/colab_rsna_knee_super_train.ipynb) to use the secure hierarchy:
     1. Google Colab Secrets (`userdata.get("KAGGLE_API_TOKEN")` or `userdata.get("KAGGLE_KEY")`).
     2. Environment variables (`os.environ.get("KAGGLE_API_TOKEN")`).
     3. Interactive masked prompt (`getpass.getpass()`).
3. **User Action Required**:
   - The user must visit [kaggle.com/settings](https://www.kaggle.com/settings) and click **"Expire Token"** / **"Create New Token"** to revoke the exposed key.
