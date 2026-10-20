"""
Uploads the trained Engine 13 Swin-Base-384 checkpoint (0.8577 Gold AUC)
from C:\\Users\\Cyrus\\Downloads\\engine13_swin_base_best.pt to Kaggle dataset
cyruspinto/rsna-knee-engine13-swin-base-weights.
"""
import os, shutil, json
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

DATASET_SLUG = "cyruspinto/rsna-knee-engine13-swin-base-weights"
UPLOAD_DIR = r"c:\Users\Cyrus\Downloads\rsna knee fip\artifacts\engine13_upload"
SRC_WEIGHTS = r"C:\Users\Cyrus\Downloads\engine13_swin_base_best.pt"

def main():
    kaggle.api.authenticate()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    dest_path = os.path.join(UPLOAD_DIR, "engine13_swin_base_best.pt")
    print(f"[*] Copying {SRC_WEIGHTS} -> {dest_path}...")
    shutil.copyfile(SRC_WEIGHTS, dest_path)
    
    meta_path = os.path.join(UPLOAD_DIR, "dataset-metadata.json")
    meta = {
        "title": "RSNA Knee Engine 13 Swin-Base Weights",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[*] Uploading new version to Kaggle dataset {DATASET_SLUG}...")
    kaggle.api.dataset_create_version(
        folder=UPLOAD_DIR,
        version_notes="Trained Swin-Base-384 Asymmetric Focal MIL (8 epochs, 0.8577 Gold AUC, 332 MB)",
        delete_old_versions=False
    )
    print(f"[+] Successfully pushed new version to {DATASET_SLUG}!")

if __name__ == "__main__":
    main()
