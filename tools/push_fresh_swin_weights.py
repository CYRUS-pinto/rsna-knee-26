import os, shutil, json, time
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("[+] Authenticated with Kaggle successfully.")

# 1. Update Fold 2
f2_src = r"C:\Users\Cyrus\Downloads\engine10_swin_384_fold2_best.pt"
f2_dir = "tmp_kaggle_push_fold2"
os.makedirs(f2_dir, exist_ok=True)
shutil.copy(f2_src, os.path.join(f2_dir, "engine10_swin_384_fold2_best.pt"))

meta_f2 = {
    "title": "RSNA Knee Engine 10 Fold 2 Swin-Base Weights",
    "id": "cyruspinto/rsna-knee-engine10-fold2-weights",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(f2_dir, "dataset-metadata.json"), "w") as f:
    json.dump(meta_f2, f, indent=2)

print("[*] Uploading fresh Fold 2 weights (332 MB) to Kaggle...")
api.dataset_create_version(f2_dir, "Update Fold 2 Swin-Base Weights (Loss: 0.4078)", dir_mode="zip")
print("[+] Successfully published Fold 2 to cyruspinto/rsna-knee-engine10-fold2-weights!")

# 2. Update Fold 3
f3_src = r"C:\Users\Cyrus\Downloads\engine10_swin_384_fold3_best.pt"
f3_dir = "tmp_kaggle_push_fold3"
os.makedirs(f3_dir, exist_ok=True)
shutil.copy(f3_src, os.path.join(f3_dir, "engine10_swin_384_fold3_best.pt"))

meta_f3 = {
    "title": "RSNA Knee Engine 10 Fold 3 Swin-Base Weights",
    "id": "cyruspinto/rsna-knee-engine10-fold3-weights",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(f3_dir, "dataset-metadata.json"), "w") as f:
    json.dump(meta_f3, f, indent=2)

print("[*] Uploading fresh Fold 3 weights (332 MB) to Kaggle...")
api.dataset_create_version(f3_dir, "Update Fold 3 Swin-Base Weights (Loss: 0.4182)", dir_mode="zip")
print("[+] Successfully published Fold 3 to cyruspinto/rsna-knee-engine10-fold3-weights!")
