"""Step 1 of the Gold-58 gate. Run inside a Kaggle notebook with the competition attached.
Builds /kaggle/working/gate_root that looks like a test root but contains the 58 gold studies.
Override the competition path with env COMP (used only for local testing)."""
import os, pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def find_comp():
    if os.environ.get("COMP"):
        return os.environ["COMP"]
    for b in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.exists(b + "/train.csv"):
            return b
    raise SystemExit("competition folder not found under /kaggle/input")

comp = find_comp()
root = os.environ.get("GATE_ROOT", "/kaggle/working/gate_root")
os.makedirs(root, exist_ok=True)

tr = pd.read_csv(comp + "/train.csv", usecols=["StudyInstanceUID"] + LAB)
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = tr[tr["ACL"].notna()].reset_index(drop=True)
uids = gold["StudyInstanceUID"].tolist()
assert len(uids) == 58, f"expected 58 gold studies, found {len(uids)}"

ts = pd.read_csv(comp + "/train_series.csv")
ts["StudyInstanceUID"] = ts["StudyInstanceUID"].astype(str)
ts = ts[ts["StudyInstanceUID"].isin(uids)]
assert ts["StudyInstanceUID"].nunique() == 58, "some gold studies have no series rows"

pd.DataFrame({"StudyInstanceUID": uids}).to_csv(root + "/test.csv", index=False)
ts.to_csv(root + "/test_series.csv", index=False)
pd.DataFrame([[u] + [0.5] * 12 for u in uids],
             columns=["StudyInstanceUID"] + LAB).to_csv(root + "/sample_submission.csv", index=False)
gold.astype({c: int for c in LAB}).to_csv(root + "/gold.csv", index=False)

link = root + "/test_series"
if not os.path.lexists(link):
    os.symlink(comp + "/train_series", link)   # same layout: <study>/<series>/*.dcm
print("gate root ready:", root, "| studies", len(uids), "| series rows", len(ts))
print("NEXT: set os.environ['GATE_ROOT'] to this path BEFORE running the pipeline cells,")
print("      and make find_test_root() (and any other root lookup) return it.")
