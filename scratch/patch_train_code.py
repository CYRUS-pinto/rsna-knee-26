with open('scratch/kernels/dread_train/knee-mri-training-the-twelve-finding-model.py', encoding='utf-8') as f:
    code = f.read()

target = '''    _lab = a.labels or _find("labels_llm_soft.parquet")
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    soft = pd.read_parquet(_lab)'''

replacement = '''    _lab = a.labels if (a.labels and os.path.exists(a.labels)) else None
    if not _lab:
        _lab = _find("SUPER_CONSENSUS_labels.parquet") or _find("SUPER_CONSENSUS_labels.csv") or _find("labels_llm_soft.parquet")
    if not _lab:
        for root, dirs, files in os.walk("/kaggle/input"):
            for fn in files:
                if any(k in fn.lower() for k in ["consensus", "soft", "label"]) and fn.endswith((".parquet", ".csv")):
                    cand = os.path.join(root, fn)
                    try:
                        t = pd.read_parquet(cand) if cand.endswith(".parquet") else pd.read_csv(cand)
                        if set(LAB).issubset(t.columns):
                            _lab = cand
                            print(f"[auto-detect] Found matching label file: {_lab}", flush=True)
                            break
                    except Exception:
                        pass
            if _lab:
                break
    if not _lab or not os.path.exists(_lab):
        print(f"DEBUG: Searched /kaggle/input directories: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'no /kaggle/input'}", flush=True)
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    print(f"[labels] Ingesting training targets from: {_lab}", flush=True)
    soft = pd.read_parquet(_lab) if _lab.endswith(".parquet") else pd.read_csv(_lab)'''

assert target in code, "Target string not found in knee-mri-training-the-twelve-finding-model.py"
code = code.replace(target, replacement)

with open('scratch/kernels/dread_train/knee-mri-training-the-twelve-finding-model.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Successfully patched training script with robust auto-discovery!")
