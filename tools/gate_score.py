"""Step 2 of the Gold-58 gate. Scores arm outputs (one CSV per model, 58 rows) against gold.
Supports both wide format (StudyInstanceUID, ACL, ...) and long format (row_id, target).
Supports both --arm name=path and --baseline/--candidate CLI interfaces.

Usage:
  python tools/gate_score.py --gold gate_root/gold.csv --arm raptor=raptor_sub.csv --arm convnext=ours_sub.csv [--boots 2000]
  OR:
  python tools/gate_score.py --gold gate_root/gold.csv --baseline baseline.csv --candidate candidate.csv [--n-boot 2000]
"""
import argparse, re, sys, os
import numpy as np, pandas as pd
try:
    from scipy.stats import spearmanr
except ImportError:
    def spearmanr(a, b):
        class Result:
            def __init__(self, c): self.correlation = c
        ra = pd.Series(a).rank().values
        rb = pd.Series(b).rank().values
        c = np.corrcoef(ra, rb)[0, 1]
        return Result(c)

def exact_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, bool)
    y_score = np.asarray(y_score, float)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method='average').to_numpy()
    sum_ranks_pos = ranks[:n_pos].sum()
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CLUSTERS = {"A_tears": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
            "B_cartilage": ["Medial OA", "Lateral OA", "PF OA"],
            "C_fluid_bone": ["Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]}
IDX = {l: i for i, l in enumerate(LAB)}
MIN_GOLD_AUC, MIN_RHO = 0.60, 0.30

def rankpct(x):
    return x.argsort(0).argsort(0).astype(float) / max(1, x.shape[0] - 1)

def aucs(y, s):
    res = []
    for j in range(y.shape[1]):
        yj, sj = y[:, j], s[:, j]
        if 0 < yj.sum() < len(yj) and np.std(sj) > 1e-9:
            try:
                res.append(exact_roc_auc(yj, sj))
            except Exception:
                res.append(np.nan)
        else:
            res.append(np.nan)
    return np.array(res)

def macro(y, s, cols=None):
    a = aucs(y, s)
    if cols is not None:
        a = a[[IDX[c] for c in cols]]
    valid = a[~np.isnan(a)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.50

def boot_diff(y, a, b, cols=None, n=2000, seed=0, ci=95.0):
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for k in range(n):
        i = rng.integers(0, len(y), len(y))
        d[k] = macro(y[i], a[i], cols) - macro(y[i], b[i], cols)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(d.mean()), np.percentile(d, [lo, hi]), float((d > 0).mean())

def load(path, uids):
    d = pd.read_csv(path)
    # Check if long format: row_id, target
    if "row_id" in d.columns and ("target" in d.columns or "Probability" in d.columns):
        val_col = "target" if "target" in d.columns else "Probability"
        d[["StudyInstanceUID", "label"]] = d["row_id"].str.rsplit("_", n=1, expand=True)
        d = d.pivot(index="StudyInstanceUID", columns="label", values=val_col).reset_index()
    
    # Normalize study column name
    if "study_id" in d.columns and "StudyInstanceUID" not in d.columns:
        d["StudyInstanceUID"] = d["study_id"]
        
    if "StudyInstanceUID" not in d.columns:
        sys.exit(f"FAIL {path}: missing StudyInstanceUID / row_id / study_id column")
        
    d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
    d = d.set_index("StudyInstanceUID")
    
    missing = [u for u in uids if u not in d.index]
    if missing:
        sys.exit(f"FAIL {path}: {len(missing)} gold studies missing (e.g. {missing[:3]})")
        
    for c in LAB:
        if c not in d.columns:
            sys.exit(f"FAIL {path}: missing pathology column '{c}'")
            
    x = d.loc[uids, LAB].to_numpy(float)
    if not np.isfinite(x).all():
        sys.exit(f"FAIL {path}: non-finite values (NaN / Inf) detected")
    return x

def main():
    ap = argparse.ArgumentParser(description="Gold-58 Gate Scoring & Statistical Significance Tool")
    ap.add_argument("--gold", required=True, help="Path to gold.csv ground truth")
    ap.add_argument("--arm", action="append", default=[], help="Model arm: name=path.csv (can be repeated)")
    ap.add_argument("--baseline", help="Alternative CLI: path to baseline CSV")
    ap.add_argument("--candidate", help="Alternative CLI: path to candidate CSV")
    ap.add_argument("--log", help="Path to run.log to scan for fallback occurrences")
    ap.add_argument("--cw", help="Cluster blend weights: A=0.6,B=0.4,C=0.5")
    ap.add_argument("--boots", "--n-boot", dest="boots", type=int, default=2000, help="Number of bootstrap iterations (default: 2000)")
    
    a = ap.parse_args()
    
    # Handle --baseline and --candidate sugar
    if a.baseline and a.candidate:
        if not any("baseline=" in s for s in a.arm):
            a.arm.insert(0, f"baseline={a.baseline}")
        if not any("candidate=" in s for s in a.arm):
            a.arm.append(f"candidate={a.candidate}")
            
    if not a.arm:
        sys.exit("Error: Must specify at least one --arm name=path OR both --baseline and --candidate.")
        
    g = pd.read_csv(a.gold)
    if "study_id" in g.columns and "StudyInstanceUID" not in g.columns:
        g["StudyInstanceUID"] = g["study_id"]
    g["StudyInstanceUID"] = g["StudyInstanceUID"].astype(str)
    uids = g["StudyInstanceUID"].tolist()
    y = g[LAB].to_numpy(int)
    
    names, S = [], []
    for spec in a.arm:
        if "=" in spec:
            nm, p = spec.split("=", 1)
        else:
            nm, p = os.path.splitext(os.path.basename(spec))[0], spec
        names.append(nm)
        S.append(load(p, uids))
        
    fails = []

    if a.log:
        txt = open(a.log, errors="ignore").read()
        nfb = len(re.findall(r"FALLBACK", txt))
        print(f"Fallback lines in log: {nfb}")
        if nfb:
            fails.append(f"{nfb} fallback line(s) detected in execution log")

    print("=" * 65)
    print("  GOLD-58 OFFLINE EVALUATION GATE (N=58 Ground-Truth Studies)")
    print("=" * 65)
    print("\n[1] Per-Class Gold AUC:")
    tab = pd.DataFrame({n: aucs(y, s) for n, s in zip(names, S)}, index=LAB)
    print(tab.round(3).to_string())
    print("\nMacro-AUC:", {n: round(float(np.nanmean(tab[n])), 4) for n in names})
    
    for n in names:
        bad = tab.index[tab[n] < MIN_GOLD_AUC].tolist()
        if bad:
            fails.append(f"{n}: gold AUC < {MIN_GOLD_AUC} on {bad}")

    print("\n[2] Inter-Arm Spearman Correlation (min_rho >= 0.30):")
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            rho = [spearmanr(S[i][:, k], S[j][:, k]).correlation for k in range(12)]
            bad = [LAB[k] for k, r in enumerate(rho) if not np.isfinite(r) or r < MIN_RHO]
            print(f"  Spearman {names[i]} vs {names[j]}: min {np.nanmin(rho):.2f} | mean {np.nanmean(rho):.2f}")
            if bad:
                fails.append(f"{names[i]} vs {names[j]} rho < {MIN_RHO} on {bad}")

    R = [rankpct(s) for s in S]
    cand = {n: r for n, r in zip(names, R)}
    cand["uniform"] = np.mean(R, axis=0)
    
    if a.cw and len(R) >= 2:
        w = {k.strip(): float(v) for k, v in (x.split("=") for x in a.cw.split(","))}
        W = np.zeros(12)
        for cl, cols in CLUSTERS.items():
            key = cl.split("_")[0]
            for c in cols:
                W[IDX[c]] = w.get(key, 0.5)
        cand["cluster"] = W * R[0] + (1 - W) * R[1]
        
    print("\n[3] Candidate Configurations (Macro Gold AUC):")
    for k, v in cand.items():
        print(f"  {k:12s}: {macro(y, v):.4f}")

    base = cand["uniform"]
    print(f"\n[4] Paired Bootstrap Significance vs Uniform Baseline ({a.boots} resamples):")
    for k, v in cand.items():
        if k == "uniform":
            continue
        m, ci, p = boot_diff(y, v, base, n=a.boots)
        status = "SUPPORTED (CI > 0)" if ci[0] > 0 else ("WORSE (CI < 0)" if ci[1] < 0 else "Neutral (Includes 0)")
        print(f"  {k:12s} d={m:+.4f} | 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] | P(d>0)={p:.3f} -> {status}")
        if k == "cluster":
            for cl, cols in CLUSTERS.items():
                m_c, ci_c, p_c = boot_diff(y, v, base, cols, n=a.boots, ci=100 - 5 / 3)
                print(f"      {cl:13s} d={m_c:+.4f} | 98.3% CI [{ci_c[0]:+.4f}, {ci_c[1]:+.4f}]")

    print("\n" + "=" * 65)
    print("GATE VERDICT:", "FAIL (REJECTED)" if fails else "PASS (APPROVED)")
    print("=" * 65)
    for f in fails:
        print("  - REASON:", f)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
