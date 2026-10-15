"""Step 2 of the Gold-58 gate. Scores arm outputs (one CSV per model, 58 rows) against gold.
Usage:
  python gate_score.py --gold gate_root/gold.csv --arm raptor=raptor_sub.csv --arm convnext=ours_sub.csv \
      [--arm colab=colab.csv] [--log run.log] [--cw "A=0.6,B=0.4,C=0.5"] [--boots 2000]
First --arm is arm0. --cw gives arm0's weight per cluster for a 2-arm cluster blend (arm0 vs arm1).
This is a BUG DETECTOR and a noisy comparison tool, not an accuracy estimate."""
import argparse, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Hypothesis, not fact. Edit if you disagree; keep it to <=3 clusters.
CLUSTERS = {"A_tears": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
            "B_cartilage": ["Medial OA", "Lateral OA", "PF OA"],
            "C_fluid_bone": ["Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]}
IDX = {l: i for i, l in enumerate(LAB)}
MIN_GOLD_AUC, MIN_RHO = 0.60, 0.30

def rankpct(x):
    return x.argsort(0).argsort(0).astype(float) / max(1, x.shape[0] - 1)

def aucs(y, s):
    return np.array([roc_auc_score(y[:, j], s[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan
                     for j in range(y.shape[1])])

def macro(y, s, cols=None):
    a = aucs(y, s)
    if cols is not None:
        a = a[[IDX[c] for c in cols]]
    return np.nanmean(a)

def boot_diff(y, a, b, cols=None, n=2000, seed=0, ci=95.0):
    rng = np.random.default_rng(seed); d = np.empty(n)
    for k in range(n):
        i = rng.integers(0, len(y), len(y))
        d[k] = macro(y[i], a[i], cols) - macro(y[i], b[i], cols)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return d.mean(), np.percentile(d, [lo, hi]), (d > 0).mean()

def load(path, uids):
    d = pd.read_csv(path); d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
    d = d.set_index("StudyInstanceUID")
    missing = [u for u in uids if u not in d.index]
    if missing: sys.exit(f"FAIL {path}: {len(missing)} gold studies missing")
    x = d.loc[uids, LAB].to_numpy(float)
    if not np.isfinite(x).all(): sys.exit(f"FAIL {path}: non-finite values")
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True); ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--log"); ap.add_argument("--cw"); ap.add_argument("--boots", type=int, default=2000)
    a = ap.parse_args()
    g = pd.read_csv(a.gold); g["StudyInstanceUID"] = g["StudyInstanceUID"].astype(str)
    uids = g["StudyInstanceUID"].tolist(); y = g[LAB].to_numpy(int)
    names, S = [], []
    for spec in a.arm:
        nm, p = spec.split("=", 1); names.append(nm); S.append(load(p, uids))
    fails = []

    if a.log:
        txt = open(a.log, errors="ignore").read()
        nfb = len(re.findall(r"FALLBACK", txt))
        print(f"fallback lines in log: {nfb}")
        if nfb: fails.append(f"{nfb} fallback line(s) in log")

    print("\nPer-class gold AUC")
    tab = pd.DataFrame({n: aucs(y, s) for n, s in zip(names, S)}, index=LAB)
    print(tab.round(3).to_string()); print("macro:", {n: round(float(np.nanmean(tab[n])), 4) for n in names})
    for n in names:
        bad = tab.index[tab[n] < MIN_GOLD_AUC].tolist()
        if bad: fails.append(f"{n}: gold AUC < {MIN_GOLD_AUC} on {bad}")

    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            rho = [spearmanr(S[i][:, k], S[j][:, k]).correlation for k in range(12)]
            bad = [LAB[k] for k, r in enumerate(rho) if not np.isfinite(r) or r < MIN_RHO]
            print(f"spearman {names[i]} vs {names[j]}: min {np.nanmin(rho):.2f} mean {np.nanmean(rho):.2f}")
            if bad: fails.append(f"{names[i]} vs {names[j]} rho < {MIN_RHO} on {bad}")

    R = [rankpct(s) for s in S]
    cand = {n: r for n, r in zip(names, R)}
    cand["uniform"] = np.mean(R, axis=0)
    if a.cw and len(R) >= 2:
        w = {k.strip(): float(v) for k, v in (x.split("=") for x in a.cw.split(","))}
        W = np.zeros(12)
        for cl, cols in CLUSTERS.items():
            key = cl.split("_")[0]
            for c in cols: W[IDX[c]] = w.get(key, 0.5)
        cand["cluster"] = W * R[0] + (1 - W) * R[1]
    print("\nCandidates (macro gold AUC)")
    for k, v in cand.items(): print(f"  {k:10s} {macro(y, v):.4f}")

    base = cand["uniform"]
    print(f"\nPaired bootstrap vs uniform ({a.boots} resamples)")
    for k, v in cand.items():
        if k == "uniform": continue
        m, ci, p = boot_diff(y, v, base, n=a.boots)
        print(f"  {k:10s} d={m:+.4f} 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] P(d>0)={p:.3f}"
              f"  -> {'SUPPORTED' if ci[0] > 0 else ('WORSE' if ci[1] < 0 else 'not distinguishable from uniform')}")
        if k == "cluster":
            for cl, cols in CLUSTERS.items():
                m, ci, p = boot_diff(y, v, base, cols, n=a.boots, ci=100 - 5 / 3)  # Bonferroni, 3 clusters
                print(f"      {cl:13s} d={m:+.4f} 98.3%CI [{ci[0]:+.4f},{ci[1]:+.4f}]")

    print("\nGATE:", "FAIL" if fails else "PASS")
    for f in fails: print("  -", f)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
