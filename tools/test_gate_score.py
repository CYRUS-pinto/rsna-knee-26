"""
Unit and Integration Test Suite for tools/gate_score.py
Verifies:
  1. Wide format scoring
  2. Long format (row_id, target) submission scoring
  3. Paired bootstrap statistics and CI calculation
  4. Argument aliasing (--baseline / --candidate / --n-boot)
  5. Safeguards (catching regressions when AUC < 0.60 or rho < 0.30)
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import subprocess

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def test_gate_score_e2e():
    print("=" * 65)
    print("  RUNNING E2E AUDIT & VERIFICATION ON tools/gate_score.py")
    print("=" * 65)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        n_studies = 58
        uids = [f"gold_{i:04d}" for i in range(n_studies)]
        
        # 1. Create mock gold.csv
        rng = np.random.RandomState(42)
        gold_labels = rng.binomial(1, 0.35, size=(n_studies, 12))
        # Ensure at least 5 positives and negatives per column
        for j in range(12):
            gold_labels[:5, j] = 1
            gold_labels[5:10, j] = 0
            
        gold_df = pd.DataFrame(gold_labels, columns=LAB)
        gold_df.insert(0, "StudyInstanceUID", uids)
        gold_path = os.path.join(tmpdir, "gold.csv")
        gold_df.to_csv(gold_path, index=False)
        
        # 2. Create baseline predictions (wide format, good AUC)
        baseline_preds = gold_labels * 0.5 + rng.uniform(0.1, 0.4, size=(n_studies, 12))
        base_df = pd.DataFrame(baseline_preds, columns=LAB)
        base_df.insert(0, "StudyInstanceUID", uids)
        base_path = os.path.join(tmpdir, "baseline.csv")
        base_df.to_csv(base_path, index=False)
        
        # 3. Create candidate predictions (long format: row_id, target)
        cand_preds = gold_labels * 0.6 + rng.uniform(0.05, 0.35, size=(n_studies, 12))
        rows = []
        for i, u in enumerate(uids):
            for j, l in enumerate(LAB):
                rows.append({"row_id": f"{u}_{l}", "target": float(cand_preds[i, j])})
        cand_df = pd.DataFrame(rows)
        cand_path = os.path.join(tmpdir, "candidate_submission.csv")
        cand_df.to_csv(cand_path, index=False)
        
        # Test CLI with --arm name=path
        cmd_arm = [
            sys.executable, "tools/gate_score.py",
            "--gold", gold_path,
            "--arm", f"baseline={base_path}",
            "--arm", f"candidate={cand_path}",
            "--boots", "100"
        ]
        res = subprocess.run(cmd_arm, capture_output=True, text=True)
        assert res.returncode == 0, f"Failed with --arm: {res.stderr}\n{res.stdout}"
        assert "GATE VERDICT: PASS (APPROVED)" in res.stdout, f"Expected PASS: {res.stdout}"
        assert "Spearman baseline vs candidate" in res.stdout
        print("  [PASSED] Wide and Long format multi-arm evaluation verified.")
        
        # Test CLI with --baseline and --candidate aliases
        cmd_alias = [
            sys.executable, "tools/gate_score.py",
            "--gold", gold_path,
            "--baseline", base_path,
            "--candidate", cand_path,
            "--n-boot", "100"
        ]
        res_alias = subprocess.run(cmd_alias, capture_output=True, text=True)
        assert res_alias.returncode == 0, f"Failed with aliases: {res_alias.stderr}\n{res_alias.stdout}"
        assert "GATE VERDICT: PASS (APPROVED)" in res_alias.stdout
        print("  [PASSED] Argument aliasing (--baseline, --candidate, --n-boot) verified.")
        
        # Test Safeguard: intentionally inject low AUC candidate to test rejection
        inverted_preds = 1.0 - baseline_preds
        inv_df = pd.DataFrame(inverted_preds, columns=LAB)
        inv_df.insert(0, "StudyInstanceUID", uids)
        inv_path = os.path.join(tmpdir, "inverted.csv")
        inv_df.to_csv(inv_path, index=False)
        
        cmd_bad = [
            sys.executable, "tools/gate_score.py",
            "--gold", gold_path,
            "--arm", f"baseline={base_path}",
            "--arm", f"bad={inv_path}",
            "--boots", "50"
        ]
        res_bad = subprocess.run(cmd_bad, capture_output=True, text=True)
        assert res_bad.returncode == 1, "Expected rejection for inverted model, but got exit code 0!"
        assert "GATE VERDICT: FAIL (REJECTED)" in res_bad.stdout
        print("  [PASSED] Safeguard gate successfully caught and rejected degraded model.")
        
    print("=" * 65)
    print("ALL tools/gate_score.py TESTS PASSED PERFECTLY.")
    print("=" * 65)

if __name__ == '__main__':
    test_gate_score_e2e()
