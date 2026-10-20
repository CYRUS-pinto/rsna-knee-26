# ADR-037: Championship Freeze and Two-Track Engineering Protocol

## Status
**APPROVED & ACTIVE** — October 18, 2026

---

## 1. Context & Motivation
Following the completion of Candidate V32-Ablation (`57338459`, Public LB 0.956) and the systematic diagnostic audit of RadImageNet E11 (`cyruspinto/rsna-knee-e11-100-study-diagnostic`), the team identified two critical operating facts:
1. **The Championship Baseline is Secure:** Candidate V17 (`57253840`, Public LB 0.958) and Candidate V6 (`57165672`, Public LB 0.956) are verified, scored, and complete. They sit stably on the Kaggle leaderboard.
2. **The Exploration Frontier Requires Strict Sandboxing:** Exploring higher-capacity representations or pseudo-label distillation must NEVER endanger the existing locked candidates. Blind blend-weight chasing on public LB overfits to the ~390 public test studies, while unvalidated third arms (like E11 with $\rho = 0.3947$) cause catastrophic failure.

---

## 2. The Two-Track Engineering Architecture

To resolve the tension between defensive safety and offensive optimization, this decision establishes a formal **Two-Track Protocol**:

### Track 1: Fort Knox Defense (Read-Only)
- **Slot 1 (The Bedrock Anchor):** Submission **`57253840`** (Candidate V17, Public LB **0.958**).
- **Slot 2 (The Checkpoint Hedge):** Submission **`57165672`** (Candidate V6, Public LB **0.956**).
- **Invariance Rule:** These two submissions remain physically checked on the Kaggle UI. No code, weights, or files associated with V17 or V6 may be modified or overwritten.

### Track 2: Gated Empirical Offense (Sandbox on GPU)
- **Target:** Investigate orthogonal 2.5D vision architectures (e.g. DINOv2-S/14) trained on distilled consensus pseudo-labels from the 0.958 ensemble to bypass clinical report text noise.
- **Offline Gating Rule:** Any newly trained model must pass:
  1. Internal reproduction correlation: $\rho \ge 0.98$ on its own training/OOF predictions.
  2. Orthogonality test: Spearman correlation with Raptor CoAtNet $\rho \le 0.88$.
  3. Paired statistical test: 95% bootstrap confidence interval on the 58 official gold studies showing $\Delta \text{AUC} \ge +0.0030$ with lower bound $> 0$.
- **Displacement Gate:** A submission candidate produced under Track 2 is strictly forbidden from displacing Slot 2 (V6) on Kaggle unless it achieves a verified Public LB score of:
  $$\text{Public LB} \ge 0.9610$$
  If it scores $< 0.9610$, Slot 2 immediately remains V6 (`57165672`).

---

## 3. Consequences & Guarantees
- The baseline (Bronze lock, coin-flip Silver on private split) cannot be compromised by experimental failures.
- No further submissions will be burned on unvalidated public notebooks or linear blend tweaks.
- All experimental compute on Colab/Kaggle is bounded by pre-registered go/no-go gates.
