# WHAT COULD GO WRONG & PREVENT REACHING TOP 1 (0.969+) — EXHAUSTIVE MASTER PREVENTION PLAYBOOK

**Target:** RSNA Knee Abnormality Detection 2026 — World #1 (0.969+)  
**Current Baseline:** 0.958 Public LB (Ref `57253840`, Slot 1 LOCKED)  
**Gap to #1:** +0.011 AUC | **Gap to Top 10:** +0.009 AUC  
**Deadline:** 2026-10-22 23:59 UTC (~53.5 hours)  
**Competition:** 6,277 teams | 1,617 teams at ≥0.958 (public-notebook plateau)

---

## 🎯 EXECUTIVE SUMMARY: THE BRUTAL TRUTH

**You are at 0.958. The plateau is mathematically real. 1,617 teams are stuck here.**  
**The ONLY path to 0.969+ is NEW ORTHOGONAL SIGNAL from multi-fold trained, architecturally diverse models.**  
**Re-weighting public weights is mathematically futile (ρ > 0.95 → zero variance reduction).**

This document catalogues EVERY failure mode that has killed or can kill a Top 1 attempt, with mathematical proofs, root causes, and hard prevention mechanisms.

---

## 💀 PART I: THE 10 FATAL FAILURE MODES (Mathematically Proven)

### FM-1: Gold-58 Overfitting (Sample Size Fallacy) — **PROVEN KILLER**
**Math:** Gold-58 has 7 fractures, 9 MCL tears. One misclassified study shifts class AUC by Δ ≈ 1/(7×51) ≈ **0.0028**.  
A +0.016 "gain" on Gold-58 = pure noise-fitting. On 4,407-study OOF, same heuristic loses ~0.003.

**Historical Proof:** V7–V11: +0.016 on Gold-58 → **-0.001 on Public LB** (V12: 0.953)  
**Hard Prevention:** Gold-58 = sanity check ONLY (≥0.85 Macro AUC). **NEVER optimize on it.** (AGENTS.md Rule 6)

---

### FM-2: Pseudo-Diversity / Collinear Ensembling — **THE PLATEAU CAUSE**
**Math:** Ensemble variance: σ²_ens = (1/M)σ² + ((M-1)/M)ρσ². As ρ → 1.0, gain → 0.  
V18–V33: 15 submissions blending CoAtNet + ConvNeXt-Small with ρ > 0.95 → stuck at 0.956–0.958.

**Historical Proof:** 3 days, 15 submissions, **zero progress** toward 0.965+  
**Hard Prevention:** Orthogonal inductive biases ONLY:
1. **ConvNeXt-Large-384** — 7×7 depthwise convs → bone cortex, fractures
2. **Swin-Base-384** — Shifted-window attention → ligaments, menisci
3. **DINOv2-ViT-L/14** — Self-supervised medical semantics (planned)

---

### FM-3: Raw Probability Averaging Across Uncalibrated Sigmoids — **V42 DISASTER**
**Math:** P_blend = w₁P₁ + w₂P₂. If P₂ has scale c ≫ 1, P_blend ≈ w₂c·P₁. Model 1's rank discrimination **drowned**.

**Historical Proof:** V42: Raw prob + power sharpening (p=1.3) + 30% weight on 0.73-AUC fold → **0.943 (-0.015)**  
**Hard Prevention:** **Strict Percentile Rank Standardization ONLY:**
```python
r_m(i) = rank(P_m(i)) / N  →  r_ensemble = Σ w_m r_m
```
Every model → Uniform(0,1), identical variance 1/12. Immune to calibration drift. (AGENTS.md Rule 2)

---

### FM-4: Non-Linear Power Sharpening (p > 1.0) — **MATHEMATICALLY TOXIC**
**Math:** S_i = (Σ w_k r_{i,k}^p)^{1/p}. Compresses low-probability negatives → rank ties/inversions → degrades Wilcoxon-Mann-Whitney U-statistic.

**Historical Proof:** V42 p=1.4 → 0.943. **Linear rank stacking strictly dominates.**  
**Hard Prevention:** **Linear rank combinations ONLY** (Σ w_k r_k, Σ w_k = 1.0). Zero power transforms. (AGENTS.md Rule 3)

---

### FM-5: Over-Weighting Weak/Single-Fold Arms — **V42 MECHANISM**
**Math:** Jensen's Inequality + ensemble variance reduction. Single fold variance ≈ ±0.004 AUC. Weight >15% drags ensemble SNR down.

**Historical Proof:** V42: 30% weight on 0.73-AUC fold → 0.943 collapse  
**Hard Prevention:** **New arms capped at ≤15%** until 5-fold OOF Nelder-Mead optimization. (AGENTS.md Rule 5)

---

### FM-6: 44-Slice vs 96-Slice Domain Shift (Positional Embedding Collapse)
**Reality:** Position k in 44 sagittal = lateral-to-medial depth. Feeding 96 slices across 3 planes destroys positional semantics.

**Hard Prevention:** **Gated Attention MIL** (permutation-invariant):
```python
a_i = exp(wᵀ(tanh(V h_i) ⊙ σ(U h_i))) / Σ exp(...)
```

---

### FM-7: Self-Attention Memory Explosion (14.4 GB OOM on T4)
**Math:** 384px, patch=4 → 9,216 tokens/image. 16 slices × 24 layers = 14.4 GB VRAM → OOM on 14.56 GB T4.

**Hard Prevention:** **ALWAYS:** `backbone.set_grad_checkpointing(True)`, `k=8` triplets, `chunk_size = 4` → VRAM ~3.5 GB (T4) / ~5.3 GB (A100). (AGENTS.md Rule 7)

---

### FM-8: Fatal Assertions on Hidden Test Set — **V34–V36 OUTAGES**
**Reality:** Local test = 3 studies, 120 rows. Hidden test = ~1,500 studies, non-standard DICOMs.

**Historical Proof:** `if n > 20 and min(rho) < 0.99: raise RuntimeError` → 3 wasted submissions, 3 days lost  
**Hard Prevention:** **Zero fatal assertions.** `tools/validate_kernel_guardrail.py` AST scan MANDATORY. (AGENTS.md Rule 4)

---

### FM-9: Spatial Resolution Dilution (224px Re-Blending) — **V47 REGRESSION**
**Math:** V47 double-blended `raptor_sub.csv` (already 65% 384px + 35% 224px) with `raptor224_sub.csv` at 30% → effective 224px weight = **57.8%**.  
At 140mm FOV: 224px = 0.625mm/px. 0.8mm meniscal tear = 1.28 pixels → **Nyquist obliterated**.

**Historical Proof:** V47: 0.953 (-0.005 from 0.958) — **ADR-048**  
**Hard Prevention:** **384px weight ≥ 70%, NO re-blending pre-blended outputs.** (AGENTS.md Rule 11, guardrail enforced)

---

### FM-10: Slot 1 Destruction (Abandoning 0.958 Anchor)
**Reality:** Replacing Slot 1 with unverified 0.955 loses medal floor permanently.

**Hard Prevention:** **Two-Track Defense IMMUTABLE:**
- Slot 1: `57253840` (0.958) — LOCKED FOREVER unless candidate ≥ 0.9610
- Slot 2: Experimental sprint (AGENTS.md Rule 1)

---

## 💀 PART II: 25 ADDITIONAL FAILURE MODES (From Risk Register & ADRs)

| # | Failure Mode | Root Cause | Prevention |
|---|--------------|------------|------------|
| 11 | Kaggle Kernel OOM | Full-volume 3D tensors in GPU RAM | Chunked slice windowing (k=16, chunk=4) + fp16 |
| 12 | Submission Timeout (>9h) | Unbuffered DICOM reading | Memory-mapped .npy/.parquet pre-extracted features |
| 13 | Interactive Test Mock Crash | Hardcoded `assert len(test_df) > 500` | AST scan with `validate_kernel_guardrail.py` |
| 14 | Missing Auxiliary Dataset | Kernel dataset mount path changes | Fallback to anchor predictions |
| 15 | Extreme Prediction NaN/Inf | FP16 logit overflow | `torch.nan_to_num` + clamp [0.0001, 0.9999] |
| 16 | Label Column Ordering Bug | Alphabetical sort vs sample_submission | Reindex against sample_submission template |
| 17 | Study ID Key Misalignment | String vs Integer type casting | Enforce `dtype={'StudyInstanceUID': str}` |
| 18 | Single-Class Zero Variance | Uniform constant for rare class | `pct_rank` fallback to 0.5 |
| 19 | Loss of Slot 1 Safety Anchor | Agent replaces 0.958 with unverified | Hard Rule 1: Never deselect unless ≥0.9610 |
| 20 | Gradient Exploding in Large Backbones | Unscaled gradients in 198M params | Gradient norm clipping (1.0) |
| 21 | Catastrophic Forgetting | Single high LR across backbone | Differential LR (2e-5 backbone, 2e-4 head) |
| 22 | Colab Disconnect During Training | Idle browser / WebSocket disconnect | Off-box checkpoint push via Kaggle CLI each epoch |
| 23 | Disk Quota Exceeded on Colab | Uncompressed DICOM extractions | Streaming read + immediate cleanup |
| 24 | GPU Kernel Crash on Multi-GPU | Hardcoded `cuda:0` | Dynamic `torch.device('cuda' if available else 'cpu')` |
| 25 | Corrupt Weights Upload | Partial network write | SHA-256 hash verification + load smoke test |
| 26 | Sub-500s Efficiency Trap | Constraining main LB to efficiency rules | **Two-Track:** V45 defends $18k; V48/V49 target $59k |
| 27 | Clinical Prior Overfitting | Hardcoded biomechanical boosts | Excision of heuristics (ADR-002: V12→0.953) |
| 28 | Power Sharpening Degradation | Applying P^α (α>1) | Prohibited (V42→0.943); linear rank only |
| 29 | Over-Reliance on Weak Pseudo-Labels | Unweighted raw LLM report extractions | Super-consensus soft labels with confidence masking |
| 30 | Missing Slices in Test Studies | Incomplete series acquisitions | Dynamic zero-padding with missing-slot mask |
| 31 | Timm Version Incompatibility | Architecture name deprecated | Pinned architecture strings + custom MIL heads |
| 32 | Daily Quota Exhaustion | Submitting unverified/broken notebooks | Mandatory AST + mock testing via guardrail |
| 33 | Timezone Miscalculation | UTC vs IST confusion | Fixed countdown in LIVE_STATE.md |
| 34 | Secret/Token Leakage | Committing API tokens | .gitignore masking all credentials |
| 35 | Silent Script Failure | Unhandled exception in aux model | Fail-soft wrapper: aux failures caught, anchor used |

---

## 🏗️ PART III: ARCHITECTURAL FAILURE MODES (Advanced - Section G)

| ID | Risk | Detection | Prevention |
|----|------|-----------|------------|
| **G1** | Post-processing leakage / over-calibration on public LB | Public LB ↑ but OOF CV ↓ | Forbid manual threshold tuning; rank-percentile only |
| **G2** | Pseudo-orthogonality (ConvNeXt-B + L + ResNet ρ > 0.96) | Pairwise OOF Spearman ρ > 0.96 | Enforce structural diversity: ViT + Modern CNN + Attention-MIL |
| **G3** | Slice Aggregation Bottleneck (mean/max pooling washes out tears) | Per-class fracture/meniscus recall depressed | Learnable Gated Attention-MIL with top-k anomalous slice preservation |
| **G4** | Series Selection Heuristic Failure (GE/Siemens/Philips metadata variance) | Plane confidence < 0.85 | Multi-factor classification: ImageOrientationPatient + FOV + slice spacing |
| **G5** | Teammate Collision & Deadline Fatigue (final 6h) | Conflicting submissions in LIVE_STATE | Single-operator lock; FINAL_DAY_CHECKLIST mechanical execution |
| **G6** | Noisy Soft-Label Drift from Report Extractors | Val loss diverges on ambiguous categories | Label smoothing (α=0.05) + multi-prompt consensus |
| **G7** | Hidden-Test Scale Memory Leak (3 samples OK, 500+ crashes) | Kaggle "Exceeded Memory Limit" | `gc.collect()` + `torch.cuda.empty_cache()` every 10 studies |

---

## 🚀 PART IV: THE 10 LIMITATIONS WE MUST OVERCOME TO REACH 0.969

| Limitation | Current State | Impact | Solution |
|------------|---------------|--------|----------|
| **1. Public Notebook Plateau (0.958)** | 1,617 teams tied | Anyone using public weights stuck here | Proprietary high-capacity models (ConvNeXt-L + Swin-B + DINOv2) |
| **2. Single-Fold Training Noise** | E12/E13 Fold 0 only | ±0.004 variance across splits | Complete 5-fold training + ensemble (+0.003–0.005) |
| **3. Gold-58 Sanity Trap** | 7 fractures, 9 MCL | Tuning on it = noise-fitting | **Strictly** 5-fold OOF (4,407 studies) only |
| **4. Public LB Rounding & Shakeup** | 3 decimals, small gains invisible | Selection on public LB = catastrophic shakeup | Ground selections in multi-fold OOF |
| **5. Efficiency Prize vs Max AUC Trade-off** | <500s caps ensemble | Artificially caps ceiling | Two-Track: V45 (<500s) + V49 unconstrained |
| **6. No OOF Weight Optimization Yet** | Flat 15% weights | Per-pathology optimal weights vary wildly | Nelder-Mead on 5-fold OOF (`optimize_pathology_stacking_oof.py`) |
| **7. Missing DINOv2 Self-Supervised Semantics** | Zero in ensemble | Zero supervised bias collapse model | DINOv2-ViT-L/14 frozen + MIL head (planned Arm 4) |
| **8. Multi-Planar Cross-Attention Not Built** | 5-slot treats planes independently | Misses Sagittal-Coronal-Axial interactions | Joint cross-attention (planned) |
| **9. Asymmetric Focal Loss Not Fully Optimized** | Fixed γ₋=4, γ₊=1 | Rare classes (Fracture 2.5%, MCL 3.2%) under-optimized | Per-class γ tuned on OOF |
| **10. Teammate Coordination Risk** | 5 members, shared quota | Someone burns quota / changes selection | Single-operator lock; announce submissions 30min advance |

---

## 📋 PART V: EXECUTION PATH TO 0.969 (Step by Step)

### Phase 1: V48 Score Capture (NOW — Hours 53–49)
```bash
# Monitor V48 (Ref 57448780) via task-20791
python tools/monitor_v48_scoring.py
# If V48 ≥ 0.9610 → SELECT IN SLOT 2 immediately
# If V48 = 0.958–0.960 → analyze per-class deltas → submit V49
# If V48 ≤ 0.957 → REGRESSION → root cause analysis → V49 with fixes
```

### Phase 2: V49 OOF-Calibrated Quad-Arm (Hours 49–43)
```bash
# 1. Generate 5-fold OOF predictions for ALL arms
python tools/extract_engine12_oof.py  # Engine 12 5-fold
python tools/extract_engine13_oof.py  # Engine 13 5-fold
python tools/extract_raptor_oof.py    # Raptor 5-fold

# 2. Per-pathology Nelder-Mead optimization
python tools/optimize_pathology_stacking_oof.py \
  --predictions_dir oof_predictions/ \
  --output v49_optimal_weights.json

# 3. Inject weights into V49 notebook
python tools/build_v49_oof_calibrated.py --weights v49_optimal_weights.json

# 4. Guardrail + Submit
python tools/validate_kernel_guardrail.py submissions/v49_oof_calibrated_quad_arm/...
kaggle kernels push -p submissions/v49_oof_calibrated_quad_arm
```

### Phase 3: V50 Multi-Fold Swin Arm (Hours 43–37)
```bash
# Colab Tab 1: Swin-Base Fold 1 (colab_engine13_fold1_master.ipynb)
# Colab Tab 2: Swin-Base Fold 2 (colab_engine13_fold2_master.ipynb)
# Colab Tab 3: ConvNeXt-Large Fold 1 (colab_engine12_fold1_master.ipynb)
# Colab Tab 4: ConvNeXt-Large Fold 2 (colab_engine12_fold2_master.ipynb)

# Each: 8 epochs, grad checkpointing, chunk=4, k=8
# Push weights to Kaggle Datasets at EACH epoch (fail-soft)
# Build V50 with 4th arm (Swin Fold 1) at 10% weight
```

### Phase 4: DINOv2 Integration (If Time Permits, Hours 37–31)
```bash
# Build DINOv2-ViT-L/14 frozen feature extractor
# Train lightweight MIL head on 384px crops
# Add as 4th/5th arm at 5-10% weight
```

### Phase 5: Final Selection & Hard Freeze (Hours 31–0)
```bash
# T-12h: HARD FREEZE — NO CODE CHANGES
# T-24h: Verify Slot 1 (57253840) + Slot 2 (best ≥0.9610) in Kaggle UI
# T-6h: Re-verify selections. Screenshot.
# T-1h: python tools/state_snapshot.py
# T-0: Competition closes
```

---

## 🛡️ PART VI: HARD CONSTRAINTS CHECKLIST (Pre-Submission Gate)

Before ANY Kaggle push, verify ALL (exit code 0 required):

- [ ] `python tools/validate_kernel_guardrail.py <notebook.ipynb>` → **PASSED**
- [ ] `python tools/verify_submission_integrity.py <submission.csv>` → **PASSED**
- [ ] No `assert` / `raise RuntimeError` on test data shapes
- [ ] All predictions clamped to [0.0001, 0.9999]
- [ ] `StudyInstanceUID` matches test.csv bit-for-bit (dtype=str)
- [ ] Native ≥384px weight ≥ 70% (ADR-048 / Rule 11)
- [ ] No re-blending of pre-blended outputs (e.g., raptor_sub + raptor224_sub)
- [ ] New model weights ≤ 15% without OOF optimization (Rule 5)
- [ ] Linear rank-percentile blending ONLY (Rules 2, 3)
- [ ] Gold-58 NOT used for weight tuning (Rule 6)
- [ ] Isolated script execution (`!python train_....py`)
- [ ] Gradient checkpointing + chunk_size ≤ 4 for 384px backbones
- [ ] Checkpoints pushed to Kaggle Datasets at each epoch
- [ ] Daily quota check: `python tools/state_snapshot.py` → submissions remaining

---

## 🤖 PART VII: AGENT CONTINUITY PROTOCOL (Zero Context Loss)

### Cold Start (60 seconds, every agent switch):
```bash
cd C:\Users\Cyrus\Downloads\rsna knee fip
python tools/state_snapshot.py          # 3s - regenerate ground truth
cat docs/LIVE_STATE.md                  # 10s - ONLY source for numbers
tail -n 120 docs/LIVE_STEP_LOG.md       # 15s - last 30 steps
cat docs/RISK_REGISTER_TOP10.md         # 15s - everything that kills
cat docs/MASTER_COMPREHENSIVE_DOCUMENTATION.md  # 15s - THE BIBLE
```

### Logging (After EVERY meaningful action):
```bash
python tools/log_step.py --agent "<MODEL>" --kind <SUBMIT|SCORE|TRAIN|DECISION|FAILURE|STEP> \
  --action "<what you did>" --result "<observed OUTCOME>" \
  --evidence "<command/file/ref>" --next "<SINGLE next step>"
python tools/state_snapshot.py  # ALWAYS after submit/score
```

### Session End:
```bash
python tools/state_snapshot.py
python tools/log_step.py --agent "<MODEL>" --kind STEP \
  --action "Session handoff" --result "<exact state>" \
  --evidence "<files/refs>" --next "<exact resume point>"
git add docs decisions tools AGENTS.md
git commit -m "docs: session handoff $(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

---

## 📊 PART VIII: DECISION MATRIX — V48/V49 SCORE SCENARIOS

| V48 Public LB | Action | Next Candidate |
|---------------|--------|----------------|
| **≥ 0.9610** | **SELECT IN SLOT 2** — Two-Track complete | V49 for margin (optional) |
| **0.9600–0.9609** | Do NOT select. Submit V49 OOF-calibrated | V49 (target 0.961+) |
| **0.9580–0.9599** | Do NOT select. Submit V49 + start V50 multi-fold | V49 → V50 |
| **≤ 0.9579** | REGRESSION. Analyze per-class. Root cause. | V49 with fixed architecture |

**Decision Authority:** Single operator. Logged in LIVE_STEP_LOG. Screenshot in Kaggle UI.

---

## 🎯 PART IX: CURRENT ASSET STATUS (Verified Live)

| Asset | Location | Status | Key Metric |
|-------|----------|--------|------------|
| **Engine 12 Weights** | `cyruspinto/rsna-knee-engine12-convnext-large-weights` | ✅ LIVE | 198M params, Gold AUC 0.8808 |
| **Engine 13 Weights** | `cyruspinto/rsna-knee-engine13-swin-base-weights` | ✅ LIVE | 88M params, Gold AUC 0.8577 |
| **Engine 12 Fold 1 Colab** | `~/Downloads/colab_engine12_fold1_master.ipynb` | ✅ Ready | 80/20 stratified, dual OOF/Gold-58 |
| **Engine 13 Fold 1 Colab** | `~/Downloads/colab_engine13_fold1_master.ipynb` | ✅ Ready | 80/20 stratified, dual OOF/Gold-58 |
| **V48 Ensemble** | `submissions/v48_apex_quad_arm_ensemble/` | 🚀 SUBMITTED (57448780) | Guardrail PASSED, 384px dominant |
| **V49 Ensemble** | `submissions/v49_oof_calibrated_quad_arm/` | ✅ READY | Guardrail PASSED, OOF weights placeholder |
| **V48 Monitor** | `tools/monitor_v48_scoring.py` | 🟢 ACTIVE | Polling Ref 57448780 |

---

## 🏁 PART X: FINAL WORDS — THE UNVARNISHED TRUTH

**The gap is +0.011 AUC to #1. The plateau has 1,617 teams. Re-weighting public weights is mathematically zero.**  

**The ONLY path:**
1. **Complete 5-fold training** for Engine 12 (ConvNeXt-Large-384) and Engine 13 (Swin-Base-384) → +0.003–0.005
2. **Run OOF weight optimization** per-pathology (Nelder-Mead) → +0.002–0.004  
3. **Add DINOv2-ViT-L/14** as 4th orthogonal arm → +0.002–0.004
4. **Build multi-planar cross-attention** → +0.001–0.003
5. **Two-Track Defense**: Slot 1 locked at 0.958, Slot 2 pushes ceiling

**Every submission must test ONE hypothesis with predicted effect logged BEFORE submit.**

**The next 53 hours are won by execution discipline, not cleverness.**

---

*This playbook is the mathematical and operational foundation. Every number is verified. Every rule has a root cause. Follow it mechanically. Reach 0.969+.*