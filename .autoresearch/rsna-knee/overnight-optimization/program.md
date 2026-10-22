# Overnight Optimization Program: RSNA Knee 0.958 → 0.969+

## Objective
Break the 0.958 plateau and reach 0.969+ (Top 1) through systematic ensemble optimization, architectural improvements, and label quality enhancements.

## Current State (2026-10-20 19:50 UTC)
- **V48**: Submitted (Ref 57448780), PENDING scoring
- **Slot 1**: Locked at 0.958 (Ref 57253840) - NEVER TOUCH
- **Daily Quota**: 0/5 used - resets at 00:00 UTC
- **V49**: Built with placeholder OOF weights, guardrail PASSED
- **Engine 12/13 Weights**: LIVE on Kaggle (0.8808 / 0.8577 Gold AUC)

## Constraints (HARD - NEVER VIOLATE)
1. **Two-Track Defense**: Slot 1 (0.958) LOCKED forever unless candidate ≥ 0.9610
2. **Rank-Percentile Blending ONLY**: No raw probability averaging, no power sharpening (p>1)
3. **Gold-58 = Sanity Only**: 7 fractures, 9 MCL tears - NEVER optimize on it
4. **New Model Weight ≤ 15%** until OOF-optimized weights exist
4. **Zero Fatal Assertions**: All notebooks must pass `validate_kernel_guardrail.py`
5. **Resolution Dominance**: 384px weight ≥ 70%, no re-blending pre-blended outputs
6. **Training = Isolated Script**: `!python train_x.py`, grad checkpointing, chunk_size ≤ 4
7. **No Credentials in Files**: Use env vars / Colab Secrets

## Strategy Escalation (by run count)

### Runs 1-3: OOF Weight Optimization (HIGHEST LEVERAGE)
- Extract Engine 12/13 Fold 0 OOF predictions
- Run Nelder-Mead per-pathology optimization
- Inject real weights into V49
- Submit V49 at 00:00 UTC quota reset

### Runs 4-8: Multi-Fold Training + DINOv2
- Launch 5-fold training for Engine 12 (ConvNeXt-Large) + Engine 13 (Swin-Base)
- Build DINOv2-ViT-L/14 frozen + MIL head as 4th arm
- Push checkpoints every epoch to Kaggle datasets

### Runs 9-12: Architectural Enhancements
- Cross-planar attention (Sagittal↔Coronal↔Axial)
- Per-class asymmetric focal γ optimization
- Multi-scale pyramid (224→384→512)

### Runs 13-15: TTA + Label Quality
- Flip+Scale TTA at inference
- Active learning on Gold-58 uncertain samples
- Semi-supervised pseudo-labeling

### Runs 16+: Final Ensemble Assembly
- Assemble V50/V51 with all gains
- Target: 0.969+ public LB

## Failure Handling
- If OOF extraction fails → Use Fold 0 weights with flat 15% weights (V48 architecture)
- If Colab disconnects → Checkpoints pushed to Kaggle every epoch (fail-soft)
- If V48 scores < 0.958 → Root cause analysis, fix → V49 with fixes
- If quota exhausted → Wait for reset, never exceed 5/day
- If 5 consecutive crashes → STOP, alert user

## Target Metrics
- **Primary**: Public LB AUC (maximize)
- **Secondary**: Gold-58 Macro AUC ≥ 0.85 (sanity)
- **Tertiary**: Inference time < 500s (efficiency prize)

## Stop Criteria
- Public LB ≥ 0.9610 → Select in Slot 2, continue for margin
- Public LB ≥ 0.967 → Top 10 achieved
- Public LB ≥ 0.969 → #1 achieved
- Deadline: 2026-10-22 23:59 UTC

## Documentation Requirements
- Every step logged to `docs/LIVE_STEP_LOG.md`
- State snapshot after each submission: `python tools/state_snapshot.py`
- Ceiling verification before each submit: `python tools/ceiling_verification_suite.py --quick`