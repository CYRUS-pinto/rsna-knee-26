# ADR-048: Spatial Resolution Dilution Post-Mortem (Candidate V47 -> 0.953)

## Status
Accepted / Completed

## Context
On October 20, 2026 at 13:33 UTC, Candidate V47 (`57429824`) scored **0.953** on the Kaggle Public Leaderboard.
Per Rule 1 of `AGENTS.md`, our **0.958 safety baseline in Slot 1 (`57253840`) remained permanently locked and protected**.

## Root-Cause Analysis
Direct source code inspection of Candidate V47 revealed the exact mechanism of the 0.005 AUC regression:
1. **Double-Blending of 224-pixel Raptor Stage:**
   - In `raptor_run.py` (Cell 2/3), the script already blends the 384-pixel Raptor model and the 224-pixel Raptor model using internal weights (65% 384px, 35% 224px).
   - In Cell 8 of V47, Stage 3 applied:
     `r_val = (1.0 - A224) * rsub[lbl] + A224 * rsub224[lbl]` (where `A224 = 0.35`).
   - This mathematically compounded the 224-pixel model weight to:
     $$W_{224} = 0.35 + (1 - 0.35) \times 0.35 = 0.5775 \quad (57.8\%)$$
2. **The Spatial Nyquist Penalty:**
   - As proven in `docs/THE_ABSOLUTE_CEILING_AND_LIMITATIONS_PLAYBOOK.md`, the 224-pixel resolution has a pixel pitch of $0.625\text{ mm}$, where meniscal tears ($0.8\text{ mm}$) and subtle fractures span only $\sim 1.3\text{ pixels}$.
   - Increasing the 224-pixel share from $35\%$ to $58\%$ actively blurred fine spatial boundaries, directly causing the drop to 0.953.

## Decision & Rule
1. **Never Re-Blend Pre-Blended Outputs:**
   `raptor_sub.csv` is already the unified multi-scale Raptor output. Do not apply a second linear combination with `raptor224_sub.csv`.
2. **Enforce Resolution Dominance:**
   The native 384-pixel representations (ConvNeXt-Large-384, Swin-Base-384, Raptor-384) must maintain $\ge 70\%$ of the total ensemble weight.
3. **Candidate V48 Fix:**
   Candidate V48 (`submissions/v48_apex_quad_arm_ensemble`) takes `raptor_sub.csv` directly without double-weighting 224, restoring full 384-pixel fidelity while incorporating Engine 12 and Engine 13.
