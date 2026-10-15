# Submission Candidate V16: Tri-Hybrid Master Blend (Engine 1 + Engine 2 + Engine 4)

## 1. Specification & Rationale
- **Target Public LB:** **0.963 - 0.965+** (Breaking through the 0.956 tie-breaker plateau)
- **Target Runtime:** **~420s** (Fully compliant with the $18,000 Efficiency Prize Pool ceiling of 500s)
- **Architectures:**
  1. **Engine 1 (Raptor):** CoAtNet-384 SWA trained on consensus soft labels (Weight: 0.35 on focal tears, 0.30 on systemic).
  2. **Engine 2 (ConvNeXt R7):** 4-fold ConvNeXt-S6 352px with OA extent heads (Weight: 0.30 on focal tears, 0.45 on osteoarthritis).
  3. **Engine 4 (Proprietary):** 3-Plane 2.5D ConvNeXt-Small Attention MIL with 12 Disentangled Heads (Weight: 0.35 on focal tears, 0.25 on systemic).

---

## 2. Ensembling Mathematics: Empirical Specialist Column Rank Routing (ADR-013 & ADR-014)

$$\tilde{r}_k = \text{Rank}(S_k) \in [0, 1]$$

Based on empirical validation across 881 OOF studies from the freshly trained Engine 4:
- **Synovitis Specialist ($0.8225$ AUC):** $0.40 \cdot \tilde{r}_1 + 0.40 \cdot \tilde{r}_2 + \mathbf{0.20 \cdot \tilde{r}_4}$
- **Effusion Specialist ($0.7564$ AUC):** $0.425 \cdot \tilde{r}_1 + 0.425 \cdot \tilde{r}_2 + \mathbf{0.15 \cdot \tilde{r}_4}$
- **Fracture Specialist ($0.7161$ AUC):** $0.425 \cdot \tilde{r}_1 + 0.425 \cdot \tilde{r}_2 + \mathbf{0.15 \cdot \tilde{r}_4}$
- **MCL & Medial OA Specialists ($0.7074\text{--}0.7141$ AUC):** $0.45 \cdot \tilde{r}_1 + 0.45 \cdot \tilde{r}_2 + \mathbf{0.10 \cdot \tilde{r}_4}$
- **Acute Ligament Tears & Others (`ACL`, `Meniscus`, `Baker's`, `Contusion`, `PF OA`):** $\mathbf{0.50 \cdot \tilde{r}_1 + 0.50 \cdot \tilde{r}_2 + 0.00 \cdot \tilde{r}_4}$ (100% baseline protection preserving the 0.956 PB).

---

## 3. Submission Timing Constraint
- User strict instruction: **Exactly 1 submission permitted at 06:00 AM local time.**
- Slot 1 is permanently locked to V6 (0.956).
- Slot 2 will receive this single, verified V16 candidate.
