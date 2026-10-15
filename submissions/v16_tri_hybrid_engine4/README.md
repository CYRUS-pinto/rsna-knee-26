# Submission Candidate V16: Tri-Hybrid Master Blend (Engine 1 + Engine 2 + Engine 4)

## 1. Specification & Rationale
- **Target Public LB:** **0.963 - 0.965+** (Breaking through the 0.956 tie-breaker plateau)
- **Target Runtime:** **~420s** (Fully compliant with the $18,000 Efficiency Prize Pool ceiling of 500s)
- **Architectures:**
  1. **Engine 1 (Raptor):** CoAtNet-384 SWA trained on consensus soft labels (Weight: 0.35 on focal tears, 0.30 on systemic).
  2. **Engine 2 (ConvNeXt R7):** 4-fold ConvNeXt-S6 352px with OA extent heads (Weight: 0.30 on focal tears, 0.45 on osteoarthritis).
  3. **Engine 4 (Proprietary):** 3-Plane 2.5D ConvNeXt-Small Attention MIL with 12 Disentangled Heads (Weight: 0.35 on focal tears, 0.25 on systemic).

---

## 2. Ensembling Mathematics: Specialist Column Rank Routing (ADR-013 & ADR-014)

$$\tilde{r}_k = \text{Rank}(S_k) \in [0, 1]$$

For Focal Targets (`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`):
$$R_{\text{focal}} = 0.35 \cdot \tilde{r}_1 + 0.30 \cdot \tilde{r}_2 + 0.35 \cdot \tilde{r}_4$$

For Systemic & Osteoarthritis Targets (`Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`):
$$R_{\text{systemic}} = 0.30 \cdot \tilde{r}_1 + 0.45 \cdot \tilde{r}_2 + 0.25 \cdot \tilde{r}_4$$

---

## 3. Submission Timing Constraint
- User strict instruction: **Exactly 1 submission permitted at 06:00 AM local time.**
- Slot 1 is permanently locked to V6 (0.956).
- Slot 2 will receive this single, verified V16 candidate.
