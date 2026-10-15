# Submission Version 8: Test-Time Augmentation (TTA) (Public LB: 0.955)

## Overview
- **Public LB Score:** **0.955**
- **Hypothesis:** Horizontal flip test-time augmentation (TTA) improves spatial generalization.
- **Empirical Result:** Knee anatomy has distinct lateral and medial compartments; horizontal flipping mirrors medial and lateral structures (e.g. Medial Meniscus vs Lateral Meniscus), diluting compartmental precision unless bilateral labels are explicitly swapped.
- **Takeaway:** TTA must be strictly plane-aware and never horizontally flip coronal/axial slices where medial-lateral orientation is anatomically fixed.
