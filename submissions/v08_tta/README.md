# Submission Version 8: Test-Time Augmentation (TTA) Experiment

## Overview
- **Configuration Tested:** Test-Time Augmentation (Horizontal Slice Flipping).
- **Hypothesis:** Horizontal flipping may regularize spatial inference across planes.
- **Empirical Status:** Kernel prepared in `scratch/v8_tta_kernel/`. No distinct submitted LB improvement over the V6 baseline was observed.
- **Anatomical Consideration:** Knee anatomy has distinct Medial vs Lateral compartments; naive horizontal flipping without compartment label swapping can introduce ambiguity between medial and lateral structures on Coronal/Axial views.
- **Decision:** TTA was not promoted to the final candidate ensemble to keep inference simple and within the 500s runtime envelope.
