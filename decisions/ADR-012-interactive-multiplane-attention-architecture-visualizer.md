# ADR-012: Interactive Visual Architecture Representation for Multi-Plane Cross-Attention

## Context & Problem Statement
Understanding how a multi-plane sequence-aware Transformer head redistributes attention weights across heterogeneous MRI views (Sagittal, Coronal, Axial) is critical for:
1. Validating that clinical anatomical priors match attention distributions (e.g. verifying that ACL queries attend to mid-sagittal slices 10-14, while Baker's cyst queries attend to posterior axial slices).
2. Facilitating team-wide collaboration and code review without requiring teammates to decipher raw tensor Einstein summations (`torch.einsum('bkn,bkf->bnf', a, h)`).

## Decision
Create an interactive, zero-dependency browser-executable HTML visualizer at [`docs/multiplane_transformer_visualizer.html`](../docs/multiplane_transformer_visualizer.html).
Features include:
1. Dynamic 12-class pathology selection query routing.
2. Real-time slice attention distribution ($a_{k, n}$) bar charts with peak anatomical callouts.
3. Interactive $24 \times 24$ self-attention cross-slice matrix heatmap.
4. Live V13 Tri-Hybrid uniform percentile rank averaging simulator.

## Status
**Accepted & Committed** (Accessible via `docs/multiplane_transformer_visualizer.html`).

## Consequences
- Provides instant visual verification of model behavior for all 12 pathologies.
- Serves as an interactive artifact for presentations, technical reviews, and team synchronization.
