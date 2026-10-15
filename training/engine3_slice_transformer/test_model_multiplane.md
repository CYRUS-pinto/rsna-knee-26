# MultiPlane Architecture Verification Suite (`test_model_multiplane.py`)

## 1. Overview
`test_model_multiplane.py` is an automated unit test suite that validates the tensor shapes, forward computation, attention pooling, and backward backpropagation gradients of `MultiPlaneSliceTransformerHead` and `MultiPlaneEngine3` before dispatching compute-heavy training jobs.

---

## 2. Test Specifications

### Test 1: Standard Forward Pass with K=44 Slices
- **Input Shape**: Features `(B=2, K=44, D=1024)`.
- **Plane IDs**: LongTensor of shape `(2, 44)` containing values $\in \{0, 1, 2\}$.
- **Expected Output**: Logits `(2, 12)` with values $\in \mathbb{R}$.
- **Verification**: Output shape matches `(2, 12)`, no NaN or Inf values.

### Test 2: Variable Sequence Length (K=96 Slices)
- Tests the ability of the dynamic positional embedding interpolation and multi-head attention to handle larger slice sequences (e.g. concatenated multi-plane studies).
- **Input Shape**: Features `(B=2, K=96, D=1024)`.
- **Expected Output**: Logits `(2, 12)`.

### Test 3: Backpropagation & Gradient Flow
- Verifies that gradients flow cleanly through:
  - Output linear classifier (`clsW`, `clsb`)
  - Gated attention query network
  - Transformer encoder self-attention layers
  - Positional and discrete plane embeddings
- **Verification**: `loss.backward()` succeeds and all parameters have non-zero `param.grad`.

---

## 3. Running the Test Suite
```bash
python training/engine3_slice_transformer/test_model_multiplane.py
```
Expected output:
```text
=== MultiPlane Architecture Unit Test ===
[PASS] Standard input (K=44): Output shape = torch.Size([2, 12])
[PASS] Long sequence (K=96): Output shape = torch.Size([2, 12])
[PASS] Multi-plane routing: Forward computation valid
[PASS] Gradient backprop: Clean gradient flow through all heads
[PASS] All 4 architecture checks passed 100%!
```
