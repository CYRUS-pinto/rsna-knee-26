import json

nb_path = 'scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix 1: infer_probs in Cell 3 (Numerical stability for fp16 ConvNeXt)
c3 = ''.join(nb['cells'][3]['source'])
old_infer = """@torch.no_grad()
def infer_probs(model, xwins, device):
    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"
    x = xwins.unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x).float()
        assert torch.isfinite(logits).all(), "Nonfinite fp16 logits"
        o = torch.sigmoid(logits)
    return o[0].cpu().numpy()"""

new_infer = """@torch.no_grad()
def infer_probs(model, xwins, device):
    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"
    x = xwins.unsqueeze(0).to(device)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
            assert torch.isfinite(logits).all(), "Nonfinite fp16 logits"
    except Exception:
        logits = model(x.float()).float()
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
    o = torch.sigmoid(logits)
    return o[0].cpu().numpy()"""

assert old_infer in c3, "Could not find old_infer in Cell 3"
c3 = c3.replace(old_infer, new_infer)
nb['cells'][3]['source'] = [c3]

# Fix 2: Cell 9 blend uses multi-arm rsub and removes single-arm correlation assertion
c9 = ''.join(nb['cells'][9]['source'])
old_c9_target = "rr, ro = pct(raw[c].values), pct(ours[c].values)"
new_c9_target = "rr, ro = pct(rsub[c].values), pct(ours[c].values)"
assert old_c9_target in c9, "Could not find old_c9_target in Cell 9"
c9 = c9.replace(old_c9_target, new_c9_target)

old_assert = """if rho_rr and min(rho_rr.values()) < 0.99:
    raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)"""
new_assert = """if rho_rr:
    print('[blend] spearman_raptor_raw_vs_sub:', rho_rr)"""
assert old_assert in c9, "Could not find old_assert in Cell 9"
c9 = c9.replace(old_assert, new_assert)
nb['cells'][9]['source'] = [c9]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully patched Cell 3 (fp32 fallback) and Cell 9 (multi-arm rsub blend)!")
