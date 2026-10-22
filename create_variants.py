import json
import shutil
import os

src = r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb'

# Variant B: High anchor (0.60)
dst_b = r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_b_high_anchor\rsna-knee-v50-variant-b.ipynb'
os.makedirs(os.path.dirname(dst_b), exist_ok=True)
shutil.copy2(src, dst_b)

with open(dst_b, 'r') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = ['# RSNA Knee V50 Variant B \u2014 High Anchor (60% Raptor, 40% MIL)\n', '## anchor_share=0.60 | More Raptor weight']

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src_text = ''.join(cell['source'])
        if 'fuse_v50_predictions' in src_text and 'anchor_share' in src_text:
            new_src = src_text.replace('anchor_share: float = 0.50', 'anchor_share: float = 0.60')
            new_src = new_src.replace('final[:, c_idx] = 0.50 * rank_anchor[:, c_idx] + 0.50 * mil_col', 'final[:, c_idx] = 0.60 * rank_anchor[:, c_idx] + 0.40 * mil_col')
            cell['source'] = [new_src]
            print('Modified Variant B')
            break

with open(dst_b, 'w') as f:
    json.dump(nb, f, indent=1)
print('Variant B done')

# Variant C: No Raptor (pure MIL)
dst_c = r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_c_no_raptor\rsna-knee-v50-variant-c.ipynb'
os.makedirs(os.path.dirname(dst_c), exist_ok=True)
shutil.copy2(src, dst_c)

with open(dst_c, 'r') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = ['# RSNA Knee V50 Variant C \u2014 No Raptor (Pure MIL Ensemble)\n', '## anchor_share=0.0 | 100% MIL, 0% Raptor']

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src_text = ''.join(cell['source'])
        if 'RAPTOR ANCHOR INFERENCE' in src_text:
            cell['source'] = ['# RAPTOR DISABLED FOR VARIANT C\n', 'raptor_blend = None']
            print('Disabled Raptor cell in Variant C')
        elif 'fuse_v50_predictions' in src_text and 'anchor_share' in src_text:
            new_src = '''# FUSE WITHOUT RAPTOR (Pure MIL)
final = np.zeros_like(rank_e12, dtype=np.float32)

for c_idx, c_name in enumerate(TARGETS):
    w12, w13, w10 = GOLD58_OPTIMIZED_3ENGINE_WEIGHTS[c_name]
    
    mil_parts, mil_weights = [], []
    if rank_e12 is not None and w12 > 0:
        mil_parts.append(rank_e12[:, c_idx]); mil_weights.append(w12)
    if rank_e13 is not None and w13 > 0:
        mil_parts.append(rank_e13[:, c_idx]); mil_weights.append(w13)
    if rank_e10 is not None and w10 > 0:
        mil_parts.append(rank_e10[:, c_idx]); mil_weights.append(w10)

    if not mil_parts:
        final[:, c_idx] = 0.5
    else:
        w_sum = sum(mil_weights)
        mil_col = sum((mw / sum(mil_weights)) * mp for mw, mp in zip(mil_weights, mil_parts))
        final[:, c_idx] = mil_col  # 100% MIL, no anchor

# Final percentile rank + clamp
final_ranked = pd.DataFrame(final).rank(pct=True).values
final_clamped = np.clip(final_ranked, 0.0001, 0.9999)

for c_idx, c_name in enumerate(TARGETS):
    bad = np.isnan(final_clamped[:, c_idx]) | np.isinf(final_clamped[:, c_idx])
    if bad.any():
        final_clamped[bad, c_idx] = FALLBACK_MEDIANS[c_name]

print(f"Final shape: {final_clamped.shape}")
print(f"Range: [{final_clamped.min():.6f}, {final_clamped.max():.6f}]")'''
            cell['source'] = [new_src]
            print('Modified fusion cell in Variant C')
            break

with open(dst_c, 'w') as f:
    json.dump(nb, f, indent=1)
print('Variant C done')

print('All variants created!')