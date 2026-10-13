import json, os

src_nb_path = 'scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb'
dst_nb_path = 'scratch/kernels/r7_quint_ensemble/rsna-knee-quint-ensemble-r7-colab.ipynb'

with open(src_nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 3 CANDIDATE_ARMS to include 224px multiscale model
c3 = ''.join(nb['cells'][3]['source'])
old_arms_block = '''CANDIDATE_ARMS = [
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.40},
    {"file": "colab_super_coatnet_0917_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.35},
    {"file": "colab_super_convnext_0917_swa.pt", "arch": "convnext_small.fb_in22k_ft_in1k_384", "res": 384, "w": 0.25},
]'''

new_arms_block = '''CANDIDATE_ARMS = [
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.30},
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.15},
    {"file": "colab_super_coatnet_0917_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.30},
    {"file": "colab_super_convnext_0917_swa.pt", "arch": "convnext_small.fb_in22k_ft_in1k_384", "res": 384, "w": 0.25},
]'''

assert old_arms_block in c3, "Could not find old_arms_block in Cell 3"
c3 = c3.replace(old_arms_block, new_arms_block)

# Ensure 224px hash check doesn't block loading
old_hash_check = 'if "raptor_ft_alldata" in pt_path:'
new_hash_check = 'if "raptor_ft_alldata_t16_blendjev_oai_d96_r384" in pt_path:'
if old_hash_check in c3:
    c3 = c3.replace(old_hash_check, new_hash_check)

nb['cells'][3]['source'] = [c3]

# 2. Update Cell 6 to prioritize Heliosli R7 (rsna-knee-compact-r7cnxfsss)
c6 = ''.join(nb['cells'][6]['source'])
old_c_slug = "C_SLUG = 'rsna-knee-compact-v1'"
new_c_slug = "C_SLUG = 'rsna-knee-compact-r7cnxfsss'"
if old_c_slug in c6:
    c6 = c6.replace(old_c_slug, new_c_slug)

old_cand_list = "for cand_c in ['rsna-knee-compact-v1', 'rsna-knee-compact-r6cnxfsss', 'compact_models']:"
new_cand_list = "for cand_c in ['rsna-knee-compact-r7cnxfsss', 'rsna-knee-compact-v1', 'rsna-knee-compact-r6cnxfsss', 'compact_models']:"
if old_cand_list in c6:
    c6 = c6.replace(old_cand_list, new_cand_list)

nb['cells'][6]['source'] = [c6]

with open(dst_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated Quint Ensemble notebook at {dst_nb_path}!")
