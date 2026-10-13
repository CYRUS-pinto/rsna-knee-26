import pandas as pd
import numpy as np

FINDINGS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# Weights optimized for per-finding agreement with adjudicated gold standard
WEIGHTS = {
    'ACL': 0.50,
    'MCL': 0.10,
    'Medial Meniscus': 0.30,
    'Lateral Meniscus': 0.20,
    'Medial OA': 0.90,
    'Lateral OA': 0.70,
    'PF OA': 0.00,
    'Effusion': 1.00,
    'Synovitis': 0.30,
    "Baker's": 1.00,
    'Contusion': 0.80,
    'Fracture': 0.90
}

df_hybrid = pd.read_csv('HYBRID_labels_scores.csv')
df_opus = pd.read_csv('scratch/ntejas_repo/labels_ds/labels_v2_repaired.csv')
df_opus['StudyInstanceUID'] = df_opus['STUDY_UID']

m = pd.merge(df_hybrid[['StudyInstanceUID'] + FINDINGS], df_opus[['StudyInstanceUID'] + FINDINGS], on='StudyInstanceUID', suffixes=('_hyb', '_opus'))

blended_df = pd.DataFrame({'StudyInstanceUID': m['StudyInstanceUID']})
for f in FINDINGS:
    w = WEIGHTS[f]
    blended_df[f] = w * m[f'{f}_hyb'] + (1.0 - w) * m[f'{f}_opus']

out_path = 'SUPER_CONSENSUS_labels.csv'
blended_df.to_csv(out_path, index=False)
print(f"Generated {out_path} with {len(blended_df)} studies across all 12 findings.")
