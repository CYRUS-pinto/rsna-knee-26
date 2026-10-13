import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('scratch/charts', exist_ok=True)

# Set high-quality aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# CHART 1: Bottleneck Analysis across all 12 findings
# ==============================================================================
findings = [
    'ACL', 'MCL', 'Medial Meniscus', 'PF OA', 'Medial OA', "Baker's",
    'Fracture', 'Lateral Meniscus', 'Effusion', 'Contusion', 'Lateral OA', 'Synovitis'
]
aucs = [1.0000, 0.9955, 0.9615, 0.9447, 0.9302, 0.9203, 0.9097, 0.8783, 0.8770, 0.8758, 0.8704, 0.8435]
prevalence = [41.4, 15.5, 44.8, 36.2, 25.9, 20.7, 31.0, 39.7, 60.3, 32.8, 19.0, 46.6]

fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)

y_pos = np.arange(len(findings))[::-1]
colors = ['#2ca02c' if a >= 0.95 else '#1f77b4' if a >= 0.90 else '#d62728' for a in aucs]

bars = ax1.barh(y_pos, aucs, color=colors, height=0.65, alpha=0.88, edgecolor='black', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(findings, fontsize=11, fontweight='bold')
ax1.set_xlim(0.70, 1.02)
ax1.set_xlabel('Super-Consensus Gold ROC-AUC Score', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('Finding-by-Finding Performance & Clinical Bottleneck Analysis\n(Green: Solved Focal Tears | Blue: Strong | Red: Critical SOTA Bottlenecks)', fontsize=13, fontweight='bold', pad=15)

for bar, auc, prev in zip(bars, aucs, prevalence):
    width = bar.get_width()
    ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{auc:.4f}  (Prev: {prev:.1f}%)',
             va='center', ha='left', fontsize=9.5, fontweight='bold', color='#222222')

ax1.axvline(0.90, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7, label='0.90 Macro Baseline')
ax1.axvline(0.95, color='#2ca02c', linestyle=':', linewidth=1.2, alpha=0.8, label='0.95 Grandmaster Tier')
ax1.legend(loc='lower left', frameon=True, fontsize=10)

plt.tight_layout()
fig1_path = 'scratch/charts/bottleneck_analysis.png'
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f'Saved Chart 1: {fig1_path}')


# ==============================================================================
# CHART 2: SOTA Progression & Leaderboard Ceiling Ladder
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

stages = [
    'Public Single CoAtNet\n(Nartaa 0.949)',
    'Public S6 Blend\n(Heliosli 70/30 Baseline)',
    'Public Plateau Ceiling\n(200+ Teams Forking)',
    'Heliosli R7 Baseline\n(Oct 13 Release)',
    'Our Version 6\n(3-Arm SWA Quad-Ensemble)',
    'Heliosli R7 + Colab SWA\n(Target Next Step)',
    'Global Rank #1\n(CentralyDS / Sergey)'
]
scores = [0.949, 0.954, 0.954, 0.956, 0.958, 0.963, 0.966]
bar_colors = ['#aec7e8', '#c7c7c7', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#d62728']

bars = ax.bar(stages, scores, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.6, alpha=0.9)
ax.set_ylim(0.940, 0.970)
ax.set_ylabel('Macro-Averaged ROC-AUC', fontsize=12, fontweight='bold')
ax.set_title('Competition Progression: From the 0.954 Trap to the 0.966 SOTA Frontier', fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=20, ha='right', fontsize=9.5, fontweight='bold')

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.0006, f'{score:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#111111')

ax.axhline(0.954, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.6, label='0.954 Public Ceiling')
ax.axhline(0.966, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.8, label='Current Leaderboard Rank #1 (0.966)')
ax.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
fig2_path = 'scratch/charts/leaderboard_roadmap.png'
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f'Saved Chart 2: {fig2_path}')


# ==============================================================================
# CHART 3: Calibration Curve (Super-Consensus Soft Probability vs Gold)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

pred_bins = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
observed_rates = [0.024, 0.132, 0.215, 0.346, 0.538, 0.570, 0.612, 0.680, 0.640, 0.870]

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration (y = x)')
ax.plot(pred_bins, observed_rates, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Super-Consensus Calibration')
ax.fill_between(pred_bins, observed_rates, pred_bins, color='#1f77b4', alpha=0.15)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('Predicted Probability Bin Center', fontsize=11, fontweight='bold')
ax.set_ylabel('Observed Gold Positive Rate', fontsize=11, fontweight='bold')
ax.set_title('Probability Calibration Reliability Curve\n(Super-Consensus Labels vs Expert Gold Ground Truth)', fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='upper left', frameon=True, fontsize=10)

for x, y in zip(pred_bins, observed_rates):
    ax.text(x, y + 0.03, f'{y*100:.1f}%', ha='center', fontsize=8.5, fontweight='bold', color='#1f77b4')

plt.tight_layout()
fig3_path = 'scratch/charts/calibration_curve.png'
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f'Saved Chart 3: {fig3_path}')

print('All 3 analytical charts successfully generated!')
