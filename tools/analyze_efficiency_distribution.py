import sys
import io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('scratch/efficiency_lb/full_leaderboard.csv')
print(f"Total entries on Efficiency Leaderboard: {len(df)}")
print(f"Unique teams: {df['TeamName'].nunique()}")

# Inspect top 25
print("\n--- TOP 25 ON EFFICIENCY LEADERBOARD ---")
top25 = df.head(25)[['EfficiencyRank', 'TeamName', 'PublicScore', 'DateSubmitted']]
print(top25.to_string(index=False))

# Filter teams with PublicScore >= 0.955
print("\n--- HIGH ACCURACY TEAMS (Score >= 0.955) ON EFFICIENCY LEADERBOARD ---")
high_acc = df[df['PublicScore'] >= 0.955].head(25)[['EfficiencyRank', 'TeamName', 'PublicScore', 'DateSubmitted']]
print(high_acc.to_string(index=False))

# Filter teams with PublicScore >= 0.958
print("\n--- TEAMS TIED WITH US (Score >= 0.958) ON EFFICIENCY LEADERBOARD ---")
top_score = df[df['PublicScore'] >= 0.958].head(25)[['EfficiencyRank', 'TeamName', 'PublicScore', 'DateSubmitted']]
print(top_score.to_string(index=False))

# Where does 0.956 land across the board?
v6_cohort = df[df['PublicScore'] == 0.956]
print(f"\nTeams scoring exactly 0.956: {len(v6_cohort)}")
print(f"Efficiency rank range for 0.956: Min={v6_cohort['EfficiencyRank'].min()}, Max={v6_cohort['EfficiencyRank'].max()}, Median={v6_cohort['EfficiencyRank'].median()}")
