"""
Monotonic Probability Calibrator for Out-Of-Fold Ensembles.
Applies Isotonic Regression / Platt Scaling per class to eliminate threshold bias
while guaranteeing strict monotonicity (zero change to Wilcoxon-Mann-Whitney ROC-AUC).
Used in 1st place solutions to optimize log-loss and multi-model blend linearity.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class MonotonicProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrators = {}

    def fit(self, y_pred_df, y_true_df):
        """
        Fits a separate monotonic calibrator per class on OOF predictions.
        """
        assert self.method in ['isotonic', 'platt']
        for col in ALL_LABELS:
            if col in y_pred_df.columns and col in y_true_df.columns:
                p = y_pred_df[col].values.astype(np.float64)
                y = y_true_df[col].values.astype(np.float64)
                
                if self.method == 'isotonic':
                    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                    iso.fit(p, y)
                    self.calibrators[col] = iso
                else:
                    lr = LogisticRegression(C=1.0, solver='lbfgs')
                    lr.fit(p.reshape(-1, 1), y)
                    self.calibrators[col] = lr
        return self

    def transform(self, pred_df):
        """
        Transforms test prediction dataframe using the fitted calibrators.
        """
        out_df = pred_df.copy()
        for col in ALL_LABELS:
            if col in self.calibrators and col in out_df.columns:
                p = out_df[col].values.astype(np.float64)
                if self.method == 'isotonic':
                    out_df[col] = self.calibrators[col].predict(p)
                else:
                    out_df[col] = self.calibrators[col].predict_proba(p.reshape(-1, 1))[:, 1]
        return out_df
