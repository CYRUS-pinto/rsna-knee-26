import numpy as np
import pandas as pd
import torch

def rank_vec(v: np.ndarray) -> np.ndarray:
    """Empirical percentile rank in [1/N, 1.0]."""
    n = len(v)
    if n <= 1 or np.isnan(v).any() or np.std(v) < 1e-8:
        return np.full_like(v, 0.5, dtype=np.float32)
    return pd.Series(v).rank(pct=True).values.astype(np.float32)

def norm_ppf_vec(p: np.ndarray) -> np.ndarray:
    """Acklam rational approximation for Probit: Phi^-1(p)."""
    p_safe = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    out = np.zeros_like(p_safe, dtype=np.float64)

    m_low = p_safe < plow
    if np.any(m_low):
        q = np.sqrt(-2.0 * np.log(p_safe[m_low]))
        out[m_low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /                      ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    m_mid = (p_safe >= plow) & (p_safe <= phigh)
    if np.any(m_mid):
        q = p_safe[m_mid] - 0.5
        r = q * q
        out[m_mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /                      (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    m_high = p_safe > phigh
    if np.any(m_high):
        q = np.sqrt(-2.0 * np.log(1.0 - p_safe[m_high]))
        out[m_high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /                        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    return out.astype(np.float32)

def norm_cdf_vec(z: np.ndarray) -> np.ndarray:
    """Gaussian CDF: Phi(z)."""
    t = torch.from_numpy(np.asarray(z, dtype=np.float32))
    return (0.5 * (1.0 + torch.erf(t / 1.41421356237))).numpy()

def power_probit_blend(v_base: np.ndarray, v_aux: np.ndarray, w_aux: float, p_exp: float = 1.0, probit_alpha: float = 0.35) -> np.ndarray:
    w_base = 1.0 - w_aux
    b_pow = np.power(np.clip(v_base, 1e-4, 0.9999), p_exp)
    a_pow = np.power(np.clip(v_aux, 1e-4, 0.9999), p_exp)
    lin_blend = w_base * b_pow + w_aux * a_pow

    try:
        n = max(len(v_base), 10)
        b_open = np.clip(v_base * (1.0 - 1.0 / n) + (0.5 / n), 1e-4, 1.0 - 1e-4)
        a_open = np.clip(v_aux * (1.0 - 1.0 / n) + (0.5 / n), 1e-4, 1.0 - 1e-4)
        z_blend = w_base * norm_ppf_vec(b_open) + w_aux * norm_ppf_vec(a_open)
        probit_prob = norm_cdf_vec(z_blend)
        out = (1.0 - probit_alpha) * lin_blend + probit_alpha * probit_prob
        return out.astype(np.float32)
    except Exception:
        return lin_blend.astype(np.float32)
