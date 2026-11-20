"""Learn-then-Test (LtT) calibration for selective text-to-SQL risk control.

Given a confidence score s in [0,1] and a target risk tolerance alpha, LtT selects an
abstention threshold lambda such that the selective risk (error rate among ANSWERED queries)
is <= alpha with probability >= 1 - delta, distribution-free, by treating each candidate
threshold as a hypothesis and controlling the family-wise error rate (Bonferroni).

Reference: Angelopoulos, Bates, Candes, Jordan, Lei (2021), "Learn then Test".
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binom


def _hb_pvalue(r_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: true selective risk > alpha.
    Small p => evidence that risk <= alpha."""
    if n == 0:
        return 1.0
    k = int(np.ceil(r_hat * n))
    p_bentkus = np.e * binom.cdf(k, n, alpha)
    if r_hat < alpha:
        if r_hat > 0:
            h = (r_hat * np.log(r_hat / alpha + 1e-12)
                 + (1 - r_hat) * np.log((1 - r_hat) / (1 - alpha) + 1e-12))
        else:
            h = np.log(1 / (1 - alpha))
        p_hoeff = np.exp(-n * h)
    else:
        p_hoeff = 1.0
    return float(min(1.0, min(p_bentkus, p_hoeff)))


def calibrate_threshold(conf_cal, correct_cal, alpha=0.10, delta=0.10,
                        n_grid=100, min_answered=20):
    """LtT with Bonferroni correction over the threshold grid.

    Returns the certified threshold of MAXIMUM coverage, or None (=> abstain entirely /
    NoCertificate) if no threshold can be certified.
    """
    conf_cal = np.asarray(conf_cal, float)
    correct_cal = np.asarray(correct_cal, int)
    grid = np.linspace(0.0, 1.0, n_grid)
    delta_corrected = delta / len(grid)  # Bonferroni FWER control

    best_lam, best_cov = None, -1.0
    for lam in grid:
        answered = conf_cal >= lam
        n_ans = int(answered.sum())
        if n_ans < min_answered:
            continue
        risk_hat = 1.0 - correct_cal[answered].mean()
        p = _hb_pvalue(risk_hat, n_ans, alpha)
        if p <= delta_corrected:
            cov = n_ans / len(correct_cal)
            if cov > best_cov:
                best_cov, best_lam = cov, float(lam)
    return best_lam
