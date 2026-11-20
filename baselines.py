"""Distribution-free risk-control backends for selective text-to-SQL.

Each backend takes a calibration split (conf, correct), a tolerance alpha, and (for the
high-probability backends) a confidence level delta, and returns an abstention threshold
lambda, or None to abstain entirely (NoCertificate). All are compared on held-out data.

Backends:
  heuristic : empirical-risk thresholding (NO guarantee).
  crc       : Conformal Risk Control (Angelopoulos et al. 2024) - MARGINAL guarantee E[risk]<=alpha.
  rcps      : Risk-Controlling Prediction Sets (Bates et al. 2021) - HIGH-PROBABILITY guarantee.
  (Learn-then-Test lives in conformal.calibrate_threshold - HIGH-PROBABILITY via multiple testing.)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binom


def _hb_upper(r_hat, n, delta):
    """(1-delta) Hoeffding-Bentkus UPPER confidence bound on the true risk, via binary search."""
    if n == 0:
        return 1.0

    def pval(a):
        k = int(np.ceil(r_hat * n))
        p_bent = np.e * binom.cdf(k, n, a)
        if r_hat < a:
            if r_hat > 0:
                h = r_hat * np.log(r_hat / a + 1e-12) + (1 - r_hat) * np.log((1 - r_hat) / (1 - a) + 1e-12)
            else:
                h = np.log(1 / (1 - a))
            p_hoef = np.exp(-n * h)
        else:
            p_hoef = 1.0
        return min(1.0, min(p_bent, p_hoef))

    lo, hi = r_hat, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if pval(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def heuristic(conf, correct, alpha=0.10, n_min=20, m=100):
    conf = np.asarray(conf, float); correct = np.asarray(correct, int)
    best = None
    for lam in np.linspace(0, 1, m):
        ans = conf >= lam; na = int(ans.sum())
        if na < n_min:
            continue
        if (1 - correct[ans].mean()) <= alpha and (best is None or lam < best):
            best = float(lam)
    return best


def crc(conf, correct, alpha=0.10, n_min=20, m=100):
    """Conformal Risk Control: lambda = inf{ (n*Rhat + B)/(n+1) <= alpha }, B=1 for 0/1 loss.
    Controls the EXPECTED (marginal) risk, not per-deployment."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, int)
    best = None; B = 1.0
    for lam in np.linspace(0, 1, m):
        ans = conf >= lam; na = int(ans.sum())
        if na < n_min:
            continue
        r_hat = 1 - correct[ans].mean()
        if (na * r_hat + B) / (na + 1) <= alpha and (best is None or lam < best):
            best = float(lam)
    return best


def rcps(conf, correct, alpha=0.10, delta=0.10, n_min=20, m=100):
    """Risk-Controlling Prediction Sets: monotone risk -> pointwise UCB, no multiplicity
    correction. Returns the lowest threshold whose risk-UCB <= alpha. HIGH-PROBABILITY."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, int)
    best = None
    for lam in np.linspace(0, 1, m):
        ans = conf >= lam; na = int(ans.sum())
        if na < n_min:
            continue
        r_hat = 1 - correct[ans].mean()
        if _hb_upper(r_hat, na, delta) <= alpha and (best is None or lam < best):
            best = float(lam)
    return best


def evaluate(conf, correct, lam):
    """Apply a threshold on held-out data; return coverage, realised risk, n answered."""
    if lam is None:
        return None
    conf = np.asarray(conf, float); correct = np.asarray(correct, int)
    ans = conf >= lam; na = int(ans.sum())
    if na == 0:
        return None
    return {"coverage": na / len(correct), "risk": float(1 - correct[ans].mean()), "n": na}
