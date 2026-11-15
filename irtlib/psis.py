"""Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao, Gabry 2015/2024) —
minimal numpy implementation (arviz 1.x on this machine no longer exposes psislw).

Used as the variational-inference diagnostic of Yao, Vehtari, Simpson & Gelman (2018),
"Yes, but did it work?": draw S samples z_s from the guide q, compute log importance
ratios r_s = log p(z_s, y) - log q(z_s), and estimate the shape k of a generalised Pareto
fitted to the right tail of exp(r_s). k < 0.5 : q is close enough for finite-variance
importance sampling; 0.5-0.7 : usable but heavy-tailed; k > 0.7 : the variational
approximation is far from the posterior (importance correction is unreliable and point
estimates from q should be treated with caution).
"""
import numpy as np


def gpdfit(x):
    """Zhang & Stephens (2009) estimator of the generalised Pareto (k, sigma) for sorted x>0."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    prior = 3.0
    m = 30 + int(np.sqrt(n))
    b = 1.0 / x[-1] + (1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / prior / x[int((n - 1) / 4 + 0.5)]
    # profile log-likelihood for each b (loo::gpdfit convention: k = mean(log1p(-b x)))
    with np.errstate(divide="ignore", invalid="ignore"):
        k_b = np.array([np.mean(np.log1p(-bi * x)) for bi in b])
        l_b = n * (np.log(-b / k_b) - k_b - 1.0)
        w = 1.0 / np.array([np.sum(np.exp(l_b - lb)) for lb in l_b])
    w = np.nan_to_num(w); w = w / w.sum()
    b_hat = np.sum(b * w)
    k_hat = float(np.mean(np.log1p(-b_hat * x)))
    sigma = -k_hat / b_hat
    # weakly informative prior on k as in loo (regularise toward 0.5 for small n)
    k_hat = (n * k_hat + prior * 0.5) / (n + prior)
    return float(k_hat), float(sigma)


def gpinv(p, k, sigma):
    """Inverse CDF of the generalised Pareto."""
    p = np.asarray(p, float)
    if abs(k) < 1e-12:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psislw(log_weights):
    """Smooth log importance weights. Returns (smoothed normalised log weights, k_hat)."""
    lw = np.asarray(log_weights, float)
    S = len(lw)
    lw = lw - lw.max()
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    if M < 5:
        return lw - np.log(np.sum(np.exp(lw))), float("nan")
    order = np.argsort(lw)
    tail_idx = order[-M:]
    cutoff = lw[order[-M - 1]]
    tail = np.exp(lw[tail_idx]) - np.exp(cutoff)
    if np.all(tail <= 0):
        return lw - np.log(np.sum(np.exp(lw))), float("nan")
    k, sigma = gpdfit(tail[tail > 0])
    if np.isfinite(k):
        # replace tail with expected order statistics of the fitted GPD
        p = (np.arange(1, M + 1) - 0.5) / M
        smoothed = np.log(gpinv(p, k, sigma) + np.exp(cutoff))
        lw_new = lw.copy()
        lw_new[tail_idx[np.argsort(lw[tail_idx])]] = smoothed
        lw_new = np.minimum(lw_new, 0.0)         # truncate at max original weight (which is 0)
        lw = lw_new
    return lw - np.log(np.sum(np.exp(lw))), float(k)
