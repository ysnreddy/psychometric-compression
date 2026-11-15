"""Score reconstruction from a subset (p-IRT / gp-IRT analogues for graded responses).

Given frozen item parameters and a held-out model's responses on a subset of items,
estimate its ability and predict its FULL-benchmark score. Three estimators are reported:
  irt          — model-based expected full score at the estimated theta (p-IRT analogue)
  subset_mean  — naive mean observed score on the subset
  gp_irt       — shrinkage blend lam*irt + (1-lam)*subset_mean (lam tuned on training)

Confirmatory simple structure makes M3 reconstruction decouple: each ability dimension is
estimated from the subset items loading on it (independent 1-D estimates).
"""
import numpy as np
from . import params as P


def _probs_subset(a, b, k0, k1, uc, keff, theta, link="grm"):
    eta = a * theta + uc
    pc = P._sig(eta - b)
    P0 = 1.0 - pc; P1 = np.zeros_like(pc); P2 = pc
    if link == "pcm":
        T0, T1, T2 = P._pcm_probs(eta, k0, k1)
    else:
        s0 = P._sig(eta - k0); s1 = P._sig(eta - k1)
        T0, T1, T2 = 1 - s0, s0 - s1, s1
    tern = keff == 3
    P0 = np.where(tern, T0, P0)
    P1 = np.where(tern, T1, P1)
    P2 = np.where(tern, T2, P2)
    eps = 1e-9
    return np.clip(P0, eps, 1), np.clip(P1, eps, 1), np.clip(P2, eps, 1)


def _theta_grid_mle(a, b, k0, k1, uc, keff, y, grid, link="grm", method="map", prior_sd=1.0):
    """1-D grid MAP of ability given subset responses (weakly N(0,1)-penalised).

    Category->probability mapping must respect keff: for BINARY items (keff==2) the
    correct response is coded y==1 == top category -> P2, NOT the (empty) partial slot P1.
    Mapping y==1 -> P1 unconditionally silently zeroed every correct binary response, so
    binary items could only push theta down — under an all-binary grading map the
    likelihood became monotone and theta railed at the grid boundary.
    """
    if len(y) == 0:
        return 0.0
    top = keff - 1                                   # index of the correct category per item
    lls = np.empty(len(grid))
    for gi, g in enumerate(grid):
        P0, P1, P2 = _probs_subset(a, b, k0, k1, uc, keff, g, link=link)
        py = np.where(y == 0, P0, np.where(y == top, P2, P1))
        lls[gi] = np.log(py).sum() - 0.5 * (g / prior_sd) ** 2      # N(0, prior_sd) prior
    if method == "eap":
        w = np.exp(lls - lls.max()); w /= w.sum()
        return float((w * np.asarray(grid)).sum())
    return float(np.asarray(grid)[int(np.argmax(lls))])


def estimate_theta(p: P.ItemParams, idx_sub, y_sub, grid=None, method="map"):
    """Estimate ability from subset responses. Returns scalar (1-D) or (n_dim,) vector.

    Grid spans +-6: +-4 railed the weakest model (and the weakly-identified safety
    dimension) at the boundary, collapsing rank information. method: "map" (grid maximum,
    default) or "eap" (posterior mean on the grid); prior sd and link come from p."""
    grid = np.linspace(-6, 6, 121) if grid is None else grid
    a, b, k0, k1 = p.a[idx_sub], p.b[idx_sub], p.k0[idx_sub], p.k1[idx_sub]
    uc, keff = p.u[p.clip[idx_sub]], p.keff[idx_sub]
    link = getattr(p, "link", "grm"); psd = float(getattr(p, "prior_sd", 1.0))
    if p.item_dim is None or p.n_dim == 1:
        return _theta_grid_mle(a, b, k0, k1, uc, keff, np.asarray(y_sub), grid, link, method, psd)
    dims = p.item_dim[idx_sub]
    theta = np.zeros(p.n_dim)
    for d in range(p.n_dim):
        m = dims == d
        theta[d] = _theta_grid_mle(a[m], b[m], k0[m], k1[m], uc[m], keff[m],
                                   np.asarray(y_sub)[m], grid, link, method, psd)
    return theta


def predicted_full_score(p: P.ItemParams, theta):
    """Model-based expected full-benchmark score at `theta` (mean over ALL items)."""
    tv = P.theta_for_items(p, theta)
    return float(P.expected_score(p, tv).mean())


def subset_mean_score(y_sub, keff_sub):
    """Naive observed mean score on the subset (wrong=0, partial=0.5, correct=1).

    y is the ordinal code (0=wrong,1=partial,2=correct for ternary; 0/1 for binary where
    1==correct). Map to [0,1]: score = y/(keff-1).
    """
    y = np.asarray(y_sub, float)
    return float((y / (np.asarray(keff_sub) - 1)).mean()) if len(y) else 0.0


def reconstruct(p: P.ItemParams, idx_sub, y_sub, lam=0.7, method="map"):
    """All three estimators for one held-out model. Returns dict of predicted full scores."""
    theta = estimate_theta(p, idx_sub, y_sub, method=method)
    irt = predicted_full_score(p, theta)
    sm = subset_mean_score(y_sub, p.keff[idx_sub])
    return {"theta": theta, "irt": irt, "subset_mean": sm,
            "gp_irt": lam * irt + (1 - lam) * sm}


def true_full_score(p: P.ItemParams, y_full, idx_full=None):
    """Observed full-benchmark score for a model (ground truth for MAE), from its full
    responses. idx_full defaults to all items; y_full aligned to it."""
    idx = np.arange(len(p.a)) if idx_full is None else idx_full
    return subset_mean_score(y_full, p.keff[idx])
