"""Evaluation harness: dense response tables, subset-reconstruction scoring, compression
curves (MAE / Kendall-tau / close-pair accuracy vs subset size), and LOMO rotations.

EA.2 compares SELECTION methods at a fixed full-data item model (in-sample item params —
standard for method comparison). EA.3 is the honest generalisation test: leave-one-MODEL-out
refits of item params, reconstructing the held-out model from its subset responses.
"""
import numpy as np
from scipy.stats import kendalltau
from . import reconstruct as R
from . import params as P


def refine_abilities(item_params, fd, Y=None, method="map"):
    """Re-estimate every respondent's ability by frozen-item MLE with the clip effect
    marginalised to 0. Decouples ability from the mean-field ability-clip posterior
    correlation (which otherwise scrambles the leaderboard). Same estimator the held-out
    reconstruction uses, so the two are consistent. Returns (M,) or (M, n_dim)."""
    Y = response_table(fd) if Y is None else Y
    D = item_params.n_dim
    theta = np.zeros((fd.n_resp, D)) if D > 1 else np.zeros(fd.n_resp)
    for i in range(fd.n_resp):
        yi = Y[i]; present = yi >= 0
        if present.sum() == 0:
            continue
        theta[i] = R.estimate_theta(item_params, np.where(present)[0], yi[present], method=method)
    return theta


# --------------------------------------------------------------------------- response tables
def response_table(fd):
    """Dense (n_resp, n_item) ordinal responses; -1 = unobserved (structural miss)."""
    Y = np.full((fd.n_resp, fd.n_item), -1, dtype=np.int64)
    Y[fd.resp, fd.item] = fd.y
    return Y


def base_indices(fd):
    gs = set(fd.group_names)
    return [i for i, n in enumerate(fd.resp_names) if n in gs]


def true_full_scores(fd, p, Y=None):
    """Observed full-benchmark score per base model (ground truth for MAE)."""
    Y = response_table(fd) if Y is None else Y
    out = {}
    for i in base_indices(fd):
        present = Y[i] >= 0
        out[fd.resp_names[i]] = R.subset_mean_score(Y[i][present], p.keff[present])
    return out


# --------------------------------------------------------------------------- subset evaluation
def _loo_affine(raw, true):
    """Leave-one-out affine calibration: predict each entry from a line fit on the others.
    Removes the IRT score's systematic scale/offset bias without peeking at its own label."""
    raw, true = np.asarray(raw, float), np.asarray(true, float)
    out = np.zeros_like(raw)
    for i in range(len(raw)):
        m = np.arange(len(raw)) != i
        if m.sum() >= 2 and np.std(raw[m]) > 1e-9:
            b, a = np.polyfit(raw[m], true[m], 1)
            out[i] = a + b * raw[i]
        else:
            out[i] = true[m].mean() if m.any() else raw[i]
    return out


def eval_subset(fd, p, S, true_full, Y=None, estimator="gp_irt", lam=0.7):
    """Reconstruct every base model from its responses on clip-subset S; return metrics.

    S is a set of CLIP indices; items whose clip is in S form the item subset. The
    'irt_cal' / 'gp_cal' estimators apply leave-one-out affine calibration to the IRT
    prediction (fixes the frozen-item theta scale bias).
    """
    Y = response_table(fd) if Y is None else Y
    Sset = set(int(c) for c in S)
    item_in_S = np.array([c in Sset for c in p.clip])
    S_items = np.where(item_in_S)[0]
    names, true = [], []
    est = {}
    for i in base_indices(fd):
        yi = Y[i]
        present = yi >= 0
        idx = S_items[present[S_items]]
        rec = R.reconstruct(p, idx, yi[idx], lam=lam)
        nm = fd.resp_names[i]
        est[nm] = rec
        names.append(nm); true.append(true_full[nm])
    true = np.array(true)
    # calibrated variants (LOO affine on the IRT prediction)
    irt_raw = np.array([est[n]["irt"] for n in names])
    sm = np.array([est[n]["subset_mean"] for n in names])
    irt_cal = _loo_affine(irt_raw, true)
    for k, nm in enumerate(names):
        est[nm]["irt_cal"] = float(irt_cal[k])
        est[nm]["gp_cal"] = float(lam * irt_cal[k] + (1 - lam) * sm[k])
    pred = np.array([est[n][estimator] for n in names])
    abs_errs = np.abs(pred - true)
    mae = float(abs_errs.mean())
    tau = float(kendalltau(pred, true).statistic) if len(true) > 2 else float("nan")
    # close-pair ranking accuracy: pairs whose TRUE scores differ by <3pp, correctly ordered
    cp_tot = cp_ok = 0
    for x in range(len(true)):
        for y in range(x + 1, len(true)):
            if abs(true[x] - true[y]) < 0.03:
                cp_tot += 1
                cp_ok += int(np.sign(pred[x] - pred[y]) == np.sign(true[x] - true[y]))
    close_pair_acc = float(cp_ok / cp_tot) if cp_tot else float("nan")
    return {"mae": mae, "kendall_tau": tau, "n_items": int(len(S_items)),
            "close_pair_acc": close_pair_acc, "n_close_pairs": cp_tot,
            "abs_errs": abs_errs.tolist(),
            "estimates": {k: {"pred": round(v[estimator], 4), "true": round(true_full[k], 4)}
                          for k, v in est.items()}}


def compression_curve(fd, p, selector, sizes, n_clips, seeds=(0,), estimator="gp_irt"):
    """Run a clip selector across subset sizes (mean over seeds for stochastic selectors).

    selector(n, seed) -> array of clip indices. Returns list of per-size metric dicts.
    """
    Y = response_table(fd)
    true_full = true_full_scores(fd, p, Y)
    out = []
    for n in sizes:
        maes, taus, cpas, nit, all_errs = [], [], [], [], []
        for s in seeds:
            S = selector(n, s)
            m = eval_subset(fd, p, S, true_full, Y=Y, estimator=estimator)
            maes.append(m["mae"]); taus.append(m["kendall_tau"])
            cpas.append(m["close_pair_acc"]); nit.append(m["n_items"])
            all_errs.extend(m["abs_errs"])
        _, lo, hi = bootstrap_ci(all_errs)                 # over model x seed abs errors
        out.append({"n_clips": int(n), "mae_mean": float(np.mean(maes)),
                    "mae_sd": float(np.std(maes)),
                    "mae_ci95": [round(lo, 4), round(hi, 4)],
                    "tau_mean": float(np.nanmean(taus)),
                    "tau_sd": float(np.nanstd(taus)), "close_pair_acc": float(np.nanmean(cpas)),
                    "n_items_mean": float(np.mean(nit)), "n_seeds": len(seeds)})
    return out


# --------------------------------------------------------------------------- bootstrap CI
def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return (float(np.mean(v)), float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)))
