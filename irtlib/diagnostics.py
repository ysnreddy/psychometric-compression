"""Model comparison (WAIC, computed directly) and posterior-predictive checks.

We compute WAIC manually from the pointwise log-likelihood (lppd - p_waic) rather than via
arviz PSIS-LOO: it needs no Pareto tail-fitting (which fails with "all tail values the same"
when the graded likelihood is locally degenerate), is far faster on 10^5 observations, and
is a valid IC for the M2-vs-M3 comparison. WAIC on SVI posteriors is approximate — the
p_waic tail fraction is reported as the honesty diagnostic — and the PPCs (fitted models
reproduce observed per-family category marginals) are the more defensible evidence at few
respondents, so both are reported.
"""
import numpy as np
from scipy.special import logsumexp


def waic_pointwise(ll):
    """ll: (n_samples, N_obs) -> (elpd_i, p_waic_i) per observation."""
    S = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(S)      # log pointwise predictive density
    p_waic_i = ll.var(axis=0)                        # effective-parameter penalty
    return lppd_i - p_waic_i, p_waic_i


def elpd_summary(ll):
    """Scalar WAIC summary + fraction of high-variance (problematic) observations."""
    elpd_i, p_i = waic_pointwise(ll)
    n = ll.shape[1]
    return {"elpd_waic": float(elpd_i.sum()), "elpd_loo": float(elpd_i.sum()),  # alias kept
            "p_waic": float(p_i.sum()), "se": float(np.sqrt(n) * elpd_i.std()),
            "n_obs": int(n), "pareto_k_bad_frac": round(float((p_i > 0.4).mean()), 4)}


def compare_models(loglik_by_model):
    """{name: (S, N) log-lik on the SAME obs set} -> ranked comparison with paired SE.

    Uses per-observation elpd differences for the standard error of the difference (the
    correct paired comparison; obs sets must be identical and aligned).
    """
    elpds = {name: waic_pointwise(ll)[0] for name, ll in loglik_by_model.items()}
    totals = {name: float(e.sum()) for name, e in elpds.items()}
    best = max(totals, key=totals.get)
    out = {}
    for name, e in elpds.items():
        diff_i = elpds[best] - e
        out[name] = {"elpd_waic": round(totals[name], 1),
                     "elpd_diff": round(float(diff_i.sum()), 1),
                     "dse": round(float(np.sqrt(len(e)) * diff_i.std()), 1)}
    return {"best": best, "models": out}


# --------------------------------------------------------------------------- PPC
def ppc_category_marginals(posterior, model, args, kwargs, y_obs_blocks):
    """Posterior-predictive category marginals vs observed, per likelihood block.

    predict=True is essential: Predictive does NOT resample observed sites, so without it
    the "posterior predictive" simply echoes the data and the check is vacuous.
    """
    from numpyro.infer import Predictive
    from jax import random
    kwargs = {**(kwargs or {}), "subsample": None, "predict": True}
    latent = {k: v for k, v in posterior.items() if not k.startswith("_")}
    sim = Predictive(model, posterior_samples=latent)(random.PRNGKey(123), *args, **kwargs)
    out = {}
    for site, obs in y_obs_blocks.items():
        if site not in sim:
            continue
        rep = np.asarray(sim[site])
        kmax = int(max(int(obs.max()), int(rep.max()))) + 1
        out[site] = {
            "observed": (np.bincount(obs, minlength=kmax) / len(obs)).round(4).tolist(),
            "predicted": (np.bincount(rep.reshape(-1), minlength=kmax) / rep.size).round(4).tolist(),
        }
    return out
