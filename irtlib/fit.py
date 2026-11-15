"""SVI fitting harness for the IRT ladder.

SVI (AutoNormal mean-field) is the production estimator: ~45k latent params over
~10^5 observations is minutes on CPU, where NUTS would be days — and the ladder needs
dozens of fits (LOMO x seeds). NUTS is used only as a validation spot-check elsewhere.
Multiple init seeds are run; the best-ELBO fit is kept.
"""
import time
import numpy as np
import jax
from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive, log_likelihood
from numpyro.infer.autoguide import AutoNormal


def fit_svi(model, args=(), kwargs=None, n_steps=20000, n_seeds=3, lr=0.01,
            n_post=400, init_scale=0.1, progress=False, clip_norm=10.0, max_backoff=3,
            init_values=None, lr_decay=None, guide_kind="normal", lowrank=8):
    """Fit `model` by SVI over `n_seeds` inits; keep the lowest final-ELBO fit.

    Self-healing: if every seed diverges (non-finite ELBO — the multidimensional M3 can
    do this at lr=0.01), the learning rate and init scale are halved and the whole set of
    seeds retried, up to `max_backoff` times. Deterministic PRNG keys so runs replay.
    Returns a dict: loss, seed, lr, losses, posterior (latent samples), timing.
    """
    kwargs = kwargs or {}
    t0 = time.time()
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    from numpyro.infer.initialization import init_to_value, init_to_median
    for attempt in range(max_backoff + 1):
        cur_lr = lr * (0.5 ** attempt)
        cur_init = init_scale * (0.5 ** attempt)
        best = None
        for seed in range(n_seeds):
            # default init (init_values None) is numpyro's init_to_uniform, exactly as before;
            # init_values="median" -> init_to_median; dict -> init_to_value (warm start)
            if init_values is None:
                gkw = {}
            elif isinstance(init_values, str) and init_values == "median":
                gkw = {"init_loc_fn": init_to_median()}
            else:
                gkw = {"init_loc_fn": init_to_value(values=init_values)}
            if guide_kind == "lowrank":
                guide = AutoLowRankMultivariateNormal(model, init_scale=cur_init, rank=lowrank, **gkw)
            else:
                guide = AutoNormal(model, init_scale=cur_init, **gkw)
            if lr_decay:
                # exponential decay to lr*lr_decay over the run (schedule = callable of step)
                from jax.example_libraries.optimizers import exponential_decay
                step_size = exponential_decay(cur_lr, n_steps, float(lr_decay))
            else:
                step_size = cur_lr
            svi = SVI(model, guide, numpyro.optim.ClippedAdam(step_size, clip_norm=clip_norm),
                      Trace_ELBO())
            res = svi.run(random.PRNGKey(seed), n_steps, *args, **kwargs, progress_bar=progress)
            final = float(np.asarray(res.losses)[-1])
            if not np.isfinite(final):
                continue
            if best is None or final < best["loss"]:
                best = {"loss": final, "seed": seed, "lr": cur_lr, "guide": guide,
                        "params": res.params, "losses": np.asarray(res.losses)}
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"all SVI seeds diverged after {max_backoff} backoffs (lr floor "
                           f"{lr * 0.5 ** max_backoff:.4g})")
    pred = Predictive(best["guide"], params=best["params"], num_samples=n_post)
    latent = pred(random.PRNGKey(9000 + best["seed"]), *args, **kwargs)
    best["posterior"] = latent           # PURE guide sample sites -> feed to pointwise_loglik
    # derived quantities (theta, a, loadings, u) for inspection: run the model on the
    # latents to materialise its deterministic sites. Predictive-obs draws are dropped.
    det = Predictive(model, posterior_samples=latent)(random.PRNGKey(1), *args, **kwargs)
    best["derived"] = {k: v for k, v in det.items() if k not in latent}
    best["seconds"] = round(time.time() - t0, 1)
    try:
        best["median"] = {k: np.asarray(v) for k, v in best["guide"].median(best["params"]).items()}
    except Exception:
        best["median"] = None
    return best


def fit_nuts(model, args=(), kwargs=None, n_warmup=500, n_samples=500, seed=0):
    """NUTS reference fit (validation tier). Only affordable on small problems — M0-full
    or M2 on a small clip sub-matrix. Kwargs are passed through unchanged (subsampling
    defaults to off in the models; M0 has no subsample parameter at all).
    """
    from numpyro.infer import MCMC, NUTS
    kwargs = dict(kwargs or {})
    mcmc = MCMC(NUTS(model), num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    t0 = time.time()
    mcmc.run(random.PRNGKey(seed), *args, **kwargs)
    return {"samples": mcmc.get_samples(), "seconds": round(time.time() - t0, 1), "mcmc": mcmc}


def pointwise_loglik(model, posterior, args=(), kwargs=None):
    """Combined per-observation log-likelihood, shape (n_samples, N_total).

    Concatenates all likelihood sites (M2/M3 split obs into y_bin + y_tern) into a
    single observation axis for arviz WAIC/LOO.
    """
    kwargs = {**(kwargs or {}), "subsample": None}     # LOO always on FULL data
    # keep only latent (sample) sites for conditioning
    latent = {k: v for k, v in posterior.items() if not k.startswith("_")}
    ll = log_likelihood(model, latent, *args, **kwargs)
    blocks = [np.asarray(ll[k]) for k in sorted(ll)]
    out = np.concatenate(blocks, axis=1)               # (S, N_total)
    # OrderedLogistic can underflow a category prob to exactly 0 (log=-inf) even with the
    # in-model logit clip; floor per-observation log-lik so WAIC stays finite (prob ~2e-22).
    return np.clip(np.nan_to_num(out, neginf=-50.0), -50.0, 0.0)


def elbo_seed_spread(model, args=(), kwargs=None, n_steps=6000, n_seeds=5, lr=0.01,
                     init_scale=0.1):
    """Final ELBO for each seed — a cheap SVI-stability diagnostic (C2 robustness)."""
    kwargs = kwargs or {}
    out = []
    for seed in range(n_seeds):
        guide = AutoNormal(model, init_scale=init_scale)
        svi = SVI(model, guide, numpyro.optim.ClippedAdam(lr, clip_norm=10.0), Trace_ELBO())
        res = svi.run(random.PRNGKey(seed), n_steps, *args, **kwargs, progress_bar=False)
        out.append(float(np.asarray(res.losses)[-1]))
    return out
