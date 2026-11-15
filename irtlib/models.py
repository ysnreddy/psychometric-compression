"""NumPyro IRT model ladder M0-M3 for a structured multi-field benchmark.

M0  clip-level dichotomous 2PL           (tinyBenchmarks-style baseline)
M1  field-level dichotomous 2PL          (fields as independent items, hierarchical by family)
M2  field-level graded-response + clip random effect + family-pooled item priors
M3  confirmatory multidimensional M2      (one dimension per capability family)

Ability grouping: respondents may be grouped by base model (pseudo-respondents share a
base). use_group=True gives theta_m = theta_group[g(m)] + sigma_resp * z_m — the
correlation-aware form that lets perturbation pseudo-respondents stabilise ITEM
parameters without injecting independent ability points. use_group=False (base-only,
groups==respondents) uses flat theta_m ~ N(0,1).
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints


# --------------------------------------------------------------------------- ability
def _ability_1d(n_resp, n_group, resp_group, use_group, theta_scale=1.0):
    if not use_group:
        z = numpyro.sample("z_theta", dist.Normal(0.0, 1.0).expand([n_resp]))
        return numpyro.deterministic("theta", theta_scale * z)      # deterministic -> lands in fit["derived"]
    theta_g = numpyro.sample("theta_group", dist.Normal(0.0, 1.0).expand([n_group]))
    sig = numpyro.sample("sigma_resp", dist.HalfNormal(0.3))
    z = numpyro.sample("z_resp", dist.Normal(0.0, 1.0).expand([n_resp]))
    return numpyro.deterministic("theta", theta_g[resp_group] + sig * z)


def _ability_md(n_resp, n_group, resp_group, D, use_group, theta_scale=1.0):
    L = numpyro.sample("L_corr", dist.LKJCholesky(D, concentration=2.0))
    if not use_group:
        # independent per-respondent ability ~ MVN(0, corr). Pseudo-respondents get their own
        # theta (unused for the leaderboard) but still contribute rows to item estimation, so
        # the base-model abilities stay clean (no shrinkage toward pseudo-contaminated means).
        z = numpyro.sample("z_theta", dist.Normal(0.0, 1.0).expand([n_resp, D]))
        return numpyro.deterministic("theta", theta_scale * (z @ L.T))
    zg = numpyro.sample("z_theta_group", dist.Normal(0.0, 1.0).expand([n_group, D]))
    theta_g = zg @ L.T
    numpyro.deterministic("theta_group", theta_g)
    sig = numpyro.sample("sigma_resp", dist.HalfNormal(0.3))
    zr = numpyro.sample("z_resp", dist.Normal(0.0, 1.0).expand([n_resp, D]))
    return numpyro.deterministic("theta", theta_g[resp_group] + sig * zr)


# --------------------------------------------------------------------------- hierarchical item priors
def _hier_discrimination(n_item, n_family, item_family):
    mu = numpyro.sample("mu_log_a", dist.Normal(0.0, 0.5).expand([n_family]))
    sig = numpyro.sample("sigma_log_a", dist.HalfNormal(0.5).expand([n_family]))
    z = numpyro.sample("z_log_a", dist.Normal(0.0, 1.0).expand([n_item]))
    # bound log-discrimination -> a in [~0.05, ~8]; prevents runaway exp() that destabilises
    # the multidimensional fit while spanning the full plausible discrimination range.
    log_a = jnp.clip(mu[item_family] + sig[item_family] * z, -3.0, 2.08)
    return numpyro.deterministic("a", jnp.exp(log_a))


def _hier_difficulty(n_item, n_family, item_family, name="b"):
    mu = numpyro.sample(f"mu_{name}", dist.Normal(0.0, 1.0).expand([n_family]))
    sig = numpyro.sample(f"sigma_{name}", dist.HalfNormal(1.0).expand([n_family]))
    z = numpyro.sample(f"z_{name}", dist.Normal(0.0, 1.0).expand([n_item]))
    return numpyro.deterministic(name, mu[item_family] + sig[item_family] * z)


# --------------------------------------------------------------------------- M0
def m0(cd_resp, cd_clip, cd_y, n_resp, n_clip, n_group, resp_group, use_group=False):
    theta = _ability_1d(n_resp, n_group, resp_group, use_group)
    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([n_clip]))
    log_a = numpyro.sample("log_a", dist.Normal(0.0, 0.5).expand([n_clip]))
    a = numpyro.deterministic("a", jnp.exp(log_a))
    eta = a[cd_clip] * (theta[cd_resp] - b[cd_clip])
    with numpyro.plate("obs", cd_y.shape[0]):
        numpyro.sample("y", dist.Bernoulli(logits=eta), obs=cd_y)


# --------------------------------------------------------------------------- subsampling helper
def _sub(n, frac):
    """Per-block subsample size for mini-batch SVI (None/1.0 -> full data)."""
    if frac is None or frac >= 1.0:
        return None
    return max(1, int(round(n * frac)))


# --------------------------------------------------------------------------- M1
def m1(resp, item, y, keff, n_resp, n_item, n_family, n_group,
       item_family, resp_group, use_group=False, subsample=None):
    theta = _ability_1d(n_resp, n_group, resp_group, use_group)
    a = _hier_discrimination(n_item, n_family, item_family)
    b = _hier_difficulty(n_item, n_family, item_family)
    N = y.shape[0]
    with numpyro.plate("obs", N, subsample_size=_sub(N, subsample), dim=-1) as ix:
        eta = jnp.clip(a[item[ix]] * (theta[resp[ix]] - b[item[ix]]), -20, 20)
        ybin = (y[ix] == (keff[ix] - 1)).astype(jnp.int32)  # correct=1 else 0
        numpyro.sample("y", dist.Bernoulli(logits=eta), obs=ybin)


# --------------------------------------------------------------------------- M2 (graded + clip effect)
def _grm_likelihood(resp, item, clip, y, idx_bin, idx_tern, a, theta, u, b, kappa0, kappa1,
                    item_dim=None, subsample=None, predict=False, c=None, item_xdim=None,
                    link="grm"):
    """Split binary (Bernoulli) and ternary (OrderedLogistic) blocks with gathers done
    inside subsampled plates (mini-batch SVI). item_dim!=None -> multidimensional ability.

    idx_bin / idx_tern are precomputed static obs-index arrays (JIT-safe). subsample is a
    fraction in (0,1] applied per block; None/1.0 -> full data (used for log-likelihood).
    predict=True passes obs=None so Predictive actually RESAMPLES the response sites —
    without it a "posterior predictive" just echoes the observed data (vacuous PPC).
    """
    def ability(gidx):
        return theta[resp[gidx]] if item_dim is None else theta[resp[gidx], item_dim[item[gidx]]]

    def linpred(gidx):
        """a_j * theta_{m,d(j)} (+ c_j * theta_{m,x(j)} for cross-loaded items)."""
        lp = a[item[gidx]] * ability(gidx)
        if c is not None:
            lp = lp + c[item[gidx]] * theta[resp[gidx], item_xdim[item[gidx]]]
        return lp

    # clip logits/predictors so category probabilities never underflow to exactly 0
    # (log(0)=-inf -> NaN ELBO). +-20 => prob floor ~2e-9; only bites in the extreme tail.
    # Empty blocks are skipped: the strict grading map makes EVERY item binary, so the
    # ternary block can be size 0 (a zero-size plate is an error in numpyro).
    Nb, Nt = idx_bin.shape[0], idx_tern.shape[0]
    if Nb > 0:
        with numpyro.plate("obs_bin", Nb, subsample_size=_sub(Nb, subsample), dim=-1) as ib:
            gi = idx_bin[ib]
            logit = jnp.clip(linpred(gi) + u[clip[gi]] - b[item[gi]], -20, 20)
            numpyro.sample("y_bin", dist.Bernoulli(logits=logit),
                           obs=None if predict else y[gi])
    if Nt > 0:
        with numpyro.plate("obs_tern", Nt, subsample_size=_sub(Nt, subsample), dim=-1) as it:
            gj = idx_tern[it]
            pred = jnp.clip(linpred(gj) + u[clip[gj]], -20, 20)
            c0 = jnp.clip(kappa0[item[gj]], -19.0, 19.0)
            gap = jnp.clip(kappa1[item[gj]] - kappa0[item[gj]], 1e-2, 15.0)
            c1 = c0 + gap
            # Log-space graded-response category log-probabilities. numpyro's OrderedLogistic
            # forms P(middle) = sigmoid(c1-eta) - sigmoid(c0-eta), which underflows to exactly
            # 0 in float32 when eta sits far above both cutpoints (very hard items answered
            # partially by a strong model, e.g. LiveBench) => log 0 => NaN ELBO. Using
            # sigma(b)-sigma(a) = sigma(b) sigma(-a) (1-exp(a-b)) keeps every category
            # log-probability finite. Categorical(logits) with normalised logits is exact.
            if link == "pcm":
                # partial-credit (adjacent-category) model with the same two step parameters
                zero = jnp.zeros_like(pred)
                logits = jnp.stack([zero, pred - c0, 2 * pred - c0 - c1], axis=-1)
            else:
                lp0 = jax.nn.log_sigmoid(c0 - pred)
                lp2 = jax.nn.log_sigmoid(pred - c1)
                lp1 = jax.nn.log_sigmoid(c1 - pred) + jax.nn.log_sigmoid(pred - c0) + jnp.log(-jnp.expm1(-gap))
                logits = jnp.stack([lp0, lp1, lp2], axis=-1)
            numpyro.sample("y_tern", dist.Categorical(logits=logits),
                           obs=None if predict else y[gj])


def _graded_item_params(n_item, n_family, item_family, hier_gap=False):
    a = _hier_discrimination(n_item, n_family, item_family)
    b = _hier_difficulty(n_item, n_family, item_family, name="b")          # binary threshold
    kappa0 = _hier_difficulty(n_item, n_family, item_family, name="k0")    # ternary base cut
    if hier_gap:
        # family-pooled log-gap: lets the DATA set each family's typical partial-credit band
        # width instead of a fixed HalfNormal(0.5) that shrinks ~20-response items toward 0
        # (the source of the partial-category under-prediction in the PPC).
        mu_g = numpyro.sample("mu_log_gap", dist.Normal(0.0, 1.0).expand([n_family]))
        sig_g = numpyro.sample("sigma_log_gap", dist.HalfNormal(0.5).expand([n_family]))
        z_g = numpyro.sample("z_log_gap", dist.Normal(0.0, 1.0).expand([n_item]))
        gap = numpyro.deterministic("k_gap", jnp.exp(jnp.clip(mu_g[item_family] + sig_g[item_family] * z_g, -4.0, 2.0)))
    else:
        # tighter gap prior keeps ordered cutpoints from drifting far apart (extreme category
        # probs -> instability without a clip effect); still spans plausible partial-credit widths
        gap = numpyro.sample("k_gap", dist.HalfNormal(0.5).expand([n_item]))
    kappa1 = numpyro.deterministic("k1", kappa0 + gap)
    return a, b, kappa0, kappa1


def _clip_effect(n_clip, clip_effect):
    """Per-clip random effect with a FIXED small scale (clip_effect = the sigma, or False/0).

    A free HalfNormal(0.5) scale lets u_c trade off with ability under mean-field VI (their
    posterior correlation is ignored) and scrambles the model ranking; a fixed small scale
    keeps u as a light within-clip-dependence regulariser without absorbing ability signal.
    False/0 removes it entirely (can destabilise the graded likelihood)."""
    if not clip_effect:
        return numpyro.deterministic("u", jnp.zeros(n_clip))
    zu = numpyro.sample("z_u", dist.Normal(0.0, 1.0).expand([n_clip]))
    if clip_effect == "free":                       # learn the scale (stable, but see leaderboard note)
        scale = numpyro.sample("sigma_u", dist.HalfNormal(0.5))
    else:
        scale = float(clip_effect)                  # fixed scale
    return numpyro.deterministic("u", scale * zu)


def m2(resp, item, clip, y, idx_bin, idx_tern, n_resp, n_item, n_clip, n_family, n_group,
       item_family, resp_group, use_group=False, subsample=None, clip_effect=False,
       predict=False, hier_gap=False, theta_scale=1.0, link="grm"):
    theta = _ability_1d(n_resp, n_group, resp_group, use_group, theta_scale)
    a, b, kappa0, kappa1 = _graded_item_params(n_item, n_family, item_family, hier_gap)
    u = _clip_effect(n_clip, clip_effect)
    _grm_likelihood(resp, item, clip, y, idx_bin, idx_tern, a, theta, u, b, kappa0, kappa1,
                    item_dim=None, subsample=subsample, predict=predict, link=link)


# --------------------------------------------------------------------------- M3 (multidimensional)
def m3(resp, item, clip, y, idx_bin, idx_tern, item_dim, n_resp, n_item, n_clip, n_family,
       n_group, n_dim, item_family, resp_group, use_group=False, subsample=None, clip_effect=False,
       predict=False, hier_gap=False, theta_scale=1.0, link="grm"):
    theta = _ability_md(n_resp, n_group, resp_group, n_dim, use_group, theta_scale)     # (M, D)
    a, b, kappa0, kappa1 = _graded_item_params(n_item, n_family, item_family, hier_gap)
    u = _clip_effect(n_clip, clip_effect)
    _grm_likelihood(resp, item, clip, y, idx_bin, idx_tern, a, theta, u, b, kappa0, kappa1,
                    item_dim=item_dim, subsample=subsample, predict=predict, link=link)


# --------------------------------------------------------------------------- M3x (cross-loadings)
def m3x(resp, item, clip, y, idx_bin, idx_tern, item_dim, item_xdim, n_resp, n_item, n_clip,
        n_family, n_group, n_dim, item_family, resp_group, use_group=False, subsample=None,
        clip_effect=False, predict=False, hier_gap=False):
    """M3 with a small non-negative secondary loading: each item may also load on a
    designated secondary dimension x(j) (e.g. safety/PPE/count items on recognition).
    item_xdim < 0 => no cross-loading (mask). Sensitivity check for the zero-cross-loading
    confirmatory structure of M3; hierarchical prior centred on a SMALL loading so the data
    must argue for a larger one."""
    theta = _ability_md(n_resp, n_group, resp_group, n_dim, use_group)
    a, b, kappa0, kappa1 = _graded_item_params(n_item, n_family, item_family, hier_gap)
    mu_c = numpyro.sample("mu_log_c", dist.Normal(-1.5, 0.5).expand([n_family]))
    sig_c = numpyro.sample("sigma_log_c", dist.HalfNormal(0.5).expand([n_family]))
    z_c = numpyro.sample("z_log_c", dist.Normal(0.0, 1.0).expand([n_item]))
    log_c = jnp.clip(mu_c[item_family] + sig_c[item_family] * z_c, -4.0, 1.5)
    mask = (item_xdim >= 0).astype(jnp.float32)
    c = numpyro.deterministic("c", jnp.exp(log_c) * mask)
    xd = jnp.where(item_xdim >= 0, item_xdim, 0)
    u = _clip_effect(n_clip, clip_effect)
    _grm_likelihood(resp, item, clip, y, idx_bin, idx_tern, a, theta, u, b, kappa0, kappa1,
                    item_dim=item_dim, subsample=subsample, predict=predict, c=c, item_xdim=xd)
