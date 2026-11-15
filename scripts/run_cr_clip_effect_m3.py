"""Camera-ready (reviewer D4yq #5): justify omitting the clip-level random effect for M3,
the model that drives selection of the released subset.

`run_partial_misfit.py` answered this for M2. Here the same test is run for M3 on a
stratified 100-clip sub-matrix, where NUTS (exact MCMC, no mean-field factorization) is
affordable. Variants: M3 without u, M3 with a free-scale clip effect u_c. Each is fitted
with NUTS and with the paper's mean-field SVI. Reported:
  - sigma_u posterior (does within-clip dependence exist? can SVI recover it?)
  - WAIC with vs without u under NUTS (does u improve predictive fit?)
  - selection invariance: Spearman of clip information from the with-u fit vs the no-u
    fit, and overlap of the top-k clips each would select
  - per-dimension ability rankings with vs without u (Kendall tau over the 9 base models)

Usage:
    python scripts/run_cr_clip_effect_m3.py --sub-clips 100
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, prep, params as P, harness as H, selection as S

_T0 = time.time()
def log(m): print(f"[CLIP +{time.time()-_T0:7.1f}s] {m}", flush=True)
DIMS = ["recognition", "perception_detail", "safety_reasoning"]


def waic(ll):
    """ll: (draws, obs) pointwise log-likelihood -> elpd_waic, se."""
    ll = np.asarray(ll)
    lppd = np.log(np.mean(np.exp(ll - ll.max(0)), 0)) + ll.max(0)
    pw = ll.var(0)
    e = lppd - pw
    return float(e.sum()), float(np.sqrt(len(e) * e.var())), e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--sub-clips", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--nuts-warmup", type=int, default=400)
    ap.add_argument("--nuts-samples", type=int, default=400)
    ap.add_argument("--chains", type=int, default=1, help="independent NUTS chains (seeds 0..chains-1)")
    ap.add_argument("--top-k", default="20,30,50")
    ap.add_argument("--out", default="results/camera_ready/clip_effect_m3.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    sub = S.select_stratified_random(strata, args.sub_clips, seed=0, min_k=1)
    sub_ids = set(fd.clip_names[c] for c in sub)
    df = pd.read_parquet(HERE / args.matrix)
    work = HERE / "results/camera_ready/tmp"; work.mkdir(parents=True, exist_ok=True)
    subp = work / "sub_matrix_clip_effect.parquet"
    df[df["clip_id"].isin(sub_ids)].to_parquet(subp)
    fs = io.load_field_data(subp)
    A = prep.field_arrays(fs)
    item_dim = prep.item_dim_array(fs, 3)
    m3a = prep.m3_args(A, item_dim, 3)
    bidx = H.base_indices(fs)
    log(f"sub-matrix: clips={fs.n_clip} items={fs.n_item} obs={len(fs.y):,} resp={fs.n_resp}")
    Ys = H.response_table(fs); keff_s = np.asarray(fs.item_keff); dim_s = np.asarray(item_dim)
    obs_sub = np.array([[float((Ys[i][(dim_s == d) & (Ys[i] >= 0)] /
                                (keff_s[(dim_s == d) & (Ys[i] >= 0)] - 1)).mean()) for d in range(3)]
                        for i in bidx])                                   # (9, 3) observed sub-scores

    from numpyro.infer import Predictive
    from jax import random
    report = {"sub_clips": fs.n_clip, "sub_items": fs.n_item, "sub_obs": int(len(fs.y)),
              "nuts_warmup": args.nuts_warmup, "nuts_samples": args.nuts_samples,
              "nuts_chains": args.chains, "svi_steps": args.steps, "variants": {}}
    fits = {}
    for name, mkw in (("M3_no_u", {"use_group": False}),
                      ("M3_free_u", {"use_group": False, "clip_effect": "free"})):
        for est in ("nuts", "svi"):
            log(f"{est.upper()} {name} ...")
            if est == "nuts":
                runs = [fit.fit_nuts(models.m3, m3a, mkw, n_warmup=args.nuts_warmup,
                                     n_samples=args.nuts_samples, seed=ch) for ch in range(args.chains)]
                by_chain = {k: np.stack([np.asarray(r["samples"][k]) for r in runs]) for k in runs[0]["samples"]}
                flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in by_chain.items()}   # chains*draws
                det = Predictive(models.m3, posterior_samples=flat)(random.PRNGKey(1), *m3a, **mkw)
                derived = {**{k: np.asarray(v) for k, v in det.items()}, **flat}
                rr = {"posterior": flat, "derived": derived}
                secs = round(sum(r["seconds"] for r in runs), 1)
                div = int(sum(np.asarray(r["mcmc"].get_extra_fields()["diverging"]).sum() for r in runs))
            else:
                rr = fit.fit_svi(models.m3, m3a, mkw, n_steps=args.steps, n_seeds=2, n_post=200)
                secs, div = rr["seconds"], None
            theta = np.asarray(rr["derived"]["theta"]).mean(0)[bidx]       # (9, 3)
            p = P.ItemParams.from_fit(rr["derived"], fs, item_dim=np.asarray(item_dim))
            p_nou = P.ItemParams.from_fit(rr["derived"], fs, item_dim=np.asarray(item_dim))
            p_nou.u = np.zeros_like(p_nou.u)
            ci, _ = S.clip_information(p_nou, theta)                      # selection uses u=0
            ent = {"seconds": secs, "divergences": div}
            ent["theta_vs_observed_subscore_tau"] = {DIMS[d]: float(kendalltau(theta[:, d], obs_sub[:, d]).statistic)
                                                     for d in range(3)}
            if est == "nuts":
                from numpyro.diagnostics import split_gelman_rubin, effective_sample_size
                diag = {}
                nc = args.chains
                sites = {"theta": derived["theta"], **flat}
                for site, v in sites.items():
                    v = np.asarray(v)
                    if v.ndim >= 1:
                        x = v.reshape(nc, v.shape[0] // nc, -1)                        # (chains, draws, flat)
                        rh = np.asarray(split_gelman_rubin(x)); es = np.asarray(effective_sample_size(x))
                        diag[site] = {"rhat_max": float(np.nanmax(rh)), "rhat_frac_gt_1.05": float(np.mean(rh > 1.05)),
                                      "ess_min": float(np.nanmin(es)), "ess_median": float(np.nanmedian(es))}
                ent["nuts_diagnostics"] = diag
            if "sigma_u" in rr["posterior"]:
                su = np.asarray(rr["posterior"]["sigma_u"])
                ent["sigma_u_mean"] = float(su.mean())
                ent["sigma_u_q05_q95"] = [float(np.quantile(su, .05)), float(np.quantile(su, .95))]
            if est == "nuts":
                ll = fit.pointwise_loglik(models.m3, rr["posterior"], m3a, mkw)
                ll = np.concatenate([np.asarray(v).reshape(np.asarray(v).shape[0], -1) for v in ll.values()], 1) \
                    if isinstance(ll, dict) else np.asarray(ll)
                ent["elpd_waic"], ent["waic_se"], pw = waic(ll)
                fits[(name, est, "pw")] = pw
            fits[(name, est)] = {"theta": theta, "clip_info": ci}
            report["variants"][f"{name}__{est}"] = ent
            log(f"  {name}/{est}: sigma_u={ent.get('sigma_u_mean')} div={div} waic={ent.get('elpd_waic')} ({secs}s)")

    # with-u vs without-u comparisons, per estimator
    ks = [int(k) for k in args.top_k.split(",")]
    cmp = {}
    for est in ("nuts", "svi"):
        a, b = fits[("M3_no_u", est)], fits[("M3_free_u", est)]
        c = {"clip_info_spearman": float(spearmanr(a["clip_info"], b["clip_info"]).statistic),
             "top_k_overlap": {str(k): int(len(set(np.argsort(-a["clip_info"])[:k]) &
                                               set(np.argsort(-b["clip_info"])[:k]))) for k in ks},
             "theta_rank_tau_by_dim": {DIMS[d]: float(kendalltau(a["theta"][:, d], b["theta"][:, d]).statistic)
                                       for d in range(3)}}
        cmp[est] = c
    # NUTS vs SVI agreement for the paper's no-u M3
    a, b = fits[("M3_no_u", "nuts")], fits[("M3_no_u", "svi")]
    cmp["no_u_nuts_vs_svi"] = {
        "clip_info_spearman": float(spearmanr(a["clip_info"], b["clip_info"]).statistic),
        "theta_rank_tau_by_dim": {DIMS[d]: float(kendalltau(a["theta"][:, d], b["theta"][:, d]).statistic)
                                  for d in range(3)}}
    d = fits[("M3_free_u", "nuts", "pw")] - fits[("M3_no_u", "nuts", "pw")]
    cmp["nuts_waic_u_minus_no_u"] = {"elpd_diff": float(d.sum()), "dse": float(np.sqrt(len(d) * d.var()))}
    report["with_vs_without_u"] = cmp
    (HERE / args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(cmp, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
