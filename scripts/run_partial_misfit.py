"""Where does the partial-category PPC misfit come from, and does fixing it change anything?

The paper's PPC under-predicts the partial category (0.115 vs 0.209 observed). Two candidate
causes are tested:

  1. PRIOR SHRINKAGE of the cutpoint gap: the fixed HalfNormal(0.5) gap prior pulls the
     ~20-response items' partial band toward zero width. Remedy: a family-pooled hierarchical
     log-gap prior (`hier_gap=True`). Tested at full scale with SVI for M2 and M3: PPC
     marginals (overall and per ternary family), WAIC against the paper's models on the same
     observations, agreement of the two-stage abilities, and downstream selection at n=200.

  2. MISSING WITHIN-CLIP (testlet) DEPENDENCE: a clip random effect u_c. Tested with NUTS on a
     clip subsample (where NUTS is affordable) for M2 without u, with a free-scale u, and with
     u + hierarchical gap; also SVI on the same subsample. Reports PPC partial marginal, the
     posterior of sigma_u, base-model ability agreement across variants, and the rank
     correlation of clip information with vs without u.

Usage:
    python scripts/run_partial_misfit.py --steps 8000 --sub-clips 100
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, diagnostics, prep, params as P, harness as H, selection as S

_T0 = time.time()
def log(m): print(f"[PART +{time.time()-_T0:7.1f}s] {m}", flush=True)
DIMS = ["recognition", "perception_detail", "safety_reasoning"]


def ppc(model, r, margs, mkw, fd, per_family=True):
    """Ternary/binary PPC marginals; per-family ternary split via observation family index."""
    ybin = fd.y[fd.binary_mask]; ytern = fd.y[~fd.binary_mask]
    out = diagnostics.ppc_category_marginals(r["posterior"], model, margs, mkw,
                                             {"y_bin": ybin, "y_tern": ytern})
    if per_family:
        from numpyro.infer import Predictive
        from jax import random
        kw = {**mkw, "subsample": None, "predict": True}
        latent = {k: v for k, v in r["posterior"].items() if not k.startswith("_")}
        sim = Predictive(model, posterior_samples=latent)(random.PRNGKey(123), *margs, **kw)
        rep = np.asarray(sim["y_tern"])                       # (S, Nt)
        fam_obs = fd.item_family[fd.item][~fd.binary_mask]
        out["y_tern_by_family"] = {}
        for f in np.unique(fam_obs):
            m = fam_obs == f
            out["y_tern_by_family"][fd.family_names[f]] = {
                "observed": (np.bincount(ytern[m], minlength=3) / m.sum()).round(4).tolist(),
                "predicted": (np.bincount(rep[:, m].reshape(-1), minlength=3) / rep[:, m].size).round(4).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sub-clips", type=int, default=100)
    ap.add_argument("--nuts-warmup", type=int, default=400)
    ap.add_argument("--nuts-samples", type=int, default=400)
    ap.add_argument("--skip-nuts", action="store_true")
    ap.add_argument("--out", default="results/partial_misfit.json")
    args = ap.parse_args()
    report = {"steps": args.steps, "seeds": args.seeds}

    # ================================================================ Part 1: full-scale SVI
    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    A = prep.field_arrays(fd)
    item_dim = prep.item_dim_array(fd, 3)
    dim_of_item = np.array(item_dim); keff = np.asarray(fd.item_keff); item_clip = np.asarray(fd.item_clip)
    base_idx = H.base_indices(fd); names = [fd.resp_names[i] for i in base_idx]
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    fitkw = dict(n_steps=args.steps, n_seeds=args.seeds, n_post=200)
    p_true = P.ItemParams(a=np.zeros(fd.n_item), b=np.zeros(fd.n_item), k0=np.zeros(fd.n_item),
                          k1=np.zeros(fd.n_item), u=np.zeros(fd.n_clip), keff=keff,
                          family=np.asarray(fd.item_family), clip=item_clip, family_names=fd.family_names)
    true_full = H.true_full_scores(fd, p_true, Y)
    def sub_true(i, d):
        sel = np.where(dim_of_item == d)[0]; yv = Y[i][sel]; pr = yv >= 0
        return float((yv[pr] / (keff[sel][pr] - 1)).mean())
    tsub = {d: np.array([sub_true(i, d) for i in base_idx]) for d in range(3)}
    zref = np.load(HERE / "results/fits/m3.npz", allow_pickle=True); theta_ref = zref["theta"]

    part1 = {}
    fits = {}
    for name, model, margs, mkw in (
            ("M2", models.m2, prep.m2_args(A), {"use_group": False}),
            ("M2_hiergap", models.m2, prep.m2_args(A), {"use_group": False, "hier_gap": True}),
            ("M3", models.m3, prep.m3_args(A, item_dim, 3), {"use_group": False}),
            ("M3_hiergap", models.m3, prep.m3_args(A, item_dim, 3), {"use_group": False, "hier_gap": True})):
        log(f"SVI {name} ...")
        r = fit.fit_svi(model, margs, mkw, **fitkw)
        ll = fit.pointwise_loglik(model, r["posterior"], margs, mkw)
        w = diagnostics.elpd_summary(ll)
        pp = ppc(model, r, margs, mkw, fd)
        idim = np.array(item_dim) if name.startswith("M3") else None
        p = P.ItemParams.from_fit(r["derived"], fd, item_dim=idim); p.u = np.zeros_like(p.u)
        theta = H.refine_abilities(p, fd, Y)
        entry = {"elpd_waic": w["elpd_waic"], "p_waic": w["p_waic"], "final_elbo": r["loss"], "seconds": r["seconds"],
                 "ppc": pp, "gap_median_by_family": {}}
        fam = np.asarray(fd.item_family)
        for f in range(fd.n_family):
            m = (fam == f) & (keff == 3)
            if m.any():
                entry["gap_median_by_family"][fd.family_names[f]] = float(np.median(p.k1[m] - p.k0[m]))
        if idim is not None:
            entry["theta_vs_subscore_tau"] = {DIMS[d]: float(kendalltau(theta[base_idx, d], tsub[d]).statistic) for d in range(3)}
            entry["theta_tau_vs_paper_M3"] = {DIMS[d]: float(kendalltau(theta[base_idx, d], theta_ref[base_idx, d]).statistic) for d in range(3)}
            L = np.asarray(r["posterior"]["L_corr"]).mean(0); om = L @ L.T; dd = np.sqrt(np.diag(om))
            entry["omega"] = (om / np.outer(dd, dd)).round(3).tolist()
            clip_info, _ = S.clip_information(p, theta[base_idx])
            Sc = S.greedy_stratified(clip_info, strata, args.n, min_k=3)
            m = H.eval_subset(fd, p, Sc, true_full, Y=Y, estimator="gp_cal")
            entry["downstream_n"] = {"n": args.n, "gp_cal_mae": m["mae"], "gp_cal_tau": m["kendall_tau"]}
        else:
            entry["theta_vs_fullscore_tau"] = float(kendalltau(theta[base_idx], [true_full[n] for n in names]).statistic)
        fits[name] = (ll, r, p, theta)
        part1[name] = entry
        log(f"  {name}: elpd={w['elpd_waic']:.0f} PPC tern obs={pp['y_tern']['observed']} pred={pp['y_tern']['predicted']} "
            f"gaps={entry['gap_median_by_family']}")
    part1["compare_M2"] = diagnostics.compare_models({"M2": fits["M2"][0], "M2_hiergap": fits["M2_hiergap"][0]})
    part1["compare_M3"] = diagnostics.compare_models({"M3": fits["M3"][0], "M3_hiergap": fits["M3_hiergap"][0]})
    part1["compare_M2h_M3h"] = diagnostics.compare_models({"M2_hiergap": fits["M2_hiergap"][0], "M3_hiergap": fits["M3_hiergap"][0]})
    part1["item_param_corr_M3_vs_M3h"] = {k: float(pearsonr(getattr(fits["M3"][2], k), getattr(fits["M3_hiergap"][2], k))[0]) for k in ("a", "b", "k0", "k1")}
    report["part1_full_svi"] = part1
    (HERE / args.out).write_text(json.dumps(report, indent=2))
    del fits

    if args.skip_nuts:
        log(f"wrote {args.out} (NUTS skipped)"); return

    # ================================================================ Part 2: NUTS on a clip subsample
    df = pd.read_parquet(HERE / args.matrix)
    rng = np.random.default_rng(0)
    sub = S.select_stratified_random(strata, args.sub_clips, seed=0, min_k=1)
    sub_ids = set(fd.clip_names[c] for c in sub)
    sp = HERE / "results" / "tmp"
    sp.mkdir(parents=True, exist_ok=True)
    subp = sp / "sub_matrix.parquet"
    df[df["clip_id"].isin(sub_ids)].to_parquet(subp)
    fs = io.load_field_data(subp)
    As = prep.field_arrays(fs)
    log(f"NUTS subsample: clips={fs.n_clip} items={fs.n_item} obs={len(fs.y):,} resp={fs.n_resp}")
    m2a = prep.m2_args(As)
    part2 = {"sub_clips": fs.n_clip, "sub_items": fs.n_item, "sub_obs": int(len(fs.y)),
             "nuts_warmup": args.nuts_warmup, "nuts_samples": args.nuts_samples, "variants": {}}
    thetas = {}
    for name, mkw in (("M2_no_u", {"use_group": False}),
                      ("M2_free_u", {"use_group": False, "clip_effect": "free"}),
                      ("M2_free_u_hiergap", {"use_group": False, "clip_effect": "free", "hier_gap": True})):
        for est in ("nuts", "svi"):
            log(f"{est.upper()} {name} ...")
            if est == "nuts":
                rn = fit.fit_nuts(models.m2, m2a, mkw, n_warmup=args.nuts_warmup, n_samples=args.nuts_samples)
                post = {k: np.asarray(v) for k, v in rn["samples"].items()}
                secs = rn["seconds"]
                # derived sites via Predictive on the samples
                from numpyro.infer import Predictive
                from jax import random
                det = Predictive(models.m2, posterior_samples=rn["samples"])(random.PRNGKey(1), *m2a, **mkw)
                derived = {**{k: np.asarray(v) for k, v in det.items()}, **post}   # sampled + deterministic sites
                rr = {"posterior": rn["samples"], "derived": derived}
                divergences = int(np.asarray(rn["mcmc"].get_extra_fields()["diverging"]).sum())
            else:
                rr = fit.fit_svi(models.m2, m2a, mkw, n_steps=args.steps, n_seeds=2, n_post=200)
                secs = rr["seconds"]; divergences = None
            pp = ppc(models.m2, rr, m2a, mkw, fs, per_family=True)
            theta = np.asarray(rr["derived"]["theta"]).mean(0)
            bidx = H.base_indices(fs)
            thetas[(name, est)] = theta[bidx]
            ent = {"seconds": secs, "ppc_y_tern": pp["y_tern"], "ppc_by_family": pp["y_tern_by_family"],
                   "divergences": divergences}
            if "sigma_u" in rr["posterior"]:
                su = np.asarray(rr["posterior"]["sigma_u"])
                ent["sigma_u_mean"] = float(su.mean()); ent["sigma_u_q05_q95"] = [float(np.quantile(su, .05)), float(np.quantile(su, .95))]
            # clip information with u vs without: rank correlation of the two clip-info vectors
            p = P.ItemParams.from_fit(rr["derived"], fs, item_dim=None)
            ci_u, _ = S.clip_information(p, theta[bidx])
            p0 = P.ItemParams.from_fit(rr["derived"], fs, item_dim=None); p0.u = np.zeros_like(p0.u)
            ci_0, _ = S.clip_information(p0, theta[bidx])
            ent["clip_info_spearman_u_vs_no_u"] = float(spearmanr(ci_u, ci_0).statistic)
            part2["variants"][f"{name}__{est}"] = ent
            log(f"  {name}/{est}: PPC tern pred={pp['y_tern']['predicted']} obs={pp['y_tern']['observed']} "
                f"sigma_u={ent.get('sigma_u_mean')} div={divergences} ({secs}s)")
    # ability agreement matrix
    keys = list(thetas)
    part2["theta_agreement_pearson"] = {f"{a[0]}/{a[1]}__vs__{b[0]}/{b[1]}": float(pearsonr(thetas[a], thetas[b])[0])
                                        for i, a in enumerate(keys) for b in keys[i + 1:]}
    report["part2_nuts_subsample"] = part2
    (HERE / args.out).write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
