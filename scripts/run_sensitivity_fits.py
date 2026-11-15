"""Two structural sensitivity fits for the confirmatory MIRT model M3.

  (a) NESTED pseudo-respondents: theta_resp = theta_base[g(resp)] + sigma * z_resp
      (use_group=True). The paper's default treats the 11 ablation runs as independent
      respondents; a reviewer notes this may inflate discriminations where prompt variants
      disagree. We compare item parameters, WAIC on the identical observation set, Omega,
      the two-stage leaderboard, and downstream compression at n=200.

  (b) CROSS-LOADINGS (M3x): safety / PPE / count items may also load (non-negatively, with a
      prior centred on a small value) on the recognition dimension. Reports WAIC vs M3, the
      fitted secondary-loading magnitudes per family (c/a ratio), Omega, and whether the
      per-dimension abilities still track the capability sub-scores (two-stage estimate by
      coordinate-ascent grid MAP because the dimensions no longer decouple).

Usage:
    python scripts/run_sensitivity_fits.py --steps 8000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, diagnostics, prep, params as P, harness as H, selection as S, reconstruct as R

DIMS = ["recognition", "perception_detail", "safety_reasoning"]
_T0 = time.time()
def log(m): print(f"[SENS +{time.time()-_T0:6.1f}s] {m}", flush=True)


def refined_theta_cross(p, c, xdim, Y, base_idx, grid=None, sweeps=4):
    """Two-stage per-dimension ability with cross-loadings: coordinate-ascent grid MAP."""
    grid = np.linspace(-6, 6, 121) if grid is None else grid
    keff = p.keff; top = keff - 1
    out = {}
    for i in base_idx:
        yi = Y[i]; idx = np.where(yi >= 0)[0]; y = yi[idx]
        a, b, k0, k1 = p.a[idx], p.b[idx], p.k0[idx], p.k1[idx]
        cc, xd, dd, ke, tp = c[idx], xdim[idx], p.item_dim[idx], keff[idx], top[idx]
        theta = np.zeros(p.n_dim)
        def ll_of(theta):
            eta = a * theta[dd] + cc * theta[xd]
            pc = 1 / (1 + np.exp(-np.clip(eta - b, -30, 30)))
            s0 = 1 / (1 + np.exp(-np.clip(eta - k0, -30, 30))); s1 = 1 / (1 + np.exp(-np.clip(eta - k1, -30, 30)))
            tern = ke == 3
            P0 = np.where(tern, 1 - s0, 1 - pc); P1 = np.where(tern, s0 - s1, 0.0); P2 = np.where(tern, s1, pc)
            py = np.where(y == 0, P0, np.where(y == tp, P2, P1))
            return np.log(np.clip(py, 1e-9, 1)).sum() - 0.5 * (theta ** 2).sum()
        for _ in range(sweeps):
            for d in range(p.n_dim):
                best, bt = -np.inf, theta[d]
                for g in grid:
                    th = theta.copy(); th[d] = g
                    v = ll_of(th)
                    if v > best:
                        best, bt = v, g
                theta[d] = bt
        out[i] = theta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fits", default="results/fits")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/sensitivity_fits.json")
    ap.add_argument("--hier-gap", action="store_true")
    ap.add_argument("--m3-init", default=None)
    args = ap.parse_args()
    HG = {"hier_gap": bool(args.hier_gap)}

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    A = prep.field_arrays(fd)
    item_dim = prep.item_dim_array(fd, 3)
    item_xdim = prep.item_xdim_array(fd)
    dim_of_item = np.array(item_dim)
    keff = np.asarray(fd.item_keff); item_clip = np.asarray(fd.item_clip)
    base_idx = H.base_indices(fd); names = [fd.resp_names[i] for i in base_idx]
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    fitkw = dict(n_steps=args.steps, n_seeds=args.seeds, n_post=200)
    if args.m3_init:
        fitkw["init_values"] = args.m3_init
    log(f"N={len(fd.y):,} items={fd.n_item} resp={fd.n_resp}")

    # true sub-scores / full scores
    p_true = P.ItemParams(a=np.zeros(fd.n_item), b=np.zeros(fd.n_item), k0=np.zeros(fd.n_item),
                          k1=np.zeros(fd.n_item), u=np.zeros(fd.n_clip), keff=keff,
                          family=np.asarray(fd.item_family), clip=item_clip, family_names=fd.family_names)
    true_full = H.true_full_scores(fd, p_true, Y)
    def sub_true(i, d):
        sel = np.where(dim_of_item == d)[0]; yv = Y[i][sel]; pr = yv >= 0
        return float((yv[pr] / (keff[sel][pr] - 1)).mean())
    tsub = {d: np.array([sub_true(i, d) for i in base_idx]) for d in range(3)}

    # reference M3 (paper fit)
    p_ref = P.ItemParams.from_npz(HERE / args.fits / "m3.npz")
    zref = np.load(HERE / args.fits / "m3.npz", allow_pickle=True)
    theta_ref = zref["theta"]                      # refined (M, 3)
    corr_ref = np.asarray(zref["corr"])
    # reference WAIC: refit M3 default to get pointwise ll on the same obs (paper's ea1 value is -91,342)
    log("fitting reference M3 (independent respondents) for pointwise WAIC...")
    r_ref = fit.fit_svi(models.m3, prep.m3_args(A, item_dim, 3), {"use_group": False, **HG}, **fitkw)
    ll_ref = fit.pointwise_loglik(models.m3, r_ref["posterior"], prep.m3_args(A, item_dim, 3), {"use_group": False, **HG})
    waic_ref = diagnostics.elpd_summary(ll_ref)
    log(f"reference M3 elpd={waic_ref['elpd_waic']:.0f}")

    report = {"steps": args.steps, "seeds": args.seeds, "n_subset": args.n,
              "reference_M3": {"elpd_waic": waic_ref["elpd_waic"], "omega": corr_ref.round(3).tolist(),
                               "theta_vs_subscore_tau": {DIMS[d]: float(kendalltau(theta_ref[base_idx, d], tsub[d]).statistic) for d in range(3)}}}

    def downstream(p, theta_base_md):
        clip_info, _ = S.clip_information(p, theta_base_md)
        Sc = S.greedy_stratified(clip_info, strata, args.n, min_k=3)
        m = H.eval_subset(fd, p, Sc, true_full, Y=Y, estimator="gp_cal")
        # sub-score tau (naive) on this subset
        Sset = set(int(c) for c in Sc); in_S = np.array([c in Sset for c in item_clip])
        taus = {}
        for d in range(3):
            it = np.where(in_S & (dim_of_item == d))[0]
            est = []
            for i in base_idx:
                yv = Y[i][it]; pr = yv >= 0; est.append(float((yv[pr] / (keff[it][pr] - 1)).mean()))
            taus[DIMS[d]] = float(kendalltau(est, tsub[d]).statistic)
        overlap = len(Sset & set(int(c) for c in S.greedy_stratified(S.clip_information(p_ref, theta_ref[base_idx])[0], strata, args.n, min_k=3)))
        return {"gp_cal_mae": m["mae"], "gp_cal_tau": m["kendall_tau"], "subscore_tau_naive": taus,
                "clip_overlap_with_reference_selection": overlap}

    # ------------------------------------------------------------ (a) nested pseudo
    log("fitting M3 with NESTED pseudo-respondents (use_group=True)...")
    r_g = fit.fit_svi(models.m3, prep.m3_args(A, item_dim, 3), {"use_group": True, **HG}, **fitkw)
    ll_g = fit.pointwise_loglik(models.m3, r_g["posterior"], prep.m3_args(A, item_dim, 3), {"use_group": True, **HG})
    waic_g = diagnostics.elpd_summary(ll_g)
    p_g = P.ItemParams.from_fit(r_g["derived"], fd, item_dim=np.array(item_dim)); p_g.u = np.zeros_like(p_g.u)
    theta_g = H.refine_abilities(p_g, fd, Y)
    L = np.asarray(r_g["posterior"]["L_corr"]).mean(0); omega_g = L @ L.T
    dd = np.sqrt(np.diag(omega_g)); omega_g = omega_g / np.outer(dd, dd)
    sig_resp = float(np.asarray(r_g["posterior"]["sigma_resp"]).mean())
    cmp = diagnostics.compare_models({"M3_independent": ll_ref, "M3_nested": ll_g})
    fam = np.asarray(fd.item_family)
    report["a_nested_pseudo"] = {
        "elpd_waic": waic_g["elpd_waic"], "elpd_diff_vs_independent": cmp["models"]["M3_nested"]["elpd_diff"] * (-1 if cmp["best"] == "M3_nested" else 1),
        "compare": cmp, "sigma_resp_posterior_mean": sig_resp,
        "item_param_corr_vs_independent": {k: float(pearsonr(getattr(p_g, k), getattr(p_ref, k))[0]) for k in ("a", "b", "k0", "k1")},
        "discrimination_ratio_nested_over_independent_by_family": {
            fd.family_names[f]: float(np.median(p_g.a[fam == f] / p_ref.a[fam == f])) for f in range(fd.n_family)},
        "omega": omega_g.round(3).tolist(),
        "theta_vs_subscore_tau": {DIMS[d]: float(kendalltau(theta_g[base_idx, d], tsub[d]).statistic) for d in range(3)},
        "leaderboard_tau_vs_independent": {DIMS[d]: float(kendalltau(theta_g[base_idx, d], theta_ref[base_idx, d]).statistic) for d in range(3)},
        "downstream_n": downstream(p_g, theta_g[base_idx]),
        "fit_seconds": r_g["seconds"],
    }
    log(f"(a) nested: elpd={waic_g['elpd_waic']:.0f} sigma_resp={sig_resp:.3f} a-corr={report['a_nested_pseudo']['item_param_corr_vs_independent']['a']:.3f} "
        f"theta-tau={report['a_nested_pseudo']['theta_vs_subscore_tau']} down={report['a_nested_pseudo']['downstream_n']}")

    # ------------------------------------------------------------ (b) cross-loadings
    log("fitting M3x (secondary loadings on recognition for safety/PPE/count)...")
    margs = prep.m3x_args(A, item_dim, item_xdim, 3)
    r_x = fit.fit_svi(models.m3x, margs, {"use_group": False, **HG}, **fitkw)
    ll_x = fit.pointwise_loglik(models.m3x, r_x["posterior"], margs, {"use_group": False, **HG})
    waic_x = diagnostics.elpd_summary(ll_x)
    p_x = P.ItemParams.from_fit(r_x["derived"], fd, item_dim=np.array(item_dim)); p_x.u = np.zeros_like(p_x.u)
    c_x = np.asarray(r_x["derived"]["c"]).mean(0)
    xdim = np.where(np.array(item_xdim) >= 0, np.array(item_xdim), 0)
    L = np.asarray(r_x["posterior"]["L_corr"]).mean(0); omega_x = L @ L.T
    dd = np.sqrt(np.diag(omega_x)); omega_x = omega_x / np.outer(dd, dd)
    th_x = refined_theta_cross(p_x, c_x, xdim, Y, base_idx)
    theta_x = np.array([th_x[i] for i in base_idx])
    cmpx = diagnostics.compare_models({"M3": ll_ref, "M3x": ll_x})
    report["b_cross_loadings"] = {
        "elpd_waic": waic_x["elpd_waic"], "compare": cmpx,
        "secondary_loading_over_primary_median_by_family": {
            fd.family_names[f]: float(np.median(c_x[fam == f] / p_x.a[fam == f])) for f in range(fd.n_family)
            if np.any(c_x[fam == f] > 0)},
        "secondary_loading_median_by_family": {
            fd.family_names[f]: float(np.median(c_x[fam == f])) for f in range(fd.n_family) if np.any(c_x[fam == f] > 0)},
        "primary_loading_median_by_family": {fd.family_names[f]: float(np.median(p_x.a[fam == f])) for f in range(fd.n_family)},
        "omega": omega_x.round(3).tolist(),
        "theta_vs_subscore_tau": {DIMS[d]: float(kendalltau(theta_x[:, d], tsub[d]).statistic) for d in range(3)},
        "leaderboard_tau_vs_M3": {DIMS[d]: float(kendalltau(theta_x[:, d], theta_ref[base_idx, d]).statistic) for d in range(3)},
        "fit_seconds": r_x["seconds"],
    }
    log(f"(b) M3x: elpd={waic_x['elpd_waic']:.0f} vs M3 {waic_ref['elpd_waic']:.0f}; c/a={report['b_cross_loadings']['secondary_loading_over_primary_median_by_family']} "
        f"omega={omega_x.round(2).tolist()} theta-tau={report['b_cross_loadings']['theta_vs_subscore_tau']}")

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
