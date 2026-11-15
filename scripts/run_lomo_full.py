"""Unified leakage-safe leave-one-model-out (LOMO) validation with ALL selectors and ALL
estimators, so the held-out table can carry the baselines a reviewer asks for.

Per held-out base model h (9 rotations):
  1. refit item parameters on the other 8 base models + THEIR pseudo-respondents only
     (h's ablation runs are excluded with it). Fits are cached in --cache so later
     analyses (released-subset evaluation, stabilisation experiments) reuse them.
  2. select clip subsets with every selector at every size:
       fisher_stratified / fisher_unconstrained   (IRT information, rotation fit)
       stratified_random / random                 (no model; several seeds)
       kmedoids                                   (anchor clustering on rotation item params)
       disagree_stratified / disagree_unconstrained
           non-IRT discrimination proxy: std across the 8 TRAINING models of the clip-mean
           score (fit-free, leakage-safe). Answers "does the psychometric layer beat an
           obvious heuristic?"
  3. reconstruct h's full-benchmark score from its subset responses with
       subset_mean  naive
       irt          frozen-item MAP ability -> expected full score
       irt_cal      irt passed through an affine map fitted on the 8 TRAINING models'
                    (irt-from-same-subset, true) pairs — a proper out-of-sample calibration
       gp_cal       0.7*irt_cal + 0.3*subset_mean
     and its per-dimension capability sub-scores (subset_mean and irt_cal per dimension).
  4. aggregate over rotations: MAE with rotation-bootstrap CI, interior/best/worst-model
     error, Kendall tau of the 9 held-out predictions vs truth (with a rotation-bootstrap
     CI), affine slope/intercept dispersion across rotations.

Usage:
    python scripts/run_lomo_full.py --models m2,m3 --sizes 100,150,200,300
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, params as P, selection as S, harness as H, prep, reconstruct as REC

_T0 = time.time()
def log(m): print(f"[LOMO +{time.time()-_T0:7.1f}s] {m}", flush=True)

DIMS = ["recognition", "perception_detail", "safety_reasoning"]
ESTIMATORS = ["subset_mean", "irt", "irt_cal", "gp_cal"]


# ----------------------------------------------------------------------------- fitting / cache
def group_members(fd, base_name):
    return [i for i, nm in enumerate(fd.resp_names)
            if (nm == base_name) or nm.startswith(base_name + "__")]


def fit_rotation(fd, model_name, excl, item_dim, D, steps, seeds, cache_path, fit_extra=None, hier_gap=False):
    """Fit (or load) the rotation item parameters. Returns (ItemParams, theta_vi, used_model)."""
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        used = str(z["used_model"])
        idim = z["item_dim"] if "item_dim" in z.files and used == "m3" else None
        p = P.ItemParams(a=z["a"], b=z["b"], k0=z["k0"], k1=z["k1"], u=z["u"],
                         keff=np.asarray(fd.item_keff), family=np.asarray(fd.item_family),
                         clip=np.asarray(fd.item_clip), family_names=list(fd.family_names),
                         item_dim=(np.asarray(idim) if idim is not None else None),
                         n_dim=(int(idim.max()) + 1 if idim is not None else 1))
        return p, z["theta_vi"], used, float(z["seconds"])
    A = prep.rotation_field_arrays(fd, excl)
    used = model_name
    fit_extra = fit_extra or {}
    mk = {"use_group": False, "hier_gap": hier_gap}
    try:
        if model_name == "m3":
            r = fit.fit_svi(models.m3, prep.m3_args(A, item_dim, D), mk,
                            n_steps=steps, n_seeds=seeds, n_post=200, **fit_extra)
        else:
            r = fit.fit_svi(models.m2, prep.m2_args(A), mk,
                            n_steps=steps, n_seeds=seeds, n_post=200)
    except RuntimeError:
        log(f"  {model_name} diverged on this rotation -> falling back to m2")
        used = "m2"
        r = fit.fit_svi(models.m2, prep.m2_args(A), mk,
                        n_steps=steps, n_seeds=seeds, n_post=200)
    idim = item_dim if used == "m3" else None
    p = P.ItemParams.from_fit(r["derived"], fd, item_dim=idim)
    p.u = np.zeros_like(p.u)
    theta_vi = np.asarray(r["derived"]["theta"]).mean(0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, a=p.a, b=p.b, k0=p.k0, k1=p.k1, u=p.u, theta_vi=theta_vi,
             used_model=used, seconds=r["seconds"], elbo=r["loss"],
             item_dim=(np.asarray(idim) if idim is not None else np.zeros(0)),
             resp_names=np.array(fd.resp_names))
    return p, theta_vi, used, float(r["seconds"])


# ----------------------------------------------------------------------------- scoring helpers
def clip_mean_scores(Y, keff, item_clip, n_clip, rows):
    """(len(rows), n_clip) mean ordinal score per clip per respondent (nan if unseen)."""
    out = np.full((len(rows), n_clip), np.nan)
    for r, i in enumerate(rows):
        yi = Y[i]; pres = yi >= 0
        sc = yi[pres] / (keff[pres] - 1)
        s = np.bincount(item_clip[pres], weights=sc, minlength=n_clip)
        c = np.bincount(item_clip[pres], minlength=n_clip)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def sub_scores(y_row, item_idx, dim_of_item, keff):
    out = {}
    for d in range(3):
        sel = item_idx[dim_of_item[item_idx] == d]
        yv = y_row[sel]; pres = yv >= 0
        out[DIMS[d]] = float((yv[pres] / (keff[sel][pres] - 1)).mean()) if pres.any() else float("nan")
    return out


def affine_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) >= 2 and np.std(x) > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    return 0.0, float(np.mean(y))


def irt_expected_by_dim(p, theta, dim_of_item):
    """Model-based expected sub-score per dimension at ability theta (mean over ALL items of that dim)."""
    tv = P.theta_for_items(p, theta)
    es = P.expected_score(p, tv)
    return {DIMS[d]: float(es[dim_of_item == d].mean()) for d in range(3)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--models", default="m2,m3")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sizes", default="100,150,200,300")
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--rand-seeds", type=int, default=5)
    ap.add_argument("--kmed-seeds", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--cache", default="results/lomo_fits")
    ap.add_argument("--m3-init", default=None, choices=[None, "median"],
                    help="M3 guide initialisation: default numpyro init_to_uniform, or 'median' (stabilised)")
    ap.add_argument("--lr-decay", type=float, default=None, help="exponential lr decay factor over the run (M3 only)")
    ap.add_argument("--hier-gap", action="store_true", help="hierarchical cutpoint-gap prior (revision default)")
    ap.add_argument("--dump-raw", action="store_true", help="also write per-rotation predictions (incl. sub-scores)")
    ap.add_argument("--out", default="results/lomo_full.json")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    model_names = args.models.split(",")

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    keff = np.asarray(fd.item_keff); item_clip = np.asarray(fd.item_clip)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    item_dim = prep.item_dim_array(fd, args.dim)
    dim_of_item = np.array(prep.item_dim_array(fd, 3))
    base_idx = H.base_indices(fd)
    base_names = [fd.resp_names[i] for i in base_idx]
    p_true = P.ItemParams(a=np.zeros(fd.n_item), b=np.zeros(fd.n_item), k0=np.zeros(fd.n_item),
                          k1=np.zeros(fd.n_item), u=np.zeros(fd.n_clip), keff=keff,
                          family=np.asarray(fd.item_family), clip=item_clip,
                          family_names=fd.family_names)
    true_full = H.true_full_scores(fd, p_true, Y)
    all_items = np.arange(fd.n_item)
    true_sub = {h: sub_scores(Y[hi], all_items, dim_of_item, keff) for hi, h in zip(base_idx, base_names)}
    best_model = max(base_names, key=lambda k: true_full[k])
    worst_model = min(base_names, key=lambda k: true_full[k])
    log(f"clips={fd.n_clip} items={fd.n_item} base={len(base_idx)} best={best_model} worst={worst_model}")

    # results[model][selector][n][h] = {"seeds": [ {est: pred}, ...], "sub": [ {dim: {est: pred}} ]}
    results = {m: {} for m in model_names}
    fit_meta = {m: {} for m in model_names}
    affine = {m: {} for m in model_names}       # affine[model][selector][n][h] = [(slope,intercept)...]

    for hi, h in zip(base_idx, base_names):
        excl = group_members(fd, h)
        train_base = [i for i in base_idx if i not in set(excl)]
        train_names = [fd.resp_names[i] for i in train_base]
        yh = Y[hi]; present_h = yh >= 0
        # fit-free heuristic: cross-model disagreement on TRAINING models only
        cms = clip_mean_scores(Y, keff, item_clip, fd.n_clip, train_base)
        disagree = np.nanstd(cms, axis=0)
        disagree = np.nan_to_num(disagree, nan=0.0)
        for mname in model_names:
            cache_path = HERE / args.cache / mname / f"holdout_{h}.npz"
            fx = {}
            if mname == "m3" and args.m3_init:
                fx["init_values"] = args.m3_init
            if mname == "m3" and args.lr_decay:
                fx["lr_decay"] = args.lr_decay
            p, theta_vi, used, secs = fit_rotation(fd, mname, excl, item_dim, args.dim,
                                                   args.steps, args.seeds, cache_path, fit_extra=fx,
                                                   hier_gap=args.hier_gap)
            fit_meta[mname][h] = {"used_model": used, "fit_seconds": secs}
            clip_info, _ = S.clip_information(p, theta_vi[train_base])
            selectors = {
                "fisher_stratified": [S.greedy_stratified(clip_info, strata, n, min_k=args.min_k) for n in sizes],
                "fisher_unconstrained": [np.argsort(-clip_info)[:n] for n in sizes],
                "disagree_stratified": [S.greedy_stratified(disagree, strata, n, min_k=args.min_k) for n in sizes],
                "disagree_unconstrained": [np.argsort(-disagree)[:n] for n in sizes],
            }
            multi = {
                "stratified_random": [[S.select_stratified_random(strata, n, seed=s, min_k=args.min_k)
                                       for n in sizes] for s in range(args.rand_seeds)],
                "random": [[S.select_random(fd.n_clip, n, seed=s) for n in sizes]
                           for s in range(args.rand_seeds)],
                "kmedoids": [[S.select_kmedoids(p, n, seed=s) for n in sizes]
                             for s in range(args.kmed_seeds)],
            }
            all_sel = {k: [v] for k, v in selectors.items()}
            all_sel.update(multi)
            for sname, seed_lists in all_sel.items():
                results[mname].setdefault(sname, {})
                affine[mname].setdefault(sname, {})
                for k, n in enumerate(sizes):
                    results[mname][sname].setdefault(str(n), {})
                    affine[mname][sname].setdefault(str(n), {})
                    seed_preds, seed_subs, seed_aff = [], [], []
                    for seed_list in seed_lists:
                        Sc = seed_list[k]
                        Sset = set(int(c) for c in Sc)
                        in_S = np.array([c in Sset for c in item_clip])
                        idx_h = np.where(in_S & present_h)[0]
                        # held-out reconstruction
                        rec = REC.reconstruct(p, idx_h, yh[idx_h], lam=args.lam)
                        # training-model calibration pairs from the SAME subset
                        xs, ys, xs_dim, ys_dim = [], [], {d: [] for d in DIMS}, {d: [] for d in DIMS}
                        for ti, tn in zip(train_base, train_names):
                            yt = Y[ti]; idx_t = np.where(in_S & (yt >= 0))[0]
                            rt = REC.reconstruct(p, idx_t, yt[idx_t], lam=args.lam)
                            xs.append(rt["irt"]); ys.append(true_full[tn])
                            ed = irt_expected_by_dim(p, rt["theta"], dim_of_item)
                            for d in DIMS:
                                xs_dim[d].append(ed[d]); ys_dim[d].append(true_sub[tn][d])
                        slope, icpt = affine_fit(xs, ys)
                        irt_cal = icpt + slope * rec["irt"]
                        preds = {"subset_mean": rec["subset_mean"], "irt": rec["irt"],
                                 "irt_cal": irt_cal,
                                 "gp_cal": args.lam * irt_cal + (1 - args.lam) * rec["subset_mean"]}
                        # sub-scores
                        sm_sub = sub_scores(yh, idx_h, dim_of_item, keff)
                        ed_h = irt_expected_by_dim(p, rec["theta"], dim_of_item)
                        sub = {}
                        for d in DIMS:
                            sd, id_ = affine_fit(xs_dim[d], ys_dim[d])
                            sub[d] = {"subset_mean": sm_sub[d], "irt": ed_h[d],
                                      "irt_cal": id_ + sd * ed_h[d]}
                        seed_preds.append(preds); seed_subs.append(sub)
                        seed_aff.append([slope, icpt])
                    results[mname][sname][str(n)][h] = {"seeds": seed_preds, "sub": seed_subs}
                    affine[mname][sname][str(n)][h] = seed_aff
            fs = results[mname]["fisher_stratified"]
            log(f"holdout {h:16s} [{mname}->{used}] " + "  ".join(
                f"n={n}: sm={abs(fs[str(n)][h]['seeds'][0]['subset_mean']-true_full[h])*100:.2f} "
                f"cal={abs(fs[str(n)][h]['seeds'][0]['irt_cal']-true_full[h])*100:.2f}pp" for n in sizes))

    # ------------------------------------------------------------------ aggregate
    def boot_ci(vals, n_boot=2000, seed=0):
        _, lo, hi = H.bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        return [round(lo, 5), round(hi, 5)]

    def tau_boot(pred, true, n_boot=2000, seed=0):
        rng = np.random.default_rng(seed)
        pred, true = np.asarray(pred), np.asarray(true)
        taus = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(pred), len(pred))
            if len(set(true[ix])) < 3:
                continue
            t = kendalltau(pred[ix], true[ix]).statistic
            if np.isfinite(t):
                taus.append(t)
        return [round(float(np.quantile(taus, 0.025)), 3), round(float(np.quantile(taus, 0.975)), 3)]

    trues = np.array([true_full[h] for h in base_names])
    interior = [h for h in base_names if h not in (best_model, worst_model)]
    summary = {m: {} for m in model_names}
    for mname in model_names:
        for sname, by_n in results[mname].items():
            summary[mname][sname] = {}
            for n, by_h in by_n.items():
                n_seeds = len(by_h[base_names[0]]["seeds"])
                entry = {"n_seeds": n_seeds, "estimators": {}, "sub_scores": {}}
                for est in ESTIMATORS:
                    # per-rotation error averaged over seeds; tau per seed then averaged
                    errs_h = {h: float(np.mean([abs(s[est] - true_full[h]) for s in by_h[h]["seeds"]]))
                              for h in base_names}
                    taus = []
                    for si in range(n_seeds):
                        preds = np.array([by_h[h]["seeds"][si][est] for h in base_names])
                        taus.append(kendalltau(preds, trues).statistic)
                    preds0 = np.array([np.mean([s[est] for s in by_h[h]["seeds"]]) for h in base_names])
                    entry["estimators"][est] = {
                        "mae": float(np.mean(list(errs_h.values()))),
                        "mae_ci95": boot_ci(list(errs_h.values())),
                        "mae_interior": float(np.mean([errs_h[h] for h in interior])),
                        "err_best_model": errs_h[best_model], "err_worst_model": errs_h[worst_model],
                        "max_err": float(max(errs_h.values())),
                        "tau_mean_over_seeds": float(np.nanmean(taus)),
                        "tau_of_seed_mean_pred": float(kendalltau(preds0, trues).statistic),
                        "tau_ci95_rotation_boot": tau_boot(preds0, trues),
                        "per_model_err": {h: round(v, 5) for h, v in errs_h.items()},
                    }
                for d in DIMS:
                    entry["sub_scores"][d] = {}
                    td = np.array([true_sub[h][d] for h in base_names])
                    for est in ("subset_mean", "irt", "irt_cal"):
                        errs = [float(np.mean([abs(s[d][est] - true_sub[h][d]) for s in by_h[h]["sub"]]))
                                for h in base_names]
                        pd0 = np.array([np.mean([s[d][est] for s in by_h[h]["sub"]]) for h in base_names])
                        entry["sub_scores"][d][est] = {
                            "mae": float(np.mean(errs)), "mae_ci95": boot_ci(errs),
                            "tau": float(kendalltau(pd0, td).statistic),
                            "tau_ci95_rotation_boot": tau_boot(pd0, td)}
                # affine stability
                sl = np.array([[a[0] for a in affine[mname][sname][n][h]] for h in base_names]).ravel()
                ic = np.array([[a[1] for a in affine[mname][sname][n][h]] for h in base_names]).ravel()
                entry["affine_calibration"] = {"slope_mean": float(sl.mean()), "slope_sd": float(sl.std()),
                                               "slope_min": float(sl.min()), "slope_max": float(sl.max()),
                                               "intercept_mean": float(ic.mean()), "intercept_sd": float(ic.std())}
                summary[mname][sname][n] = entry

    report = {"matrix": args.matrix, "sizes": sizes, "steps": args.steps, "fit_seeds": args.seeds,
              "m3_init": args.m3_init, "lr_decay": args.lr_decay, "hier_gap": args.hier_gap,
              "min_k": args.min_k, "lam": args.lam, "rand_seeds": args.rand_seeds,
              "kmed_seeds": args.kmed_seeds, "best_model": best_model, "worst_model": worst_model,
              "true_full": true_full, "true_subscores": true_sub, "fit_meta": fit_meta,
              "summary": summary,
              "protocol_note": ("Item parameters refit per rotation on 8 base models + their pseudo-"
                                "respondents; held-out model and its ablation runs excluded. irt_cal "
                                "affine map fitted on the 8 training models' same-subset IRT predictions "
                                "(out-of-sample for the held-out model). tau over 9 held-out predictions "
                                "each from its own rotation subset.")}
    if args.dump_raw:
        report["raw"] = results
    (HERE / args.out).write_text(json.dumps(report, indent=2))

    for mname in model_names:
        print(f"\n[LOMO {mname}] MAE pp [CI] / tau  (rows: selector x estimator)")
        for sname in summary[mname]:
            for est in ESTIMATORS:
                cells = []
                for n in sizes:
                    e = summary[mname][sname][str(n)]["estimators"][est]
                    cells.append(f"{e['mae']*100:4.2f}[{e['mae_ci95'][0]*100:.1f},{e['mae_ci95'][1]*100:.1f}]/{e['tau_mean_over_seeds']:+.2f}")
                print(f"  {sname:24s} {est:12s} " + "  ".join(f"{c:>22s}" for c in cells))
        print(f"  sub-score (subset_mean) tau at n={sizes[-1]}: " + "  ".join(
            f"{sname}: " + "/".join(f"{summary[mname][sname][str(sizes[-1])]['sub_scores'][d]['subset_mean']['tau']:+.2f}" for d in DIMS)
            for sname in ("fisher_stratified", "stratified_random", "disagree_stratified")))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
