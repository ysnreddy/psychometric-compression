"""EA.3 — leave-one-MODEL-out validation (C1 compression; answers the Zhang frontier
caution). For each of the 9 base models: refit item parameters on the other 8 base models
(and their pseudo-respondents; the held-out model's pseudo are also excluded — leakage-safe),
select a subset, reconstruct the held-out model's full score from its subset responses, and
compare to truth. Best-model and worst-model holdouts are reported SEPARATELY (extrapolation
beyond the observed ability range is where subset prediction is expected to degrade).

Usage:
    python scripts/run_ea3_lomo.py --model m2 --steps 6000 --sizes 100,150,200
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
def log(m): print(f"[EA.3 +{time.time()-_T0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--model", default="m2", choices=["m2", "m3"])
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sizes", default="100,150,200")
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--estimator", default="subset_mean")
    ap.add_argument("--selector", default="fisher_stratified",
                    choices=["fisher_stratified", "random"])
    ap.add_argument("--out", default="results/ea3_lomo.json")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    item_dim = prep.item_dim_array(fd, args.dim) if args.model == "m3" else None
    # true full score per base model (for MAE / ranking)
    p_full = P.ItemParams(a=np.zeros(fd.n_item), b=np.zeros(fd.n_item), k0=np.zeros(fd.n_item),
                          k1=np.zeros(fd.n_item), u=np.zeros(fd.n_clip),
                          keff=np.asarray(fd.item_keff), family=np.asarray(fd.item_family),
                          clip=np.asarray(fd.item_clip), family_names=fd.family_names)
    true_full = H.true_full_scores(fd, p_full, Y)
    base_idx = H.base_indices(fd)
    base_names = [fd.resp_names[i] for i in base_idx]
    log(f"clips={fd.n_clip} base models={len(base_idx)}; true scores ready")

    # resp indices grouped by base model (base + its pseudo) for leakage-safe exclusion
    def group_members(base_name):
        return [i for i, nm in enumerate(fd.resp_names)
                if (nm == base_name) or nm.startswith(base_name + "__")]

    per_rotation = {}
    fallbacks = []
    for hi, h in zip(base_idx, base_names):
        excl = group_members(h)
        A = prep.rotation_field_arrays(fd, excl)
        margs = (prep.m3_args(A, item_dim, args.dim) if args.model == "m3" else prep.m2_args(A))
        model = models.m3 if args.model == "m3" else models.m2
        rot_item_dim = item_dim
        try:
            r = fit.fit_svi(model, margs, {"use_group": False}, n_steps=args.steps,
                            n_seeds=args.seeds, n_post=200)
        except RuntimeError:
            # M3 can diverge on some 8-model rotation subsets; fall back to M2 for this
            # rotation (the estimator is subset_mean either way — the fit only drives selection)
            log(f"holdout {h}: {args.model} diverged — falling back to m2 for this rotation")
            fallbacks.append(h)
            rot_item_dim = None
            r = fit.fit_svi(models.m2, prep.m2_args(A), {"use_group": False},
                            n_steps=args.steps, n_seeds=args.seeds, n_post=200)
        p = P.ItemParams.from_fit(r["derived"], fd, item_dim=rot_item_dim)
        # training base thetas (exclude held-out group) for Fisher info
        train_base = [i for i in base_idx if i not in set(excl)]
        tmean = np.asarray(r["derived"]["theta"]).mean(0)
        clip_info, _ = S.clip_information(p, tmean[train_base])
        # reconstruct held-out h across sizes
        yh = Y[hi]; present = yh >= 0
        rot = {}
        for n in sizes:
            if args.selector == "fisher_stratified":
                Sc = S.greedy_stratified(clip_info, strata, n, min_k=args.min_k)
            else:
                Sc = S.select_random(fd.n_clip, n, seed=0)
            Sset = set(int(c) for c in Sc)
            item_in_S = np.array([c in Sset for c in p.clip])
            idx = np.where(item_in_S & present)[0]
            rec = REC.reconstruct(p, idx, yh[idx])
            rot[n] = {"pred": float(rec[args.estimator]), "true": float(true_full[h]),
                      "abs_err": abs(rec[args.estimator] - true_full[h]), "n_items": int(len(idx))}
        per_rotation[h] = rot
        log(f"holdout {h}: " + "  ".join(f"n={n}:err={rot[n]['abs_err']*100:.2f}pp" for n in sizes)
            + f"  (fit {r['seconds']}s)")

    # ---- aggregate ---------------------------------------------------------
    best_model = max(true_full, key=lambda k: true_full[k] if k in base_names else -9)
    worst_model = min((k for k in base_names), key=lambda k: true_full[k])
    report = {"matrix": args.matrix, "model": args.model, "selector": args.selector,
              "estimator": args.estimator, "sizes": sizes, "steps": args.steps,
              "best_model": best_model, "worst_model": worst_model, "m2_fallback_rotations": fallbacks,
              "per_rotation": per_rotation, "by_size": {}}
    for n in sizes:
        errs = {h: per_rotation[h][n]["abs_err"] for h in base_names}
        preds = np.array([per_rotation[h][n]["pred"] for h in base_names])
        trues = np.array([true_full[h] for h in base_names])
        interior = [h for h in base_names if h not in (best_model, worst_model)]
        _, lo, hi = H.bootstrap_ci(list(errs.values()))       # over the 9 rotations
        report["by_size"][str(n)] = {
            "mae_all": float(np.mean(list(errs.values()))),
            "mae_ci95": [round(lo, 4), round(hi, 4)],
            "mae_interior": float(np.mean([errs[h] for h in interior])),
            "err_best_model": float(errs[best_model]),
            "err_worst_model": float(errs[worst_model]),
            "kendall_tau_recon_vs_true": float(kendalltau(preds, trues).statistic),
            "max_err": float(max(errs.values())),
        }
    (HERE / args.out).write_text(json.dumps(report, indent=2))

    print(f"\n[EA.3] LOMO ({args.model}, {args.selector}, {args.estimator}) — "
          f"best={best_model} worst={worst_model}")
    print(f"  {'n':>5} {'MAE_all':>8} {'MAE_int':>8} {'err_best':>9} {'err_worst':>10} {'tau':>6}")
    for n in sizes:
        d = report["by_size"][str(n)]
        print(f"  {n:>5} {d['mae_all']*100:7.2f}p {d['mae_interior']*100:7.2f}p "
              f"{d['err_best_model']*100:8.2f}p {d['err_worst_model']*100:9.2f}p {d['kendall_tau_recon_vs_true']:6.2f}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
