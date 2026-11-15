"""EA.3b — leave-one-MODEL-out validation of per-dimension SUB-SCORE reconstruction.

The paper's flagship claim is that capability sub-scores are where compression earns its
keep; run_subscore_compression.py validates that on the training pool only. This script
closes the gap: for each of the 9 base models, refit item parameters on the other 8 (and
only their pseudo-respondents — leakage-safe, identical protocol to run_ea3_lomo.py),
select a Fisher-stratified subset from the rotation fit, and reconstruct the held-out
model's per-dimension sub-scores (recognition / perception-detail / safety) by subset mean
over the dimension's in-subset items. Reports per-dimension MAE over rotations and the
Kendall tau of the LOMO-reconstructed sub-score leaderboard vs the true one.

Usage:
    python scripts/run_ea3_lomo_subscores.py --model m2 --steps 3000 --sizes 100,150,200
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, params as P, selection as S, harness as H, prep

DIMS = ["recognition", "perception_detail", "safety_reasoning"]
_T0 = time.time()
def log(m): print(f"[EA.3b +{time.time()-_T0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--model", default="m2", choices=["m2", "m3"])
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sizes", default="100,150,200")
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--out", default="results/ea3_lomo_subscores.json")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    item_dim_fit = prep.item_dim_array(fd, args.dim) if args.model == "m3" else None
    dim_of_item = np.array(prep.item_dim_array(fd, 3))     # GT-defined, fit-independent
    keff = np.asarray(fd.item_keff)
    base_idx = H.base_indices(fd)
    base_names = [fd.resp_names[i] for i in base_idx]

    def sub_scores(y_row, item_idx):
        """Per-dim mean ordinal score over item_idx where the model responded."""
        out = {}
        for d in range(3):
            sel = item_idx[(dim_of_item[item_idx] == d)]
            yv = y_row[sel]; pres = yv >= 0
            out[DIMS[d]] = (float((yv[pres] / (keff[sel][pres] - 1)).mean())
                            if pres.any() else float("nan"))
        return out

    all_items = np.arange(fd.n_item)
    true_sub = {h: sub_scores(Y[hi], all_items) for hi, h in zip(base_idx, base_names)}

    def group_members(base_name):
        return [i for i, nm in enumerate(fd.resp_names)
                if (nm == base_name) or nm.startswith(base_name + "__")]

    per_rotation, fallbacks = {}, []
    for hi, h in zip(base_idx, base_names):
        excl = group_members(h)
        A = prep.rotation_field_arrays(fd, excl)
        margs = (prep.m3_args(A, item_dim_fit, args.dim) if args.model == "m3"
                 else prep.m2_args(A))
        model = models.m3 if args.model == "m3" else models.m2
        rot_item_dim = item_dim_fit
        try:
            r = fit.fit_svi(model, margs, {"use_group": False}, n_steps=args.steps,
                            n_seeds=args.seeds, n_post=200)
        except RuntimeError:
            log(f"holdout {h}: {args.model} diverged — falling back to m2")
            fallbacks.append(h); rot_item_dim = None
            r = fit.fit_svi(models.m2, prep.m2_args(A), {"use_group": False},
                            n_steps=args.steps, n_seeds=args.seeds, n_post=200)
        p = P.ItemParams.from_fit(r["derived"], fd, item_dim=rot_item_dim)
        train_base = [i for i in base_idx if i not in set(excl)]
        tmean = np.asarray(r["derived"]["theta"]).mean(0)
        clip_info, _ = S.clip_information(p, tmean[train_base])
        rot = {}
        for n in sizes:
            Sc = S.greedy_stratified(clip_info, strata, n, min_k=args.min_k)
            Sset = set(int(c) for c in Sc)
            item_in_S = np.where(np.array([c in Sset for c in p.clip]))[0]
            est = sub_scores(Y[hi], item_in_S)
            rot[n] = {d: {"pred": est[d], "true": true_sub[h][d],
                          "abs_err": abs(est[d] - true_sub[h][d])} for d in DIMS}
        per_rotation[h] = rot
        log(f"holdout {h}: " + "  ".join(
            f"n={n}:" + "/".join(f"{rot[n][d]['abs_err']*100:.1f}" for d in DIMS)
            for n in sizes) + "  (rec/perc/saf pp)")

    report = {"matrix": args.matrix, "model": args.model, "selector": "fisher_stratified",
              "estimator": "subset_mean", "sizes": sizes, "steps": args.steps,
              "m2_fallback_rotations": fallbacks, "true_subscores": true_sub,
              "per_rotation": per_rotation, "by_size": {}}
    for n in sizes:
        bs = {}
        for d in DIMS:
            errs = [per_rotation[h][n][d]["abs_err"] for h in base_names]
            preds = np.array([per_rotation[h][n][d]["pred"] for h in base_names])
            trues = np.array([true_sub[h][d] for h in base_names])
            _, lo, hi_ = H.bootstrap_ci(errs)
            bs[d] = {"mae": float(np.mean(errs)), "mae_ci95": [round(lo, 4), round(hi_, 4)],
                     "tau": float(kendalltau(preds, trues).statistic)}
        report["by_size"][str(n)] = bs
    (HERE / args.out).write_text(json.dumps(report, indent=2))

    print(f"\n[EA.3b] LOMO sub-scores ({args.model}, fisher_stratified, subset_mean) — "
          f"MAE pp / tau per dimension:")
    print(f"  {'n':>5} " + " ".join(f"{d:>22}" for d in DIMS))
    for n in sizes:
        bs = report["by_size"][str(n)]
        print(f"  {n:>5} " + " ".join(
            f"{bs[d]['mae']*100:8.2f}p /{bs[d]['tau']:+.2f}     " for d in DIMS))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
