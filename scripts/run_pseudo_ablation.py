"""Pseudo-respondent ablation — what does response augmentation actually buy? (EA.6, main)

Fits M2 on (a) the base-only matrix (9 respondents) and (b) the augmented matrix (20
respondents: + prompt/frame perturbations of the same base models), then quantifies:
  1. item-parameter POSTERIOR SD shrinkage (the claimed small-pool stabilisation),
  2. agreement between the two fits' item parameters,
  3. downstream: compression fidelity (fisher_stratified, n=150, gp_cal) and
     leaderboard quality (ability vs true-score corr) under each fit.

Usage:
    python scripts/run_pseudo_ablation.py --steps 4000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, prep, params as P, selection as S, harness as H

_T0 = time.time()
def log(m): print(f"[ABL +{time.time()-_T0:6.1f}s] {m}", flush=True)


def fit_one(matrix, steps, seeds, model_name="m2", dim=3, hier_gap=False, m3_init=None):
    fd = io.load_field_data(HERE / matrix)
    A = prep.field_arrays(fd)
    if model_name == "m3":
        item_dim = prep.item_dim_array(fd, dim)
        fk = {"init_values": m3_init} if m3_init else {}
        r = fit.fit_svi(models.m3, prep.m3_args(A, item_dim, dim), {"use_group": False, "hier_gap": hier_gap},
                        n_steps=steps, n_seeds=seeds, n_post=200, **fk)
        p = P.ItemParams.from_fit(r["derived"], fd, item_dim=np.asarray(item_dim))
    else:
        r = fit.fit_svi(models.m2, prep.m2_args(A), {"use_group": False, "hier_gap": hier_gap},
                        n_steps=steps, n_seeds=seeds, n_post=200)
        p = P.ItemParams.from_fit(r["derived"], fd, item_dim=None)
    draws = {k: np.asarray(r["derived"][k]) for k in ("a", "b", "k0", "k1")}
    theta = H.refine_abilities(p, fd)
    return fd, p, draws, theta, r["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--model", default="m2", choices=["m2", "m3"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="results/pseudo_ablation.json")
    ap.add_argument("--hier-gap", action="store_true")
    ap.add_argument("--m3-init", default=None)
    args = ap.parse_args()

    out = {"steps": args.steps, "n": args.n, "conditions": {}}
    fits = {}
    for cond, matrix in (("base_only", "data/response_matrix_v1_base.parquet"),
                         ("augmented", "data/response_matrix_v1.parquet")):
        log(f"fitting {args.model} on {cond}...")
        fd, p, draws, theta, secs = fit_one(matrix, args.steps, args.seeds, args.model, hier_gap=args.hier_gap, m3_init=args.m3_init)
        fits[cond] = (fd, p, draws, theta)
        # posterior SD of item params (the stabilisation claim)
        sds = {k: float(np.mean(draws[k].std(0))) for k in draws}
        # leaderboard quality
        Y = H.response_table(fd)
        base = H.base_indices(fd)
        true = {fd.resp_names[i]: float((Y[i][Y[i] >= 0] / (fd.item_keff[Y[i] >= 0] - 1)).mean())
                for i in base}
        names = [fd.resp_names[i] for i in base]
        th_agg = theta.mean(axis=1) if theta.ndim == 2 else theta
        corr = float(np.corrcoef([th_agg[i] for i in base], [true[n] for n in names])[0, 1])
        # downstream compression at n
        sdf = pd.read_csv(HERE / "data/clip_strata_v1.csv").set_index("clip_id") \
                .loc[fd.clip_names].reset_index()
        strata = S.build_strata(sdf)
        clip_info, _ = S.clip_information(p, theta[base])
        Sc = S.greedy_stratified(clip_info, strata, args.n, min_k=3)
        m = H.eval_subset(fd, p, Sc, H.true_full_scores(fd, p, Y), Y=Y, estimator="gp_cal")
        out["conditions"][cond] = {
            "n_respondents": fd.n_resp, "fit_seconds": secs,
            "item_param_posterior_sd": {k: round(v, 4) for k, v in sds.items()},
            "ability_vs_truescore_corr": round(corr, 3),
            "downstream_mae_n150": round(m["mae"], 4), "downstream_tau_n150": round(m["kendall_tau"], 3),
        }
        log(f"{cond}: sd(a)={sds['a']:.3f} corr={corr:.3f} mae={m['mae']*100:.2f}pp tau={m['kendall_tau']:.2f}")

    # item-parameter agreement + SD shrinkage between conditions
    (_, pb, db, _), (_, pa, da, _) = fits["base_only"], fits["augmented"]
    out["agreement_corr"] = {k: round(float(np.corrcoef(getattr(pb, k), getattr(pa, k))[0, 1]), 3)
                             for k in ("a", "b", "k0", "k1")}
    out["sd_shrinkage_pct"] = {k: round(float(100 * (1 - np.mean(da[k].std(0)) / np.mean(db[k].std(0)))), 1)
                               for k in db}
    (HERE / args.out).write_text(json.dumps(out, indent=2))
    print("\n[ABL] item-param posterior-SD shrinkage from augmentation (%, positive=stabler):",
          out["sd_shrinkage_pct"])
    print("[ABL] base<->augmented item-param agreement corr:", out["agreement_corr"])
    for c, v in out["conditions"].items():
        print(f"[ABL] {c:10s} corr={v['ability_vs_truescore_corr']} "
              f"mae@150={v['downstream_mae_n150']*100:.2f}pp tau={v['downstream_tau_n150']}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
