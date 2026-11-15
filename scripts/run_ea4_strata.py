"""EA.4 — stratum-coverage stress (C4, the paper's most quotable finding).

Aggregate-optimal (Fisher-unconstrained) selection silently starves tail strata — rare
visibility conditions / rare-class bands — because those items are low-information for the
AGGREGATE ranking (all models fail them). Per-stratum score MAE then blows up while the
aggregate Kendall-tau barely moves. Stratified selection caps per-stratum MAE for a small
aggregate-information cost. This script quantifies exactly that.

Usage:
    python scripts/run_ea4_strata.py --fit results/fits/m3.npz --n 150
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, selection as S, harness as H

_T0 = time.time()
def log(m): print(f"[EA.4 +{time.time()-_T0:6.1f}s] {m}", flush=True)


def per_stratum_scores(fd, p, Y, strata, base_idx):
    """True per-(model, stratum) mean observed score. Returns {stratum: {model: score}}."""
    out = {}
    for s in sorted(set(strata)):
        clips_s = set(np.where(strata == s)[0])
        item_mask = np.array([c in clips_s for c in p.clip])
        items = np.where(item_mask)[0]
        out[s] = {}
        for i in base_idx:
            yi = Y[i][items]; present = yi >= 0
            out[s][fd.resp_names[i]] = (float((yi[present] / (p.keff[items][present] - 1)).mean())
                                        if present.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fit", default="results/fits/m3.npz")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--estimator", default="gp_irt")
    ap.add_argument("--strata-cols", default="visibility_primary,layer,freq_band",
                    help="comma-sep clip_strata columns defining a stratum; "
                         "use 'calibration_condition' for the EA.7 provenance slices")
    ap.add_argument("--out", default="results/ea4_strata.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    p = P.ItemParams.from_npz(HERE / args.fit)
    Y = H.response_table(fd)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf, cols=tuple(args.strata_cols.split(",")))
    base_idx = H.base_indices(fd)
    tload = np.load(HERE / args.fit, allow_pickle=True)["theta"]
    tmean = tload  # saved theta is a per-respondent point estimate (M,) or (M,D)
    base_thetas = tmean[base_idx]
    clip_info, _ = S.clip_information(p, base_thetas)
    true_full = H.true_full_scores(fd, p, Y)
    true_strat = per_stratum_scores(fd, p, Y, strata, base_idx)
    log(f"clips={fd.n_clip} strata={len(set(strata))} n={args.n}")

    methods = {
        "fisher_unconstrained": np.argsort(-clip_info)[:args.n],
        "fisher_stratified": S.greedy_stratified(clip_info, strata, args.n, min_k=args.min_k),
    }
    report = {"matrix": args.matrix, "fit": args.fit, "n": args.n, "min_k": args.min_k,
              "n_strata": len(set(strata)), "methods": {}}
    def score(yv, kv):
        return (yv / (kv - 1))

    for name, Sc in methods.items():
        Sset = set(int(c) for c in Sc)
        cov = {s: int(np.sum([(c in Sset) for c in np.where(strata == s)[0]]))
               for s in sorted(set(strata))}
        dropped = [s for s, k in cov.items() if k < args.min_k]
        zero_cov = [s for s, k in cov.items() if k == 0]
        m_agg = H.eval_subset(fd, p, Sc, true_full, Y=Y, estimator=args.estimator)
        S_items = np.where(np.array([c in Sset for c in p.clip]))[0]

        # Per-stratum score estimated by SUBSET-MEAN over that stratum's IN-SUBSET clips.
        # When naive selection leaves a stratum with no in-subset clips, the only estimate
        # available is the global subset mean -> error = |stratum_true - global_mean|, which
        # is exactly how aggregate-optimal compression fails the deployment-critical strata.
        strat_mae, band_err = {}, {}
        for s in sorted(set(strata)):
            clips_s = set(np.where(strata == s)[0])
            items_s = np.array([c in clips_s for c in p.clip])
            items_s_in_S = np.where(items_s & np.array([c in Sset for c in p.clip]))[0]
            errs = []
            for i in base_idx:
                yi = Y[i]
                # global fallback = this model's subset-mean over ALL in-subset items
                gpres = yi[S_items] >= 0
                global_est = float(score(yi[S_items][gpres], p.keff[S_items][gpres]).mean())
                sp = yi[items_s_in_S] >= 0
                est = (float(score(yi[items_s_in_S][sp], p.keff[items_s_in_S][sp]).mean())
                       if sp.any() else global_est)
                errs.append(abs(est - true_strat[s][fd.resp_names[i]]))
            strat_mae[s] = float(np.nanmean(errs))
        report["methods"][name] = {
            "aggregate_mae": m_agg["mae"], "aggregate_tau": m_agg["kendall_tau"],
            "n_items": m_agg["n_items"], "strata_dropped_below_k": len(dropped),
            "strata_zero_coverage": len(zero_cov), "zero_cov_strata": zero_cov,
            "per_stratum_mae_mean": float(np.mean(list(strat_mae.values()))),
            "per_stratum_mae_max": float(np.max(list(strat_mae.values()))),
            "worst_strata": sorted(strat_mae.items(), key=lambda kv: -kv[1])[:5],
            "per_stratum_mae": {k: round(v, 4) for k, v in strat_mae.items()},
            "coverage": cov,
        }
        log(f"{name}: agg_MAE={m_agg['mae']*100:.2f}pp agg_tau={m_agg['kendall_tau']:.3f} "
            f"zero_cov_strata={len(zero_cov)} per_stratum_MAE_mean={report['methods'][name]['per_stratum_mae_mean']*100:.1f}pp "
            f"max={report['methods'][name]['per_stratum_mae_max']*100:.1f}pp")

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    u, s = report["methods"]["fisher_unconstrained"], report["methods"]["fisher_stratified"]
    print(f"\n[EA.4] The stratum-collapse exhibit (n={args.n}):")
    print(f"  {'metric':32s} {'unconstrained':>14s} {'stratified':>12s}")
    print(f"  {'aggregate Kendall-tau':32s} {u['aggregate_tau']:>14.3f} {s['aggregate_tau']:>12.3f}")
    print(f"  {'aggregate MAE (pp)':32s} {u['aggregate_mae']*100:>14.2f} {s['aggregate_mae']*100:>12.2f}")
    print(f"  {'strata with ZERO subset coverage':32s} {u['strata_zero_coverage']:>14d} {s['strata_zero_coverage']:>12d}")
    print(f"  {'per-stratum MAE, mean (pp)':32s} {u['per_stratum_mae_mean']*100:>14.2f} {s['per_stratum_mae_mean']*100:>12.2f}")
    print(f"  {'per-stratum MAE, MAX (pp)':32s} {u['per_stratum_mae_max']*100:>14.2f} {s['per_stratum_mae_max']*100:>12.2f}")
    print("  => aggregate tau barely moves; per-stratum MAX MAE is where naive selection fails.")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
