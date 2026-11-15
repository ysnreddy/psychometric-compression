"""Per-dimension SUB-SCORE compression — the structured-benchmark capability no flat
compression method reports: does the compressed subset preserve each CAPABILITY sub-score
(recognition / perception-detail / safety), not just the aggregate?

For each base model x dimension: true sub-score = mean ordinal score over all items of that
dimension; subset estimate = mean over the dimension's items inside the subset (plus a
LOO-affine calibrated variant). Selections compared:
  random               (5 seeds)
  m0_stratified        flat clip-2PL information (no field structure) + stratum quotas
  m3_stratified        structured MIRT information + stratum quotas
  m3_unconstrained     structured MIRT information, no quotas

Flat clip-level modelling can still *compute* sub-scores from the data post hoc, but its
selection is blind to the field/dimension composition — the experiment measures what that
blindness costs per capability.

Usage:
    python scripts/run_subscore_compression.py --sizes 50,150,300
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, selection as S, harness as H

DIM_NAME = {0: "recognition", 1: "perception_detail", 2: "safety_reasoning"}
_T0 = time.time()
def log(m): print(f"[SUB +{time.time()-_T0:6.1f}s] {m}", flush=True)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def m0_clip_info(fits_dir):
    d = np.load(fits_dir / "m0.npz", allow_pickle=True)
    a = d["a"].mean(0) if d["a"].ndim == 2 else d["a"]
    b = d["b"].mean(0) if d["b"].ndim == 2 else d["b"]
    th = d["theta"].mean(0) if d["theta"].ndim == 2 else d["theta"]
    info = np.zeros(len(a))
    for t in th:
        pr = _sig(a * (t - b))
        info += a ** 2 * pr * (1 - pr)
    return info / len(th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fits", default="results/fits")
    ap.add_argument("--sizes", default="50,150,300")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--out", default="results/subscore_compression.json")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    fits_dir = HERE / args.fits

    fd = io.load_field_data(HERE / args.matrix)
    p3 = P.ItemParams.from_npz(fits_dir / "m3.npz")
    Y = H.response_table(fd)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    base_idx = H.base_indices(fd)
    names = [fd.resp_names[i] for i in base_idx]
    D = p3.n_dim

    # true per-dimension sub-scores
    dim_items = {d: np.where(p3.item_dim == d)[0] for d in range(D)}
    true_sub = {d: {} for d in range(D)}
    for d in range(D):
        it = dim_items[d]
        for i in base_idx:
            yi = Y[i][it]; pr = yi >= 0
            true_sub[d][fd.resp_names[i]] = float((yi[pr] / (p3.keff[it][pr] - 1)).mean())

    th3 = np.load(fits_dir / "m3.npz", allow_pickle=True)["theta"]
    info3, _ = S.clip_information(p3, th3[base_idx])
    info0 = m0_clip_info(fits_dir)
    log(f"true sub-scores + information ready (D={D})")

    selectors = {
        "random": (lambda n, s: S.select_random(fd.n_clip, n, seed=s), range(args.seeds)),
        "m0_stratified": (lambda n, s: S.greedy_stratified(info0, strata, n, min_k=args.min_k), (0,)),
        "m3_stratified": (lambda n, s: S.greedy_stratified(info3, strata, n, min_k=args.min_k), (0,)),
        "m3_unconstrained": (lambda n, s: np.argsort(-info3)[:n], (0,)),
    }

    report = {"sizes": sizes, "dim_names": [DIM_NAME.get(d, f"dim{d}") for d in range(D)],
              "n_items_per_dim": {DIM_NAME.get(d, f"dim{d}"): int(len(dim_items[d])) for d in range(D)},
              "methods": {}}
    for mname, (sel, seeds) in selectors.items():
        rows = {}
        for n in sizes:
            per_dim = {d: {"maes": [], "taus": [], "cov": []} for d in range(D)}
            for s in seeds:
                Sset = set(int(c) for c in sel(n, s))
                in_S = np.array([c in Sset for c in p3.clip])
                for d in range(D):
                    it_in = dim_items[d][in_S[dim_items[d]]]
                    ests, tv = [], []
                    for i in base_idx:
                        yi = Y[i][it_in]; pr = yi >= 0
                        ests.append(float((yi[pr] / (p3.keff[it_in][pr] - 1)).mean()) if pr.any()
                                    else float("nan"))
                        tv.append(true_sub[d][fd.resp_names[i]])
                    ests, tv = np.array(ests), np.array(tv)
                    per_dim[d]["maes"].append(float(np.nanmean(np.abs(ests - tv))))
                    per_dim[d]["taus"].append(float(kendalltau(ests, tv).statistic))
                    per_dim[d]["cov"].append(int(len(it_in)))
            rows[n] = {}
            for d in range(D):
                errs = per_dim[d]["maes"]
                _, lo, hi = H.bootstrap_ci(errs) if len(errs) > 1 else (0, errs[0], errs[0])
                rows[n][DIM_NAME.get(d, f"dim{d}")] = {
                    "mae": round(float(np.mean(errs)), 4),
                    "mae_ci95": [round(lo, 4), round(hi, 4)],
                    "tau": round(float(np.nanmean(per_dim[d]["taus"])), 3),
                    "n_items_in_subset": int(np.mean(per_dim[d]["cov"])),
                    "item_share_vs_full": round(float(np.mean(per_dim[d]["cov"]) / len(dim_items[d])), 4),
                }
        report["methods"][mname] = rows
        log(f"{mname}: " + "  ".join(
            f"n={n}:" + "/".join(f"{rows[n][DIM_NAME[d]]['mae']*100:.1f}" for d in range(D))
            for n in sizes) + "  (MAE pp per dim r/p/s)")

    (HERE / args.out).write_text(json.dumps(report, indent=2))

    print("\n[SUB] Per-dimension sub-score MAE(pp)/tau at each size (dims: "
          + ", ".join(report["dim_names"]) + "):")
    for n in sizes:
        print(f"  n={n}:")
        for mname, rows in report["methods"].items():
            cells = [f"{rows[n][dn]['mae']*100:4.1f}/{rows[n][dn]['tau']:+.2f}" for dn in report["dim_names"]]
            print(f"    {mname:18s} " + "  ".join(f"{c:>12s}" for c in cells))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
