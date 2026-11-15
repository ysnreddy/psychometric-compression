"""EA.2 — selection-method comparison (C1 compression, C4 stratified guarantees).

At a fixed full-data item model, compares five clip selectors across subset sizes:
random, stratified-random, k-medoids (anchor clustering), Fisher-unconstrained (greedy info,
no quotas), Fisher-stratified (greedy info under per-stratum floor quotas). Reports score
MAE, Kendall tau, and close-pair ranking accuracy vs subset size (mean +/- sd over seeds).

Uses in-sample item params (standard for method comparison); EA.3 is the held-out test.

Usage:
    python scripts/run_ea2_selection.py --fit results/fits/m3.npz --seeds 5
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, selection as S, harness as H

_T0 = time.time()
def log(m): print(f"[EA.2 +{time.time()-_T0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fit", default="results/fits/m3.npz")
    ap.add_argument("--sizes", default="50,100,150,200,300")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--estimator", default="gp_irt", choices=["irt","gp_irt","subset_mean","irt_cal","gp_cal"])
    ap.add_argument("--out", default="results/ea2_selection.json")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    fd = io.load_field_data(HERE / args.matrix)
    p = P.ItemParams.from_npz(HERE / args.fit)
    log(f"loaded matrix (clips={fd.n_clip}) + fit {Path(args.fit).name} (dim={p.n_dim})")

    # align strata rows to the matrix clip index order (by clip_id, not row order)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    n_clips = fd.n_clip

    # base-model abilities from the fit -> Fisher information.
    # saved theta is (draws, M) for M2 or (draws, M, D) for M3 -> mean over draws.
    bidx = H.base_indices(fd)
    tload = np.load(HERE / args.fit, allow_pickle=True)["theta"]
    tmean = tload  # saved theta is a per-respondent point estimate (M,) or (M,D)
    base_thetas = tmean[bidx]
    clip_info, clip_info_dim = S.clip_information(p, base_thetas)
    log(f"clip information computed; strata={len(set(strata))} groups")

    selectors = {
        "random": lambda n, s: S.select_random(n_clips, n, seed=s),
        "stratified_random": lambda n, s: S.select_stratified_random(strata, n, seed=s, min_k=args.min_k),
        "kmedoids": lambda n, s: S.select_kmedoids(p, n, seed=s),
        "fisher_unconstrained": lambda n, s: np.argsort(-clip_info)[:n],
        "fisher_stratified": lambda n, s: S.greedy_stratified(clip_info, strata, n, min_k=args.min_k),
    }
    deterministic = {"fisher_unconstrained", "fisher_stratified"}

    report = {"matrix": args.matrix, "fit": args.fit, "sizes": sizes, "seeds": args.seeds,
              "estimator": args.estimator, "min_k": args.min_k, "curves": {}}
    for name, sel in selectors.items():
        seeds = (0,) if name in deterministic else tuple(range(args.seeds))
        curve = H.compression_curve(fd, p, sel, sizes, n_clips, seeds=seeds, estimator=args.estimator)
        report["curves"][name] = curve
        log(f"{name}: " + "  ".join(f"n={c['n_clips']}:MAE={c['mae_mean']*100:.1f}pp,tau={c['tau_mean']:.2f}"
                                    for c in curve))

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    # headline: fisher_stratified vs random at each size
    print("\n[EA.2] MAE(pp) / Kendall-tau  by size:")
    print(f"  {'method':22s} " + "  ".join(f"n={n}" for n in sizes))
    for name in selectors:
        row = {c["n_clips"]: c for c in report["curves"][name]}
        print(f"  {name:22s} " + "  ".join(f"{row[n]['mae_mean']*100:4.1f}/{row[n]['tau_mean']:.2f}"
                                           for n in sizes))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
