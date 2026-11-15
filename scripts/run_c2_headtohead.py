"""C2 head-to-head — does STRUCTURE beat FLATNESS at matched subset budgets?

Each response-model rung competes with its OWN machinery end-to-end:
  M0  clip-level dichotomous 2PL (tinyBenchmarks-style flat baseline): selects clips by its
      clip Fisher information, estimates theta from subset BINARY clip outcomes, predicts the
      full-benchmark pass rate.
  M2  field-level graded (1-D): field-item Fisher selection + graded theta -> expected score.
  M3  confirmatory multidimensional graded: per-dimension information + per-dim theta.

All raw predictions get the SAME leave-one-out affine calibration onto the true ordinal
full-score scale (removes scale mismatch between pass-rate and ordinal-score predictors —
the p-IRT convention of using the training models' known full scores). All selections are
Fisher-stratified with the same strata and min-k. Metrics: MAE + Kendall tau over the 9
base models, bootstrap CI over models. Small n (25, 50) included: the regime where a
model-based estimator should beat the naive subset mean if structure helps.

Also reports, per rung, the naive subset_mean on the SAME subsets (isolates selection value
from estimation value).

Usage:
    python scripts/run_c2_headtohead.py --sizes 25,50,100,150,200,300
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, selection as S, harness as H, reconstruct as R

_T0 = time.time()
def log(m): print(f"[C2 +{time.time()-_T0:6.1f}s] {m}", flush=True)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------- M0 machinery (flat 2PL)
def m0_load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    a = d["a"].mean(0) if d["a"].ndim == 2 else d["a"]          # (C,)
    b = d["b"].mean(0) if d["b"].ndim == 2 else d["b"]          # (C,)
    th = d["theta"].mean(0) if d["theta"].ndim == 2 else d["theta"]  # (M,)
    return a, b, th, list(d["resp_names"]), float(d["threshold"])


def m0_clip_info(a, b, thetas):
    info = np.zeros(len(a))
    for t in np.atleast_1d(thetas):
        p = _sig(a * (t - b))
        info += a ** 2 * p * (1 - p)
    return info / len(np.atleast_1d(thetas))


def m0_theta_mle(a, b, y, grid=None):
    grid = np.linspace(-4, 4, 81) if grid is None else grid
    if len(y) == 0:
        return 0.0
    best, bt = -np.inf, 0.0
    for g in grid:
        p = np.clip(_sig(a * (g - b)), 1e-9, 1 - 1e-9)
        ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum()) - 0.5 * g * g
        if ll > best:
            best, bt = ll, g
    return bt


# ---------------------------------------------------------------- M1 machinery (field-level binary 2PL)
def m1_load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    a = d["a"].mean(0) if d["a"].ndim == 2 else d["a"]          # (J,)
    b = d["b"].mean(0) if d["b"].ndim == 2 else d["b"]          # (J,)
    th = d["theta"].mean(0) if d["theta"].ndim == 2 else d["theta"]  # (M,)
    return a, b, th, list(d["resp_names"])


def m1_clip_info(a, b, thetas, item_clip, n_clip):
    info = np.zeros(len(a))
    for t in np.atleast_1d(thetas):
        p = _sig(a * (t - b))
        info += a ** 2 * p * (1 - p)
    info /= len(np.atleast_1d(thetas))
    return np.bincount(item_clip, weights=info, minlength=n_clip)


def eval_rung(names, raw_pred, sm_pred, true, lam=0.7):
    """Calibrate raw predictions (LOO affine), compute MAE/tau for calibrated + subset-mean."""
    tv = np.array([true[n] for n in names])
    cal = H._loo_affine(np.array(raw_pred), tv)
    sm = np.array(sm_pred)
    out = {}
    gp = lam * cal + (1 - lam) * sm
    for tag, pred in (("model_cal", cal), ("subset_mean", sm), ("gp_cal", gp)):
        errs = np.abs(pred - tv)
        mae, lo, hi = H.bootstrap_ci(errs)
        out[tag] = {"mae": float(errs.mean()), "mae_ci": [round(lo, 4), round(hi, 4)],
                    "tau": float(kendalltau(pred, tv).statistic)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fits", default="results/fits")
    ap.add_argument("--sizes", default="25,50,100,150,200,300")
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--out", default="results/c2_headtohead.json")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    base_idx = H.base_indices(fd)
    names = [fd.resp_names[i] for i in base_idx]
    # target: true ordinal full score (same for all rungs)
    p2 = P.ItemParams.from_npz(HERE / args.fits / "m2.npz")
    true_full = H.true_full_scores(fd, p2, Y)
    log(f"target true scores ready ({len(names)} base models)")

    # ---- M0 rung -----------------------------------------------------------
    a0, b0, th0, m0_names, thr = m0_load(HERE / args.fits / "m0.npz")
    cd = io.load_clip_data(HERE / args.matrix, threshold=thr)
    Yc = np.full((cd.n_resp, cd.n_clip), -1, dtype=np.int64)
    Yc[cd.resp, cd.clip] = cd.y
    m0_base = [i for i, n in enumerate(cd.resp_names) if n in set(names)]
    info0 = m0_clip_info(a0, b0, th0[m0_base])

    # ---- M1 rung (field-level binary 2PL) ----------------------------------
    a1, b1, th1, m1_names = m1_load(HERE / args.fits / "m1.npz")
    item_clip = np.asarray(fd.item_clip); item_keff = np.asarray(fd.item_keff)
    info1 = m1_clip_info(a1, b1, th1[base_idx], item_clip, fd.n_clip)

    # ---- M2 / M3 rungs -----------------------------------------------------
    p3 = P.ItemParams.from_npz(HERE / args.fits / "m3.npz")
    th2 = np.load(HERE / args.fits / "m2.npz", allow_pickle=True)["theta"]
    th3 = np.load(HERE / args.fits / "m3.npz", allow_pickle=True)["theta"]
    info2, _ = S.clip_information(p2, th2[base_idx])
    info3, _ = S.clip_information(p3, th3[base_idx])
    log("clip information ready for all rungs")

    report = {"sizes": sizes, "min_k": args.min_k, "rungs": {}}
    for rung, (info, kind) in {"M0_flat_2pl": (info0, "m0"),
                               "M1_field_2pl": (info1, "m1"),
                               "M2_graded": (info2, "m2"),
                               "M3_mirt": (info3, "m3")}.items():
        p = {"m0": None, "m1": None, "m2": p2, "m3": p3}[kind]
        curve = {}
        for n in sizes:
            Sc = S.greedy_stratified(info, strata, n, min_k=args.min_k)
            Sset = set(int(c) for c in Sc)
            raw, sm = [], []
            for k, i in enumerate(base_idx):
                if kind == "m0":
                    ci = np.array(sorted(Sset))
                    yc = Yc[m0_base[k]][ci]
                    pres = yc >= 0
                    t_hat = m0_theta_mle(a0[ci][pres], b0[ci][pres], yc[pres])
                    raw.append(float(_sig(a0 * (t_hat - b0)).mean()))     # predicted full pass rate
                    sm.append(float(yc[pres].mean()) if pres.any() else 0.0)
                elif kind == "m1":
                    yi = Y[i]
                    item_in = np.array([c in Sset for c in item_clip])
                    idx = np.where(item_in & (yi >= 0))[0]
                    ybin = (yi[idx] == (item_keff[idx] - 1)).astype(float)   # correct vs not
                    t_hat = m0_theta_mle(a1[idx], b1[idx], ybin, grid=np.linspace(-6, 6, 121))
                    raw.append(float(_sig(a1 * (t_hat - b1)).mean()))         # predicted full correct-rate
                    sm.append(float((yi[idx] / (item_keff[idx] - 1)).mean()) if len(idx) else 0.0)
                else:
                    yi = Y[i]
                    item_in = np.array([c in Sset for c in p.clip])
                    idx = np.where(item_in & (yi >= 0))[0]
                    rec = R.reconstruct(p, idx, yi[idx])
                    raw.append(rec["irt"])
                    sm.append(rec["subset_mean"])
            curve[n] = eval_rung(names, raw, sm, true_full)
        report["rungs"][rung] = curve
        log(f"{rung}: " + "  ".join(
            f"n={n}:cal={curve[n]['model_cal']['mae']*100:.1f}pp/sm={curve[n]['subset_mean']['mae']*100:.1f}pp"
            for n in sizes))

    (HERE / args.out).write_text(json.dumps(report, indent=2))

    print("\n[C2] Calibrated model-based MAE(pp) [95% CI] / tau — structure vs flat:")
    print(f"  {'rung':14s} " + "  ".join(f"{'n='+str(n):>16s}" for n in sizes))
    for rung, curve in report["rungs"].items():
        cells = []
        for n in sizes:
            c = curve[n]["model_cal"]
            cells.append(f"{c['mae']*100:4.1f}[{c['mae_ci'][0]*100:.1f},{c['mae_ci'][1]*100:.1f}]/{c['tau']:.2f}")
        print(f"  {rung:14s} " + "  ".join(f"{x:>16s}" for x in cells))
    print("\n[C2] naive subset_mean on the SAME subsets (selection value only):")
    for rung, curve in report["rungs"].items():
        cells = [f"{curve[n]['subset_mean']['mae']*100:4.1f}/{curve[n]['subset_mean']['tau']:.2f}" for n in sizes]
        print(f"  {rung:14s} " + "  ".join(f"{x:>16s}" for x in cells))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
