"""Evaluate the ACTUAL released artifact (the fixed 200-clip SteelBench-mini) rather than
a surrogate protocol.

Two views, both on the fixed clip list in artifacts/steelbench_mini_manifest.csv:

  A. in-sample (full-pool M3 fit, results/fits/m3.npz): every base model reconstructed from
     its mini responses — aggregate (subset_mean / irt / irt_cal / gp_cal) and per-dimension
     capability sub-scores (subset_mean / irt / irt_cal), per-model errors, signed bias.
     Caveat: the selection saw all nine models and the item parameters were fitted on them.

  B. fixed-subset held-out (uses the LOMO rotation fits cached by run_lomo_full.py):
     for each base model h, item parameters come from the rotation that excluded h (and its
     pseudo-respondents); h is reconstructed on the FIXED mini; the affine calibration is
     fitted on the 8 training models' mini responses. Item parameters and calibration are
     therefore held out; only the CLIP SELECTION saw h. This is the number a guarantee sheet
     for the released subset needs, and it sits between the fully in-sample view (A) and the
     per-rotation re-selection protocol of the paper's LOMO table.

Usage:
    python scripts/run_released_subset_eval.py
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, selection as S, harness as H, reconstruct as R, prep

DIMS = ["recognition", "perception_detail", "safety_reasoning"]
_T0 = time.time()
def log(m): print(f"[MINI +{time.time()-_T0:6.1f}s] {m}", flush=True)


def load_rotation(fd, path):
    z = np.load(path, allow_pickle=True)
    used = str(z["used_model"])
    idim = np.asarray(z["item_dim"]) if used == "m3" and z["item_dim"].size else None
    return P.ItemParams(a=z["a"], b=z["b"], k0=z["k0"], k1=z["k1"], u=z["u"],
                        keff=np.asarray(fd.item_keff), family=np.asarray(fd.item_family),
                        clip=np.asarray(fd.item_clip), family_names=list(fd.family_names),
                        item_dim=idim, n_dim=(int(idim.max()) + 1 if idim is not None else 1)), used


def affine(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) >= 2 and np.std(x) > 1e-9:
        s, i = np.polyfit(x, y, 1); return float(s), float(i)
    return 0.0, float(y.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fit", default="results/fits/m3.npz")
    ap.add_argument("--manifest", default="artifacts/steelbench_mini_manifest.csv")
    ap.add_argument("--lomo-cache", default="results/lomo_fits")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="results/released_subset_eval.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    keff = np.asarray(fd.item_keff); item_clip = np.asarray(fd.item_clip)
    dim_of_item = np.array(prep.item_dim_array(fd, 3))
    base_idx = H.base_indices(fd); names = [fd.resp_names[i] for i in base_idx]
    p3 = P.ItemParams.from_npz(HERE / args.fit)
    mini_ids = pd.read_csv(HERE / args.manifest)["clip_id"].tolist()
    cidx = {c: k for k, c in enumerate(fd.clip_names)}
    mini = np.array([cidx[c] for c in mini_ids])
    Sset = set(int(c) for c in mini)
    in_S = np.array([c in Sset for c in item_clip])
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    log(f"mini clips={len(mini)} items in mini={int(in_S.sum())} strata covered={len(set(strata[mini]))}/{len(set(strata))}")

    p_true = P.ItemParams(a=np.zeros(fd.n_item), b=np.zeros(fd.n_item), k0=np.zeros(fd.n_item),
                          k1=np.zeros(fd.n_item), u=np.zeros(fd.n_clip), keff=keff,
                          family=np.asarray(fd.item_family), clip=item_clip, family_names=fd.family_names)
    true_full = H.true_full_scores(fd, p_true, Y)
    def sub_true(i, items):
        out = {}
        for d in range(3):
            sel = items[dim_of_item[items] == d]; yv = Y[i][sel]; pr = yv >= 0
            out[DIMS[d]] = float((yv[pr] / (keff[sel][pr] - 1)).mean()) if pr.any() else float("nan")
        return out
    all_items = np.arange(fd.n_item)
    true_sub = {n: sub_true(i, all_items) for i, n in zip(base_idx, names)}
    ta = np.array([true_full[n] for n in names])
    tsub = {d: np.array([true_sub[n][d] for n in names]) for d in DIMS}

    def recon_all(p, rows):
        """irt/subset_mean aggregate + per-dim (subset_mean, irt) for each respondent row on the mini."""
        out = {}
        for i in rows:
            yi = Y[i]; idx = np.where(in_S & (yi >= 0))[0]
            rec = R.reconstruct(p, idx, yi[idx], lam=args.lam)
            es = P.expected_score(p, P.theta_for_items(p, rec["theta"]))
            out[i] = {"subset_mean": rec["subset_mean"], "irt": rec["irt"], "theta": rec["theta"],
                      "sub_sm": sub_true(i, idx),
                      "sub_irt": {DIMS[d]: float(es[dim_of_item == d].mean()) for d in range(3)}}
        return out

    def summarize(preds, trues):
        preds, trues = np.asarray(preds), np.asarray(trues)
        e = np.abs(preds - trues)
        return {"mae": float(e.mean()), "max_err": float(e.max()), "signed_bias": float((preds - trues).mean()),
                "tau": float(kendalltau(preds, trues).statistic),
                "per_model_err_pp": {n: round(float(v) * 100, 2) for n, v in zip(names, e)}}

    report = {"manifest": args.manifest, "n_clips": int(len(mini)), "n_items": int(in_S.sum()),
              "strata_covered": f"{len(set(strata[mini]))}/{len(set(strata))}", "lam": args.lam}

    # ---------------------------------------------------------------- A. in-sample
    ra = recon_all(p3, base_idx)
    sm = np.array([ra[i]["subset_mean"] for i in base_idx]); irt = np.array([ra[i]["irt"] for i in base_idx])
    cal = H._loo_affine(irt, ta)
    A = {"aggregate": {"subset_mean": summarize(sm, ta), "irt": summarize(irt, ta),
                       "irt_cal": summarize(cal, ta), "gp_cal": summarize(args.lam * cal + (1 - args.lam) * sm, ta)},
         "sub_scores": {}}
    for d in DIMS:
        smd = np.array([ra[i]["sub_sm"][d] for i in base_idx]); irtd = np.array([ra[i]["sub_irt"][d] for i in base_idx])
        cald = H._loo_affine(irtd, tsub[d])
        A["sub_scores"][d] = {"subset_mean": summarize(smd, tsub[d]), "irt": summarize(irtd, tsub[d]),
                              "irt_cal": summarize(cald, tsub[d]),
                              "gp_cal": summarize(args.lam * cald + (1 - args.lam) * smd, tsub[d])}
    report["A_in_sample"] = A
    log("A in-sample: agg " + " ".join(f"{e}={A['aggregate'][e]['mae']*100:.2f}pp/{A['aggregate'][e]['tau']:+.2f}"
                                       for e in A["aggregate"]))
    for d in DIMS:
        log(f"   {d:18s} " + " ".join(f"{e}={A['sub_scores'][d][e]['mae']*100:.2f}pp(bias {A['sub_scores'][d][e]['signed_bias']*100:+.1f})/{A['sub_scores'][d][e]['tau']:+.2f}"
                                    for e in ("subset_mean", "irt_cal")))

    # ---------------------------------------------------------------- B. fixed-subset held-out
    report["B_fixed_subset_heldout"] = {}
    for mname in ("m2", "m3"):
        cache = HERE / args.lomo_cache / mname
        if not cache.exists():
            continue
        preds = {e: [] for e in ("subset_mean", "irt", "irt_cal", "gp_cal")}
        sub_preds = {d: {e: [] for e in ("subset_mean", "irt", "irt_cal")} for d in DIMS}
        used_models, missing = [], []
        for hi, h in zip(base_idx, names):
            path = cache / f"holdout_{h}.npz"
            if not path.exists():
                missing.append(h); continue
            p, used = load_rotation(fd, path); used_models.append(used)
            train = [i for i in base_idx if i != hi]
            rr = recon_all(p, train + [hi])
            xs = [rr[i]["irt"] for i in train]; ys = [true_full[fd.resp_names[i]] for i in train]
            s, ic = affine(xs, ys)
            rec = rr[hi]
            preds["subset_mean"].append(rec["subset_mean"]); preds["irt"].append(rec["irt"])
            preds["irt_cal"].append(ic + s * rec["irt"])
            preds["gp_cal"].append(args.lam * (ic + s * rec["irt"]) + (1 - args.lam) * rec["subset_mean"])
            for d in DIMS:
                sd, id_ = affine([rr[i]["sub_irt"][d] for i in train], [true_sub[fd.resp_names[i]][d] for i in train])
                sub_preds[d]["subset_mean"].append(rec["sub_sm"][d]); sub_preds[d]["irt"].append(rec["sub_irt"][d])
                sub_preds[d]["irt_cal"].append(id_ + sd * rec["sub_irt"][d])
        if missing:
            log(f"{mname}: missing rotations {missing} — skipped")
            continue
        B = {"rotation_models_used": used_models,
             "aggregate": {e: summarize(v, ta) for e, v in preds.items()},
             "sub_scores": {d: {e: summarize(v, tsub[d]) for e, v in sub_preds[d].items()} for d in DIMS}}
        # rotation-bootstrap CI on the MAE for the headline estimators
        for e in ("subset_mean", "irt_cal", "gp_cal"):
            errs = np.abs(np.array(preds[e]) - ta)
            _, lo, hi_ = H.bootstrap_ci(errs.tolist())
            B["aggregate"][e]["mae_ci95_rotation_boot"] = [round(lo, 5), round(hi_, 5)]
        report["B_fixed_subset_heldout"][mname] = B
        log(f"B fixed-subset held-out [{mname} rotations]: agg " + " ".join(
            f"{e}={B['aggregate'][e]['mae']*100:.2f}pp/{B['aggregate'][e]['tau']:+.2f}" for e in B["aggregate"]))
        for d in DIMS:
            log(f"   {d:18s} " + " ".join(f"{e}={B['sub_scores'][d][e]['mae']*100:.2f}pp/{B['sub_scores'][d][e]['tau']:+.2f}"
                                        for e in ("subset_mean", "irt_cal")))

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
