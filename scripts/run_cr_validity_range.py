"""Camera-ready (reviewer D4yq #4): a validity range for future models.

Reads a LOMO dump (`run_lomo_full.py --dump-raw`) and, for every held-out rotation, places
the held-out model relative to the score range of its eight training models:
  inside   -> interpolation (the regime the guarantee sheet should cover)
  outside  -> extrapolation, with the distance beyond the nearest training edge
Done for the aggregate score and for each capability sub-score. Reports per-rotation
errors, interior vs extrapolation MAE, and the full nine-model calibration range that the
released subset's validity statement should quote.

Usage:
    python scripts/run_cr_validity_range.py --lomo results/camera_ready/lomo_m3_k3_raw.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent.parent
DIMS = ["recognition", "perception_detail", "safety_reasoning"]


def position(x, train):
    lo, hi = min(train), max(train)
    if x < lo:
        return "below", lo - x
    if x > hi:
        return "above", x - hi
    return "inside", 0.0


def summarize(rows):
    ins = [r["err_pp"] for r in rows if r["position"] == "inside"]
    out = [r["err_pp"] for r in rows if r["position"] != "inside"]
    dist = [r["dist_pp"] for r in rows]; err = [r["err_pp"] for r in rows]
    return {"n_inside": len(ins), "n_outside": len(out),
            "mae_inside_pp": float(np.mean(ins)) if ins else None,
            "max_inside_pp": float(np.max(ins)) if ins else None,
            "mae_outside_pp": float(np.mean(out)) if out else None,
            "max_outside_pp": float(np.max(out)) if out else None,
            "spearman_err_vs_distance": float(spearmanr(dist, err).statistic) if len(set(dist)) > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lomo", default="results/camera_ready/lomo_m3_k3_raw.json")
    ap.add_argument("--selector", default="fisher_stratified")
    ap.add_argument("--sizes", default="150,200,300")
    ap.add_argument("--out", default="results/camera_ready/validity_range.json")
    args = ap.parse_args()

    d = json.loads((HERE / args.lomo).read_text())
    raw = d["raw"]["m3"][args.selector]
    tf, ts = d["true_full"], d["true_subscores"]
    names = sorted(tf)
    report = {"lomo": args.lomo, "selector": args.selector,
              "calibration_range_9_models": {
                  "aggregate": [min(tf.values()), max(tf.values())],
                  **{dm: [min(ts[n][dm] for n in names), max(ts[n][dm] for n in names)] for dm in DIMS}},
              "by_n": {}}
    for n in args.sizes.split(","):
        by_h = raw[n]
        ent = {"aggregate": {}, "sub_scores": {}}
        for est in ("subset_mean", "irt_cal", "gp_cal"):
            rows = []
            for h in names:
                pred = float(np.mean([s[est] for s in by_h[h]["seeds"]]))
                pos, dist = position(tf[h], [tf[t] for t in names if t != h])
                rows.append({"model": h, "true": tf[h], "pred": pred, "err_pp": abs(pred - tf[h]) * 100,
                             "position": pos, "dist_pp": dist * 100})
            ent["aggregate"][est] = {"summary": summarize(rows), "rows": rows}
        for dm in DIMS:
            ent["sub_scores"][dm] = {}
            for est in ("subset_mean", "irt_cal"):
                rows = []
                for h in names:
                    pred = float(np.mean([s[dm][est] for s in by_h[h]["sub"]]))
                    pos, dist = position(ts[h][dm], [ts[t][dm] for t in names if t != h])
                    rows.append({"model": h, "true": ts[h][dm], "pred": pred,
                                 "err_pp": abs(pred - ts[h][dm]) * 100, "position": pos, "dist_pp": dist * 100})
                ent["sub_scores"][dm][est] = {"summary": summarize(rows), "rows": rows}
        report["by_n"][n] = ent

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    cr = report["calibration_range_9_models"]
    print("calibration range (9 models): " + "  ".join(f"{k}=[{v[0]:.3f}, {v[1]:.3f}] (span {100*(v[1]-v[0]):.1f} pp)"
                                                      for k, v in cr.items()))
    for n, ent in report["by_n"].items():
        print(f"\n== n={n} ({args.selector})")
        for est, e in ent["aggregate"].items():
            s = e["summary"]
            outs = ", ".join(f"{r['model']}({r['position']} {r['dist_pp']:.1f}pp -> {r['err_pp']:.2f})"
                             for r in e["rows"] if r["position"] != "inside")
            print(f"  aggregate {est:12s} inside {s['n_inside']}: MAE {s['mae_inside_pp']:.2f} max {s['max_inside_pp']:.2f} | "
                  f"outside {s['n_outside']}: MAE {s['mae_outside_pp']:.2f} [{outs}]")
        for dm in DIMS:
            for est, e in ent["sub_scores"][dm].items():
                s = e["summary"]
                print(f"  {dm:18s} {est:12s} inside {s['n_inside']}: MAE {s['mae_inside_pp']:.2f} max {s['max_inside_pp']:.2f} | "
                      f"outside {s['n_outside']}: MAE {s['mae_outside_pp'] if s['mae_outside_pp'] is None else round(s['mae_outside_pp'], 2)}"
                      f"  rho(err,dist)={s['spearman_err_vs_distance']}")


if __name__ == "__main__":
    main()
