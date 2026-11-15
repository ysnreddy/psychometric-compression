"""Clip-clustered paired SE for the M2-vs-M3 WAIC difference.

The ladder's paired dSE treats the 170k field observations as independent, but field items
within a clip are dependent (the paper models this only through family-pooled priors). This
script refits M2 and M3 with the ladder's settings, recomputes the pointwise elpd
difference, and reports the paired SE under three clustering assumptions:
  naive             independent observations (reproduces the ladder's dSE)
  resp_clip_cell    one cluster per (respondent, clip) cell — within-clip dependence of one
                    respondent's answers
  clip              one cluster per clip (most conservative: pools all respondents)

Usage:
    python scripts/run_waic_clustered.py --steps 4000 --seeds 3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, prep
from irtlib.diagnostics import waic_pointwise

_T0 = time.time()
def log(m): print(f"[WAICC +{time.time()-_T0:6.1f}s] {m}", flush=True)


def cluster_se(diff_i, labels):
    """SE of sum(diff_i) with iid clusters: sqrt(C) * std over cluster sums."""
    order = np.argsort(labels, kind="stable")
    lab = labels[order]; d = diff_i[order]
    cuts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    sums = np.add.reduceat(d, cuts)
    return float(np.sqrt(len(sums)) * sums.std()), int(len(sums))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--out", default="results/waic_clustered.json")
    ap.add_argument("--hier-gap", action="store_true")
    ap.add_argument("--m3-init", default=None)
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    A = prep.field_arrays(fd)
    item_dim = prep.item_dim_array(fd, args.dim)
    fitkw = dict(n_steps=args.steps, n_seeds=args.seeds, n_post=200)

    log("fitting M2 ...")
    r2 = fit.fit_svi(models.m2, prep.m2_args(A), {"use_group": False, "hier_gap": args.hier_gap}, **fitkw)
    log(f"M2 done ({r2['seconds']}s); pointwise loglik ...")
    ll2 = fit.pointwise_loglik(models.m2, r2["posterior"], prep.m2_args(A),
                               {"use_group": False, "hier_gap": args.hier_gap})
    log("fitting M3 ...")
    fk3 = dict(fitkw)
    if args.m3_init:
        fk3["init_values"] = args.m3_init
    r3 = fit.fit_svi(models.m3, prep.m3_args(A, item_dim, args.dim),
                     {"use_group": False, "hier_gap": args.hier_gap}, **fk3)
    log(f"M3 done ({r3['seconds']}s); pointwise loglik ...")
    ll3 = fit.pointwise_loglik(models.m3, r3["posterior"],
                               prep.m3_args(A, item_dim, args.dim), {"use_group": False, "hier_gap": args.hier_gap})

    elpd2, _ = waic_pointwise(ll2)
    elpd3, _ = waic_pointwise(ll3)
    diff_i = elpd3 - elpd2

    # obs order in pointwise_loglik: sorted site names -> y_bin block then y_tern block,
    # each indexing the long-format rows via idx_bin / idx_tern.
    idx_bin = np.where(fd.binary_mask)[0]
    idx_tern = np.where(~fd.binary_mask)[0]
    obs_rows = np.concatenate([idx_bin, idx_tern])
    clip = np.asarray(fd.clip)[obs_rows]
    respclip = np.asarray(fd.resp)[obs_rows].astype(np.int64) * (fd.n_clip + 1) + clip

    naive_se = float(np.sqrt(len(diff_i)) * diff_i.std())
    se_cell, n_cell = cluster_se(diff_i, respclip)
    se_clip, n_clip_c = cluster_se(diff_i, clip)
    d = float(diff_i.sum())
    report = {
        "steps": args.steps, "seeds": args.seeds, "n_obs": int(len(diff_i)),
        "elpd_diff_M3_minus_M2": round(d, 1),
        "dse": {"naive": round(naive_se, 1),
                "resp_clip_cell": {"se": round(se_cell, 1), "n_clusters": n_cell},
                "clip": {"se": round(se_clip, 1), "n_clusters": n_clip_c}},
        "z": {"naive": round(d / naive_se, 1), "resp_clip_cell": round(d / se_cell, 1),
              "clip": round(d / se_clip, 1)},
    }
    (HERE / args.out).write_text(json.dumps(report, indent=2))
    log(f"elpd diff M3-M2 = {d:.1f}")
    log(f"dSE naive={naive_se:.1f} (z={d/naive_se:.1f})  "
        f"resp-clip cell={se_cell:.1f} (z={d/se_cell:.1f}, C={n_cell})  "
        f"clip={se_clip:.1f} (z={d/se_clip:.1f}, C={n_clip_c})")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
