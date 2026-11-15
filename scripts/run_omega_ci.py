"""Omega (dimension-correlation) posterior CIs + raw sub-score correlations.

Defensive analysis for the paper's anti-correlation claim: the saved fits carry only a
point estimate of the M3 ability-correlation matrix. This script refits M3 under three
conditions (default map / abstain-drop / strict map), extracts per-draw Omega = L L^T from
the variational posterior, and reports mean + 95% interval per off-diagonal entry. It also
computes the model-free comparators a reviewer would check first: raw capability sub-score
Pearson correlations and Kendall tau between sub-score rankings across the 9 base models.

Usage:
    python scripts/run_omega_ci.py --steps 4000 --seeds 3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, pearsonr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, prep

DIMS = ["recognition", "perception_detail", "safety_reasoning"]
PAIRS = [(0, 1), (0, 2), (1, 2)]
_T0 = time.time()
def log(m): print(f"[OMEGA +{time.time()-_T0:6.1f}s] {m}", flush=True)


def omega_draws(matrix, abstain, steps, seeds, D=3):
    fd = io.load_field_data(HERE / matrix, abstain=abstain)
    A = prep.field_arrays(fd)
    item_dim = prep.item_dim_array(fd, D)
    r = fit.fit_svi(models.m3, prep.m3_args(A, item_dim, D), {"use_group": False},
                    n_steps=steps, n_seeds=seeds, n_post=200)
    L = np.asarray(r["posterior"]["L_corr"])          # (n_post, D, D)
    om = np.einsum("nij,nkj->nik", L, L)              # per-draw Omega = L L^T
    return om, r["seconds"]


def raw_subscores(matrix, abstain):
    """Per-base-model mean ordinal score per dimension (correct=1, partial=0.5, wrong=0;
    abstain 0 under merge, excluded under drop)."""
    fd = io.load_field_data(HERE / matrix, abstain=abstain)
    dim_of_item = np.array(prep.item_dim_array(fd, 3))
    resp_kind_base = [i for i, n in enumerate(fd.resp_names) if "__" not in n]
    sub = {}
    y = np.asarray(fd.y); resp = np.asarray(fd.resp); item = np.asarray(fd.item)
    keff = np.asarray(fd.keff)
    score = y / (keff - 1)
    for i in resp_kind_base:
        m = fd.resp_names[i]
        sub[m] = []
        for d in range(3):
            mask = (resp == i) & (dim_of_item[item] == d)
            sub[m].append(float(score[mask].mean()))
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/omega_ci.json")
    args = ap.parse_args()

    conditions = {
        "default_merge": ("data/response_matrix_v1.parquet", "merge"),
        "abstain_drop": ("data/response_matrix_v1.parquet", "drop"),
        "strict_map": ("data/response_matrix_v1_v2_strict.parquet", "merge"),
    }
    report = {"steps": args.steps, "seeds": args.seeds, "dims": DIMS, "conditions": {}}

    for name, (matrix, abstain) in conditions.items():
        log(f"fitting M3 [{name}] ...")
        om, secs = omega_draws(matrix, abstain, args.steps, args.seeds)
        entry = {"matrix": matrix, "abstain": abstain, "fit_seconds": secs, "omega": {}}
        for i, j in PAIRS:
            d = om[:, i, j]
            entry["omega"][f"{DIMS[i]}__{DIMS[j]}"] = {
                "mean": round(float(d.mean()), 3),
                "ci95": [round(float(np.percentile(d, 2.5)), 3),
                         round(float(np.percentile(d, 97.5)), 3)],
                "sd": round(float(d.std()), 3),
            }
        report["conditions"][name] = entry
        log(f"  {name}: " + "  ".join(f"{k}={v['mean']} {v['ci95']}"
                                      for k, v in entry["omega"].items()))

    # model-free comparators (reviewer's two-minute check), default map
    sub = raw_subscores("data/response_matrix_v1.parquet", "merge")
    models_l = sorted(sub)
    mat = np.array([sub[m] for m in models_l])        # (9, 3)
    raw = {"per_model_subscores": {m: [round(x, 4) for x in sub[m]] for m in models_l},
           "pearson": {}, "kendall_tau": {}}
    for i, j in PAIRS:
        key = f"{DIMS[i]}__{DIMS[j]}"
        raw["pearson"][key] = round(float(pearsonr(mat[:, i], mat[:, j])[0]), 3)
        raw["kendall_tau"][key] = round(float(kendalltau(mat[:, i], mat[:, j])[0]), 3)
    report["raw_subscores_default_map"] = raw
    log(f"raw sub-score pearson: {raw['pearson']}")
    log(f"raw sub-score kendall: {raw['kendall_tau']}")

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
