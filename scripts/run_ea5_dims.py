"""EA.5 — dimensionality analysis (C3). Uses the M3 fit.

Reports: (a) the ability-dimension correlation matrix; (b) per-dimension base-model
ability rankings vs true per-dimension sub-scores (Kendall tau per dim); (c) rank
FRAGMENTATION — model pairs that swap order between dimensions (the capability-fragmentation
observation, now psychometrically formalised); (d) unidimensional mis-ranking — the M2 1-D
ability ranking vs each dimension's true sub-score, showing a single number mis-ranks
capability sub-scores even when the aggregate survives.

Usage:
    python scripts/run_ea5_dims.py --m3 results/fits/m3.npz --m2 results/fits/m2.npz
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, harness as H

DIM_NAME = {0: "recognition", 1: "perception_detail", 2: "safety_reasoning", 3: "counting"}


def rank(names, scores):
    order = np.argsort(-np.asarray(scores))
    return [names[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--m3", default="results/fits/m3.npz")
    ap.add_argument("--m2", default="results/fits/m2.npz")
    ap.add_argument("--out", default="results/ea5_dims.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    p = P.ItemParams.from_npz(HERE / args.m3)
    Y = H.response_table(fd)
    base_idx = H.base_indices(fd)
    names = [fd.resp_names[i] for i in base_idx]
    D = p.n_dim

    d3 = np.load(HERE / args.m3, allow_pickle=True)
    corr = d3["corr"] if "corr" in d3.files else np.eye(D)
    theta3 = d3["theta"].mean(0) if d3["theta"].ndim == 3 else d3["theta"]   # (M, D)
    theta2 = np.load(HERE / args.m2, allow_pickle=True)["theta"]
    theta2 = theta2.mean(0) if theta2.ndim == 2 else theta2                  # (M,)

    # true per-dimension sub-scores per base model (mean observed score over that dim's items)
    true_sub = {d: {} for d in range(D)}
    for d in range(D):
        items_d = np.where(p.item_dim == d)[0]
        for i in base_idx:
            yi = Y[i][items_d]; present = yi >= 0
            true_sub[d][fd.resp_names[i]] = float((yi[present] / (p.keff[items_d][present] - 1)).mean())

    report = {"matrix": args.matrix, "n_dim": D, "dim_names": [DIM_NAME.get(d, f"dim{d}") for d in range(D)],
              "dimension_correlation": np.round(corr, 3).tolist(),
              "per_dim": {}, "unidim_mis_ranking": {}, "fragmentation": {}}

    # per-dim: theta3[:,d] ranking vs true sub-score ranking
    m3_theta_by_dim = {}
    for d in range(D):
        th = theta3[base_idx, d]
        ts = np.array([true_sub[d][n] for n in names])
        m3_theta_by_dim[d] = th
        report["per_dim"][DIM_NAME.get(d, f"dim{d}")] = {
            "theta_vs_subscore_tau": float(kendalltau(th, ts).statistic),
            "ability_ranking": rank(names, th),
            "true_subscore_ranking": rank(names, ts),
        }

    # unidimensional mis-ranking: M2 1-D theta vs each dim's true sub-score
    th2 = theta2[base_idx]
    for d in range(D):
        ts = np.array([true_sub[d][n] for n in names])
        report["unidim_mis_ranking"][DIM_NAME.get(d, f"dim{d}")] = {
            "m2_unidim_vs_this_subscore_tau": float(kendalltau(th2, ts).statistic)}

    # fragmentation: pairs that swap order between two dimensions (using M3 per-dim theta)
    frag_examples = []
    for da in range(D):
        for db in range(da + 1, D):
            ta, tb = m3_theta_by_dim[da], m3_theta_by_dim[db]
            for x in range(len(names)):
                for y in range(x + 1, len(names)):
                    # strict inversion only: a grid-MAP theta tie in one dimension is not
                    # a rank swap, so require strictly opposite orderings
                    if np.sign(ta[x] - ta[y]) * np.sign(tb[x] - tb[y]) < 0:
                        frag_examples.append({
                            "model_a": names[x], "model_b": names[y],
                            "dim_a": DIM_NAME.get(da, f"dim{da}"), "dim_b": DIM_NAME.get(db, f"dim{db}"),
                            "order_dim_a": f"{names[x]}{'>' if ta[x]>ta[y] else '<'}{names[y]}",
                            "order_dim_b": f"{names[x]}{'>' if tb[x]>tb[y] else '<'}{names[y]}"})
    report["fragmentation"] = {"n_swapping_pairs": len(frag_examples), "examples": frag_examples[:10]}

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    print("\n[EA.5] Ability-dimension correlation:")
    print(np.round(corr, 2))
    print("[EA.5] per-dimension theta-vs-subscore Kendall-tau:")
    for d in range(D):
        dn = DIM_NAME.get(d, f"dim{d}")
        print(f"   {dn:20s} tau={report['per_dim'][dn]['theta_vs_subscore_tau']:.2f}  "
              f"| M2-unidim vs this subscore tau={report['unidim_mis_ranking'][dn]['m2_unidim_vs_this_subscore_tau']:.2f}")
    print(f"[EA.5] rank fragmentation: {len(frag_examples)} model-pairs swap order across dimensions")
    for e in frag_examples[:3]:
        print(f"   {e['dim_a']}: {e['order_dim_a']}   vs   {e['dim_b']}: {e['order_dim_b']}")
    print(f"[EA.5] wrote {args.out}")


if __name__ == "__main__":
    main()
