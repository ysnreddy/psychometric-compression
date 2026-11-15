"""EA.8 — artifact delivery (C5): SteelBench-mini + frozen dev subset + guarantee sheet.

Selects SteelBench-mini (default 150 clips) by Fisher-stratified selection on the full M3
fit, then a dev subset (default 200) from the complement (disjoint where possible), and
emits manifests plus a guarantee sheet (per-size tau/MAE with CIs pulled from EA.3, per-
stratum MAE bound from EA.4, and explicit validity conditions incl. the no-extrapolation
caveat). Also a sanity table: dev-subset reconstructed vs full scores for all 9 models.

Usage:
    python scripts/run_ea8_artifacts.py --fit results/fits/m3.npz --mini 150 --dev 200
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, params as P, selection as S, harness as H


def _load(path):
    return json.load(open(path)) if Path(path).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--strata", default="data/clip_strata_v1.csv")
    ap.add_argument("--fit", default="results/fits/m3.npz")
    ap.add_argument("--mini", type=int, default=150)
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()
    outdir = HERE / args.outdir
    outdir.mkdir(exist_ok=True)

    fd = io.load_field_data(HERE / args.matrix)
    p = P.ItemParams.from_npz(HERE / args.fit)
    Y = H.response_table(fd)
    sdf = pd.read_csv(HERE / args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
    strata = S.build_strata(sdf)
    base_idx = H.base_indices(fd)
    tload = np.load(HERE / args.fit, allow_pickle=True)["theta"]
    tmean = tload  # saved theta is a per-respondent point estimate (M,) or (M,D)
    clip_info, _ = S.clip_information(p, tmean[base_idx])
    true_full = H.true_full_scores(fd, p, Y)

    # SteelBench-mini
    mini = S.greedy_stratified(clip_info, strata, args.mini, min_k=args.min_k)
    mini_set = set(int(c) for c in mini)
    # Dev subset from the complement (disjoint where possible)
    remaining = np.array([c for c in range(fd.n_clip) if c not in mini_set])
    info_rem = clip_info.copy(); info_rem[list(mini_set)] = -np.inf
    dev = S.greedy_stratified(info_rem, strata, args.dev, min_k=args.min_k)
    dev = np.array([c for c in dev if c not in mini_set])[:args.dev]

    def manifest(sel, path, label):
        rows = [{"clip_id": fd.clip_names[c], "stratum": strata[c],
                 "clip_info": round(float(clip_info[c]), 4)} for c in sel]
        pd.DataFrame(rows).to_csv(path, index=False)
        return len(rows)

    n_mini = manifest(mini, outdir / "steelbench_mini_manifest.csv", "mini")
    n_dev = manifest(dev, outdir / "dev_subset_manifest.csv", "dev")

    # sanity: reconstruct all base models from dev subset vs full
    m_dev = H.eval_subset(fd, p, dev, true_full, Y=Y, estimator="gp_cal")
    sanity = {k: {"full_score": round(v["true"], 4), "dev_recon": round(v["pred"], 4),
                  "abs_err_pp": round(abs(v["true"] - v["pred"]) * 100, 2)}
              for k, v in m_dev["estimates"].items()}

    # guarantee numbers from EA.3 (LOMO CIs) + EA.3b (LOMO sub-scores) + EA.4 (per-stratum bound)
    ea3 = _load(HERE / "results/ea3_lomo.json")
    ea3b = _load(HERE / "results/ea3_lomo_subscores.json")
    ea4 = _load(HERE / "results/ea4_strata_n200.json") or _load(HERE / "results/ea4_strata.json")
    guarantee = {
        "artifact": "SteelBench-mini", "n_clips": n_mini, "n_items_covered":
            int(sum(1 for c in p.clip if c in mini_set)),
        "selection": "Fisher-information greedy under per-stratum floor quotas "
                     f"(visibility x layer x freq-band, min {args.min_k}/stratum)",
        "response_model": Path(args.fit).stem, "matrix_version": "v1",
        "grading_map": "v1_default (sha in grading_maps/)",
        "held_out_fidelity_EA3": (ea3.get("by_size") if ea3 else "run run_ea3_lomo.py"),
        "held_out_subscores_EA3b": (ea3b.get("by_size") if ea3b
                                    else "run run_ea3_lomo_subscores.py"),
        "per_stratum_mae_bound_EA4": (ea4["methods"]["fisher_stratified"]["per_stratum_mae_max"]
                                      if ea4 else "run run_ea4_strata.py"),
        "dev_subset": {"n_clips": n_dev, "disjoint_from_mini": True,
                       "sanity_dev_vs_full": sanity, "sanity_mae_pp":
                           round(float(np.mean([s["abs_err_pp"] for s in sanity.values()])), 2)},
        "validity_conditions": [
            "Estimates are calibrated on a 9-model black-box respondent pool; extrapolation "
            "beyond that ability range is not guaranteed (see EA.3 best-model holdout).",
            "Item parameters assume the v1 ordinal grading map; a stricter map (v2_strict) "
            "shifts absolute scores — re-derive the subset if the map changes.",
            "The subset reproduces AGGREGATE and per-stratum scores; it is not a substitute "
            "for full evaluation when a new capability dimension is introduced.",
            "Pseudo-respondents (prompt/frame perturbations) stabilise item parameters but do "
            "not add independent ability points; ranking claims rest on the 9 base models.",
        ],
        "responsible_use": "Report subset-based scores as estimates with the EA.3 CI, not as "
                           "exact full-benchmark numbers; disclose the subset size and version.",
    }
    (outdir / "steelbench_mini_guarantee.json").write_text(json.dumps(guarantee, indent=2))

    # human-readable guarantee sheet
    md = [f"# SteelBench-mini — Guarantee Sheet",
          f"", f"- **Clips:** {n_mini} (of {fd.n_clip})  ~{100*n_mini/fd.n_clip:.1f}% compression",
          f"- **Selection:** {guarantee['selection']}",
          f"- **Response model:** {guarantee['response_model']};  grading map: {guarantee['grading_map']}",
          f"- **Dev subset:** {n_dev} clips, disjoint; dev-vs-full sanity MAE "
          f"{guarantee['dev_subset']['sanity_mae_pp']}pp", f"", f"## Validity conditions"]
    md += [f"{i+1}. {c}" for i, c in enumerate(guarantee["validity_conditions"])]
    md += [f"", f"**Responsible use.** {guarantee['responsible_use']}"]
    if ea3:
        md += [f"", f"## Held-out fidelity (EA.3 LOMO)", "| n | MAE all | MAE interior | err best-model | tau |",
               "|---|---|---|---|---|"]
        for n, d in ea3["by_size"].items():
            md.append(f"| {n} | {d['mae_all']*100:.2f}pp | {d['mae_interior']*100:.2f}pp "
                      f"| {d['err_best_model']*100:.2f}pp | {d['kendall_tau_recon_vs_true']:.2f} |")
    if ea3b:
        md += [f"", f"## Held-out capability sub-scores (EA.3b LOMO; naive subset-mean levels, ranking tau)",
               "| n | recognition MAE/tau | perception MAE/tau | safety MAE/tau |",
               "|---|---|---|---|"]
        for n, d in ea3b["by_size"].items():
            md.append("| " + n + " | " + " | ".join(
                f"{d[k]['mae']*100:.2f}pp / {d[k]['tau']:.2f}"
                for k in ("recognition", "perception_detail", "safety_reasoning")) + " |")
    (outdir / "steelbench_mini_guarantee.md").write_text("\n".join(md))

    print(f"[EA.8] SteelBench-mini: {n_mini} clips -> {outdir}/steelbench_mini_manifest.csv")
    print(f"[EA.8] dev subset: {n_dev} clips (disjoint)  dev-vs-full sanity MAE="
          f"{guarantee['dev_subset']['sanity_mae_pp']}pp")
    print(f"[EA.8] mini stratum coverage: {len(set(strata[mini]))}/{len(set(strata))} strata")
    print(f"[EA.8] sanity (dev recon vs full):")
    for k, v in sorted(sanity.items(), key=lambda kv: -kv[1]['full_score']):
        print(f"     {k:20s} full={v['full_score']:.3f} dev={v['dev_recon']:.3f} err={v['abs_err_pp']:.2f}pp")
    print(f"[EA.8] guarantee sheet -> {outdir}/steelbench_mini_guarantee.md")


if __name__ == "__main__":
    main()
