"""Regenerate the SteelBench-mini guarantee sheet (json + md) for the camera-ready.

The submitted sheet led with the M2-driven LOMO protocol. The camera-ready protocol refits
M3 in every rotation, so the headline held-out numbers, the validity ranges, and the usage
statement change. This script rebuilds the sheet from the camera-ready result files; the
submitted sheet is kept as artifacts/steelbench_mini_guarantee_submitted.{json,md}.
The clip list (artifacts/steelbench_mini_manifest.csv) is unchanged.

Usage:
    python scripts/make_guarantee_cr.py
"""
import json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ART = HERE / "artifacts"
R = HERE / "results"
CR = R / "camera_ready"
DIMS = ["recognition", "perception_detail", "safety_reasoning"]
DL = {"recognition": "recognition", "perception_detail": "perception", "safety_reasoning": "safety"}


def load(p):
    return json.loads(Path(p).read_text())


def pp(x, nd=2):
    return round(100 * x, nd)


def main():
    gj, gm = ART / "steelbench_mini_guarantee.json", ART / "steelbench_mini_guarantee.md"
    bj, bm = ART / "steelbench_mini_guarantee_submitted.json", ART / "steelbench_mini_guarantee_submitted.md"
    if not bj.exists():
        shutil.copy2(gj, bj); shutil.copy2(gm, bm)
    old = load(bj)

    lomo = load(CR / "lomo_m3_k3_raw.json")
    base = load(CR / "lomo_m3_k3_seeds10.json")
    mini = load(CR / "released_subset_eval_m3.json")
    vr = load(CR / "validity_range.json")
    rel = load(R / "reliability.json")
    quota = {k: load(CR / "quota_insample" / f"n200_k{k}_gp_irt.json")["methods"]["fisher_stratified"]
             for k in (3, 5)}
    assert all(v["used_model"] == "m3" for v in lomo["fit_meta"]["m3"].values())

    fs = lomo["summary"]["m3"]["fisher_stratified"]
    held = {n: {"blend_mae_pp": pp(fs[n]["estimators"]["gp_cal"]["mae"]),
                "blend_mae_ci95_pp": [pp(x) for x in fs[n]["estimators"]["gp_cal"]["mae_ci95"]],
                "blend_tau": round(fs[n]["estimators"]["gp_cal"]["tau_of_seed_mean_pred"], 2),
                "blend_tau_ci95": fs[n]["estimators"]["gp_cal"]["tau_ci95_rotation_boot"],
                "interior_mae_pp": pp(fs[n]["estimators"]["gp_cal"]["mae_interior"]),
                "top_model_err_pp": pp(fs[n]["estimators"]["gp_cal"]["err_best_model"]),
                "naive_mae_pp": pp(fs[n]["estimators"]["subset_mean"]["mae"]),
                "subscores": {d: {"calibrated_mae_pp": pp(fs[n]["sub_scores"][d]["irt_cal"]["mae"], 1),
                                  "naive_ranking_tau": round(fs[n]["sub_scores"][d]["subset_mean"]["tau"], 2)}
                              for d in DIMS}}
            for n in ("100", "150", "200", "300")}
    B = mini["B_fixed_subset_heldout"]["m3"]; A = mini["A_in_sample"]
    ranges = vr["calibration_range_9_models"]
    v200 = vr["by_n"]["200"]
    validity = {
        "calibrated_score_ranges": {k: [round(a, 3), round(b, 3)] for k, (a, b) in ranges.items()},
        "inside_range_blend_mae_pp": round(v200["aggregate"]["gp_cal"]["summary"]["mae_inside_pp"], 2),
        "inside_range_blend_max_pp": round(v200["aggregate"]["gp_cal"]["summary"]["max_inside_pp"], 2),
        "edge_rotation_blend_mae_pp": round(v200["aggregate"]["gp_cal"]["summary"]["mae_outside_pp"], 2),
        "subscore_calibrated_mae_pp_inside_vs_outside": {
            d: [round(v200["sub_scores"][d]["irt_cal"]["summary"]["mae_inside_pp"], 1),
                round(v200["sub_scores"][d]["irt_cal"]["summary"]["mae_outside_pp"], 1)] for d in DIMS},
    }
    sel_rows = {}
    for s in ("fisher_stratified", "fisher_unconstrained", "stratified_random", "random", "kmedoids",
              "disagree_stratified"):
        e = base["summary"]["m3"][s]["200"]["estimators"]
        sel_rows[s] = {est: {"mae_pp": pp(e[est]["mae"]), "mae_ci95_pp": [pp(x) for x in e[est]["mae_ci95"]],
                             "tau": round(e[est]["tau_mean_over_seeds"], 2)} for est in ("subset_mean", "gp_cal")}

    g = {k: old[k] for k in ("artifact", "n_clips", "n_items_covered", "selection", "response_model",
                             "matrix_version", "grading_map", "dev_subset")}
    g.update({
        "version": "camera-ready (CODS 2026); clip list unchanged from submission",
        "held_out_protocol": ("LOMO: M3 refit on 8 primary models + their pseudo-respondents in every "
                              "rotation (median init, lr decay; 9/9 converged), selection rerun, affine "
                              "calibration fitted on the 8 training models. Headline estimator: blend "
                              "0.7*calibrated + 0.3*subset mean."),
        "held_out_fidelity_LOMO_M3": held,
        "fixed_subset_heldout_M3": {
            "note": "This exact clip list; item parameters and calibration exclude the held-out model, only the selection saw it.",
            "blend_mae_pp": pp(B["aggregate"]["gp_cal"]["mae"]),
            "blend_mae_ci95_pp": [pp(x) for x in B["aggregate"]["gp_cal"]["mae_ci95_rotation_boot"]],
            "blend_tau": round(B["aggregate"]["gp_cal"]["tau"], 2),
            "subscores": {d: {"naive_mae_pp": pp(B["sub_scores"][d]["subset_mean"]["mae"], 1),
                              "naive_tau": round(B["sub_scores"][d]["subset_mean"]["tau"], 2),
                              "calibrated_mae_pp": pp(B["sub_scores"][d]["irt_cal"]["mae"], 1)} for d in DIMS},
            "in_sample_safety_bias": {"naive_mae_pp": pp(A["sub_scores"]["safety_reasoning"]["subset_mean"]["mae"], 1),
                                      "calibrated_mae_pp": pp(A["sub_scores"]["safety_reasoning"]["irt_cal"]["mae"], 1)}},
        "validity_range": validity,
        "per_stratum_mae_bound_pp": {"k3_released": pp(quota[3]["per_stratum_mae_max"], 1),
                                     "k5_alternative": pp(quota[5]["per_stratum_mae_max"], 1)},
        "held_out_baselines_LOMO_M3_n200": sel_rows,
        "subscore_ranking_reliability_full_benchmark": old["subscore_ranking_reliability_full_benchmark"],
        "validity_conditions": [
            "Validated for models whose aggregate and capability scores fall inside the calibrated ranges "
            "of the nine-model pool (validity_range.calibrated_score_ranges). Outside them the estimate is an "
            "extrapolation: held-out error roughly doubles to quadruples per dimension; confirm on the full benchmark.",
            "Item parameters assume the v1 ordinal grading map; any change to the grading map requires revalidation.",
            "Pseudo-respondents stabilise item parameters but are variants of the nine models, not independent ability points.",
            "The subset is not a substitute for full evaluation when a new capability dimension is introduced.",
        ],
        "usage": [
            "Designed for RANKING models and recovering CAPABILITY PROFILES.",
            "Score levels: use the calibrated estimator (aggregate: blend; capability: calibrated per-dimension); "
            "the naive subset mean is biased on this subset (safety naive MAE ~13 pp).",
            "Capability rankings: use the naive per-dimension subset means.",
            "If only aggregate score LEVELS are needed, a uniform random subset of equal size is at least as accurate.",
        ],
        "responsible_use": ("Report subset-based scores as estimates with the LOMO interval, disclose the subset "
                            "version and size, and state whether the model lies inside the validated range."),
    })
    gj.write_text(json.dumps(g, indent=2))

    L = ["# SteelBench-mini — Guarantee Sheet (camera-ready)", "",
         f"- **Clips:** {g['n_clips']} of 1345 (unchanged from the submitted version), {g['n_items_covered']} field items, 31/31 strata",
         f"- **Selection:** {g['selection']}",
         f"- **Response model:** M3 (confirmatory 3-D graded MIRT); grading map {g['grading_map']}",
         f"- **Dev subset:** {g['dev_subset']['n_clips']} clips, disjoint from the evaluation subset",
         f"- **Held-out protocol:** {g['held_out_protocol']}", "",
         "## How to use", *[f"- {u}" for u in g["usage"]], "",
         "## Validity conditions", *[f"{i}. {c}" for i, c in enumerate(g["validity_conditions"], 1)], "",
         "## Validated score ranges (nine calibration models)", "",
         "| quantity | range |", "|---|---|",
         *[f"| {DL.get(k, k)} | [{a:.3f}, {b:.3f}] |" for k, (a, b) in validity["calibrated_score_ranges"].items()], "",
         f"Inside the range the blend errs by {validity['inside_range_blend_mae_pp']} pp on average "
         f"(max {validity['inside_range_blend_max_pp']} pp); edge rotations by {validity['edge_rotation_blend_mae_pp']} pp. "
         "Calibrated sub-score MAE inside vs outside the range: " +
         ", ".join(f"{DL[d]} {a} vs {b} pp" for d, (a, b) in validity["subscore_calibrated_mae_pp_inside_vs_outside"].items()) + ".",
         "", "## Held-out fidelity (LOMO, M3 in every rotation, Fisher + k=3 floors)", "",
         "| n | blend MAE [95% CI] | interior | top model | tau | naive MAE | recognition cal. MAE / tau | perception | safety |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, h in held.items():
        sub = " | ".join(f"{h['subscores'][d]['calibrated_mae_pp']:.1f} pp / {h['subscores'][d]['naive_ranking_tau']:.2f}" for d in DIMS)
        L.append(f"| {n} | {h['blend_mae_pp']:.2f} pp [{h['blend_mae_ci95_pp'][0]:.1f}, {h['blend_mae_ci95_pp'][1]:.1f}] | "
                 f"{h['interior_mae_pp']:.2f} | {h['top_model_err_pp']:.2f} | {h['blend_tau']:.2f} | {h['naive_mae_pp']:.2f} | {sub} |")
    F = g["fixed_subset_heldout_M3"]
    L += ["", "## This exact clip list, held out (M3 rotation item parameters and calibration)", "",
          f"Aggregate blend: {F['blend_mae_pp']:.2f} pp [{F['blend_mae_ci95_pp'][0]:.1f}, {F['blend_mae_ci95_pp'][1]:.1f}], tau {F['blend_tau']:.2f}.", "",
          "| dimension | naive MAE / tau | calibrated MAE |", "|---|---|---|",
          *[f"| {DL[d]} | {F['subscores'][d]['naive_mae_pp']} pp / {F['subscores'][d]['naive_tau']:.2f} | {F['subscores'][d]['calibrated_mae_pp']} pp |" for d in DIMS],
          "", f"Per-stratum MAE bound (in-sample, n=200): {g['per_stratum_mae_bound_pp']['k3_released']} pp with the released k=3 floors "
          f"({g['per_stratum_mae_bound_pp']['k5_alternative']} pp with k=5).", "",
          "## Held-out baselines (LOMO, same M3 rotation fits, n=200; stochastic selectors 10 seeds)", "",
          "| selector | naive mean MAE [CI] / tau | blend MAE [CI] / tau |", "|---|---|---|",
          *[f"| {s} | {r['subset_mean']['mae_pp']:.2f} [{r['subset_mean']['mae_ci95_pp'][0]:.1f}, {r['subset_mean']['mae_ci95_pp'][1]:.1f}] / {r['subset_mean']['tau']:.2f} | "
            f"{r['gp_cal']['mae_pp']:.2f} [{r['gp_cal']['mae_ci95_pp'][0]:.1f}, {r['gp_cal']['mae_ci95_pp'][1]:.1f}] / {r['gp_cal']['tau']:.2f} |" for s, r in sel_rows.items()],
          "", "## Ranking reliability of the FULL benchmark (clip bootstrap)", "",
          "| quantity | tau(bootstrap vs full) | split-half tau | model pairs stable at 95% |", "|---|---|---|---|"]
    for k, v in g["subscore_ranking_reliability_full_benchmark"].items():
        L.append(f"| {DL.get(k, k)} | {v['tau_boot_vs_full']:.2f} | {v['split_half_tau']:.2f} | {v['stable_pairs_95_of_36']}/36 |")
    L += ["", "The aggregate ranking of the nine models is the least reliable quantity even on the full benchmark; "
          "subset-based aggregate rank claims cannot exceed this ceiling.", "",
          f"**Responsible use.** {g['responsible_use']}", ""]
    gm.write_text("\n".join(L))
    print(f"wrote {gj.relative_to(HERE)} and {gm.relative_to(HERE)} (submitted copies kept as *_submitted.*)")


if __name__ == "__main__":
    main()
