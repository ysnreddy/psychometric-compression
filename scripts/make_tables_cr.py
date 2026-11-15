"""Generate the camera-ready table bodies and caption facts from the result JSONs.

Camera-ready changes relative to scripts/make_tables.py (the submission generator, kept
unchanged for provenance):
  - LOMO uses M3 rotation fits on all nine rotations (results/camera_ready/lomo_m3_k3_raw.json)
    instead of the M2-driven protocol; the headline estimator is the calibrated blend.
  - New tables: LOMO baselines, quota-strength ablation.
  - Table C2 reports n=50 (was n=25) to match every other budget grid, and includes M1.
Output: paper/cods_camera_ready/tables/*.tex and tables/caption_facts_cr.json
(override the directory with TABLES_DIR, e.g. TABLES_DIR=tables in the artifact repo).

Usage:
    python scripts/make_tables_cr.py
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / os.environ.get("TABLES_DIR", "paper/cods_camera_ready/tables")
OUT.mkdir(parents=True, exist_ok=True)
R = HERE / "results"
CR = R / "camera_ready"
DIMS = ["recognition", "perception_detail", "safety_reasoning"]
facts = {}


def pp(x, nd=1):
    return f"{x * 100:.{nd}f}"


def tau(x):
    return f"{x:.2f}"


def ci(c, nd=1):
    return f"[{c[0] * 100:.{nd}f}, {c[1] * 100:.{nd}f}]"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote tables/{name}")


def load(p):
    return json.loads(Path(p).read_text())


# ---- Table C2: structure vs flatness (n=50 replaces n=25; M1 row added) ----------
c2 = load(R / "c2_headtohead.json")["rungs"]
RUNG = {"M0_flat_2pl": "M0 flat 2PL", "M1_field_2pl": "M1 field 2PL",
        "M2_graded": "M2 graded 1-D", "M3_mirt": "M3 MIRT"}
SIZES_C2 = ["50", "150", "200", "300"]
rows = []
for r, label in RUNG.items():
    cells = []
    for n in SIZES_C2:
        m = c2[r][n]["model_cal"]
        t = m["tau"]
        ts = f"$\\mathbf{{{t:.2f}}}$" if t < 0 else tau(t)
        cells.append(f"{pp(m['mae'])} {ci(m['mae_ci'])} / {ts}")
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
write("tab_c2.tex", [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
                     "rung & " + " & ".join(f"$n{{=}}{n}$" for n in SIZES_C2) + r" \\", r"\midrule",
                     *rows, r"\bottomrule", r"\end{tabular}"])
facts["c2_m2_tau_range"] = [tau(c2["M2_graded"][n]["model_cal"]["tau"]) for n in SIZES_C2]
facts["c2_m1_mae_range"] = [pp(min(c2["M1_field_2pl"][n]["model_cal"]["mae"] for n in SIZES_C2)),
                            pp(max(c2["M1_field_2pl"][n]["model_cal"]["mae"] for n in SIZES_C2))]
facts["c2_m1_tau_max"] = tau(max(c2["M1_field_2pl"][n]["model_cal"]["tau"] for n in SIZES_C2))

# ---- Table LOMO: M3 rotations, Fisher + floors (k=3) ----------------------------
L = load(CR / "lomo_m3_k3_raw.json")
assert all(v["used_model"] == "m3" for v in L["fit_meta"]["m3"].values()), "an M3 rotation fell back"
fs = L["summary"]["m3"]["fisher_stratified"]
rows = []
for n in ["100", "150", "200", "300"]:
    g = fs[n]["estimators"]["gp_cal"]
    sm = fs[n]["estimators"]["subset_mean"]
    sub = fs[n]["sub_scores"]
    cells = [f"{pp(g['mae'], 2)} {ci(g['mae_ci95'], 1)}", pp(g["mae_interior"], 2),
             pp(g["err_best_model"], 2), pp(g["err_worst_model"], 2), tau(g["tau_of_seed_mean_pred"]),
             pp(sm["mae"], 2)]
    for d in DIMS:
        cells.append(f"{pp(sub[d]['irt_cal']['mae'])} / {tau(sub[d]['subset_mean']['tau'])}")
    rows.append(f"{n} & " + " & ".join(cells) + r" \\")
write("tab_lomo.tex", [
    r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
    r" & \multicolumn{6}{c}{aggregate score} & \multicolumn{3}{c}{sub-score: calibrated MAE (pp) / naive $\tau$} \\",
    r"\cmidrule(lr){2-7}\cmidrule(lr){8-10}",
    r"$n$ & blend MAE [95\% CI] & interior & top & bottom & $\tau$ & naive MAE & recognition & perception & safety \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
g200 = fs["200"]["estimators"]["gp_cal"]
facts.update({
    "lomo_n_rot_m3": len(L["fit_meta"]["m3"]),
    "lomo_n200_mae": pp(g200["mae"], 2), "lomo_n200_ci": [pp(g200["mae_ci95"][0], 2), pp(g200["mae_ci95"][1], 2)],
    "lomo_n200_tau": tau(g200["tau_of_seed_mean_pred"]), "lomo_n200_tau_ci": [tau(x) for x in g200["tau_ci95_rotation_boot"]],
    "lomo_n200_interior": pp(g200["mae_interior"], 2), "lomo_n200_top": pp(g200["err_best_model"], 2),
    "lomo_n100_mae": pp(fs["100"]["estimators"]["gp_cal"]["mae"], 2),
    "lomo_n300_mae": pp(fs["300"]["estimators"]["gp_cal"]["mae"], 2),
    "lomo_n300_tau": tau(fs["300"]["estimators"]["gp_cal"]["tau_of_seed_mean_pred"]),
    "lomo_n200_naive_mae": pp(fs["200"]["estimators"]["subset_mean"]["mae"], 2),
    "lomo_n200_naive_tau": tau(fs["200"]["estimators"]["subset_mean"]["tau_of_seed_mean_pred"]),
    "lomo_n200_irtcal_mae": pp(fs["200"]["estimators"]["irt_cal"]["mae"], 2),
    "lomo_n200_sub_tau_naive": [tau(fs["200"]["sub_scores"][d]["subset_mean"]["tau"]) for d in DIMS],
    "lomo_n200_sub_mae_cal": [pp(fs["200"]["sub_scores"][d]["irt_cal"]["mae"]) for d in DIMS],
    "lomo_n200_sub_mae_naive": [pp(fs["200"]["sub_scores"][d]["subset_mean"]["mae"]) for d in DIMS],
    "lomo_affine_slope": [f"{fs['200']['affine_calibration']['slope_mean']:.2f}",
                          f"{fs['200']['affine_calibration']['slope_sd']:.2f}"],
})
# previous (submitted) M2-driven protocol and old M3-with-fallback variant, for the text
old = load(R / "ea3_lomo.json")["by_size"]["200"]
facts["submitted_m2_lomo_n200"] = [pp(old["mae_all"], 2), tau(old["kendall_tau_recon_vs_true"])]

# ---- Table LOMO baselines (n=200, M3 rotations) ---------------------------------
SEL = [("fisher_stratified", "Fisher + floors (ours)"), ("fisher_unconstrained", "Fisher, unconstrained"),
       ("stratified_random", "stratified random"), ("random", "uniform random"),
       ("kmedoids", "$k$-medoids anchors"), ("disagree_stratified", "disagreement + floors")]
# baselines use the 10-seed rerun (same cached rotation fits; random and k-medoids selectors
# are seed-sensitive, so 2-5 seeds are not enough to rank them)
LB = load(CR / "lomo_m3_k3_seeds10.json")
assert LB["kmed_seeds"] == 10 and LB["rand_seeds"] == 10
summ = LB["summary"]["m3"]
rows = []
for s, label in SEL:
    e = summ[s]["200"]["estimators"]
    cells = [f"{pp(e[est]['mae'], 2)} / {tau(e[est]['tau_mean_over_seeds'])}" for est in ("subset_mean", "gp_cal")]
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
write("tab_lomo_baselines.tex", [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
                                 r"selector ($n{=}200$) & naive mean & calibrated blend \\", r"\midrule",
                                 *rows, r"\bottomrule", r"\end{tabular}"])
for s, _ in SEL:
    e = summ[s]["200"]["estimators"]
    facts[f"lomo_base_{s}"] = {est: [pp(e[est]["mae"], 2), tau(e[est]["tau_mean_over_seeds"])]
                               for est in ("subset_mean", "gp_cal")}

rs = load(CR / "released_subset_eval_m3.json")
B = rs["B_fixed_subset_heldout"]["m3"]
assert set(B["rotation_models_used"]) == {"m3"}
facts["released_subset_heldout"] = {
    "blend_mae": pp(B["aggregate"]["gp_cal"]["mae"], 2),
    "blend_ci": [pp(x, 1) for x in B["aggregate"]["gp_cal"]["mae_ci95_rotation_boot"]],
    "blend_tau": tau(B["aggregate"]["gp_cal"]["tau"]),
    "safety_naive_mae": pp(rs["A_in_sample"]["sub_scores"]["safety_reasoning"]["subset_mean"]["mae"]),
    "safety_cal_mae_heldout": pp(B["sub_scores"]["safety_reasoning"]["irt_cal"]["mae"])}

# ---- Table quota ablation (n=200) -----------------------------------------------
KS = [0, 1, 2, 3, 5]
ins = {k: load(CR / "quota_insample" / f"n200_k{k}_gp_irt.json") for k in KS}
lom = {k: load(CR / f"lomo_m3_k{k}_raw.json")["summary"]["m3"] for k in KS}
cols = [("unconstr.", ins[0]["methods"]["fisher_unconstrained"], lom[3]["fisher_unconstrained"]["200"])]
cols += [(f"$k{{=}}{k}$", ins[k]["methods"]["fisher_stratified"], lom[k]["fisher_stratified"]["200"]) for k in KS]
def row(label, f):
    return f"{label} & " + " & ".join(f(i, l) for _, i, l in cols) + r" \\"
write("tab_quota.tex", [
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}", r"\toprule",
    "metric & " + " & ".join(c[0] for c in cols) + r" \\", r"\midrule",
    row("empty strata", lambda i, l: str(i["strata_zero_coverage"])),
    row("worst stratum (pp)", lambda i, l: pp(i["per_stratum_mae_max"])),
    row("mean stratum (pp)", lambda i, l: pp(i["per_stratum_mae_mean"])),
    row("in-sample MAE", lambda i, l: pp(i["aggregate_mae"], 2)),
    row("in-sample $\\tau$", lambda i, l: tau(i["aggregate_tau"])),
    row("LOMO MAE", lambda i, l: pp(l["estimators"]["gp_cal"]["mae"], 2)),
    row("LOMO $\\tau$", lambda i, l: tau(l["estimators"]["gp_cal"]["tau_of_seed_mean_pred"])),
    r"\bottomrule", r"\end{tabular}"])
facts["quota"] = {c[0]: {"empty": c[1]["strata_zero_coverage"], "worst": pp(c[1]["per_stratum_mae_max"]),
                         "mean": pp(c[1]["per_stratum_mae_mean"]), "ins_mae": pp(c[1]["aggregate_mae"], 2),
                         "lomo_mae": pp(c[2]["estimators"]["gp_cal"]["mae"], 2),
                         "lomo_tau": tau(c[2]["estimators"]["gp_cal"]["tau_of_seed_mean_pred"])} for c in cols}

# ---- facts for the D4yq responses (text only) -----------------------------------
ln = load(CR / "loading_null.json")
facts["loading"] = {"paper_swaps": ln["paper_swaps"], "total": ln["total_pairs"],
                    "family_swaps": ln["family_level"]["swaps"], "family_total": ln["family_level"]["total_pairs"],
                    "partitions_n": ln["partitions"]["n"], "partitions_min": ln["partitions"]["swaps_min"],
                    "partitions_max": ln["partitions"]["swaps_max"], "paper_rank": ln["partitions"]["paper_rank_by_swaps"],
                    "perm_mean": round(ln["item_permutation_null"]["mean"], 1),
                    "perm_max": ln["item_permutation_null"]["max"],
                    "perm_p": ln["item_permutation_null"]["p_value_ge_paper"],
                    "boot_ci": ln["clip_bootstrap"]["ci95"]}
sens = load(R / "sensitivity_fits.json")["b_cross_loadings"]
facts["cross_loadings"] = {"waic_gain": sens["compare"]["models"]["M3"]["elpd_diff"],
                           "dse": sens["compare"]["models"]["M3"]["dse"],
                           "primary_action": round(sens["primary_loading_median_by_family"]["action"], 2),
                           "primary_scene": round(sens["primary_loading_median_by_family"]["scene_type"], 2),
                           "theta_tau": {k: round(v, 2) for k, v in sens["theta_vs_subscore_tau"].items()}}
vr = load(CR / "validity_range.json")
v200 = vr["by_n"]["200"]
facts["validity"] = {"range": {k: [round(a, 3), round(b, 3)] for k, (a, b) in vr["calibration_range_9_models"].items()},
                     "agg_inside": pp(v200["aggregate"]["gp_cal"]["summary"]["mae_inside_pp"] / 100, 2),
                     "agg_inside_max": pp(v200["aggregate"]["gp_cal"]["summary"]["max_inside_pp"] / 100, 2),
                     "agg_outside": pp(v200["aggregate"]["gp_cal"]["summary"]["mae_outside_pp"] / 100, 2),
                     "sub_inside_cal": {d: pp(v200["sub_scores"][d]["irt_cal"]["summary"]["mae_inside_pp"] / 100)
                                        for d in DIMS},
                     "sub_outside_cal": {d: pp(v200["sub_scores"][d]["irt_cal"]["summary"]["mae_outside_pp"] / 100)
                                         for d in DIMS},
                     "rho_err_dist_cal": {d: round(v200["sub_scores"][d]["irt_cal"]["summary"]["spearman_err_vs_distance"], 2)
                                          for d in DIMS}}
wc = load(R / "waic_clustered.json")
facts["waic"] = {"diff": wc["elpd_diff_M3_minus_M2"], "dse_naive": wc["dse"]["naive"],
                 "dse_clip": wc["dse"]["clip"]["se"], "z_clip": wc["z"]["clip"]}
for tag, f in (("clip_m3_200", CR / "clip_effect_m3_200_4ch.json"), ("clip_m3_100", CR / "clip_effect_m3_100_4ch.json")):
    if f.exists():
        c = load(f)
        u = c["variants"]["M3_free_u__nuts"]; w = c["with_vs_without_u"]
        facts[tag] = {"chains": c.get("nuts_chains"), "clips": c["sub_clips"], "obs": c["sub_obs"],
                      "sigma_u_nuts": round(u["sigma_u_mean"], 2), "sigma_u_q": [round(x, 2) for x in u["sigma_u_q05_q95"]],
                      "sigma_u_svi": round(c["variants"]["M3_free_u__svi"]["sigma_u_mean"], 3),
                      "clip_info_spearman": round(w["nuts"]["clip_info_spearman"], 3),
                      "top30_overlap": w["nuts"]["top_k_overlap"]["30"],
                      "waic_diff": round(w["nuts_waic_u_minus_no_u"]["elpd_diff"], 1),
                      "waic_dse": round(w["nuts_waic_u_minus_no_u"]["dse"], 1),
                      "nuts_vs_svi_spearman": round(w["no_u_nuts_vs_svi"]["clip_info_spearman"], 3),
                      "rhat_max": round(max(x["rhat_max"] for x in u["nuts_diagnostics"].values()), 3),
                      "theta_ess_min": round(u["nuts_diagnostics"]["theta"]["ess_min"]),
                      "divergences": u["divergences"]}

(OUT / "caption_facts_cr.json").write_text(json.dumps(facts, indent=2))
print("wrote tables/caption_facts_cr.json")
