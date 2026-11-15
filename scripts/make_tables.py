"""Generate LaTeX table bodies for the CODS submission from the result JSONs.

Every numeric cell in the paper's tables is produced here, from the same files the
experiments wrote. Rounding lives in this script only, so table cells, captions, and the
caption-facts block cannot drift apart. Output: paper/cods_submission/tables/*.tex plus
tables/caption_facts.json holding the numbers the prose is allowed to quote.

Usage:
    python scripts/make_tables.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "paper" / "cods_submission" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
R = HERE / "results"

DIMS = ["recognition", "perception_detail", "safety_reasoning"]
facts = {}


def pp(x, nd=1):
    """Score-scale value -> percentage points string."""
    return f"{x * 100:.{nd}f}"


def tau(x):
    return f"{x:.2f}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote tables/{name}")


# ---- Table 2: structure vs flatness (C2 head-to-head) -----------------------
c2 = json.load(open(R / "c2_headtohead.json"))["rungs"]
RUNG_LABEL = {"M0_flat_2pl": "M0 flat 2PL", "M1_field_2pl": "M1 field 2PL", "M2_graded": "M2 graded 1-D",
              "M3_mirt": "M3 MIRT"}
SIZES_C2 = ["25", "150", "200", "300"]
rows = []
for rung, label in RUNG_LABEL.items():
    cells = []
    for n in SIZES_C2:
        m = c2[rung][n]["model_cal"]
        lo, hi = m["mae_ci"]
        t = m["tau"]
        ts = f"$\\mathbf{{{t:.2f}}}$" if t < 0 else tau(t)
        cells.append(f"{pp(m['mae'])} [{lo*100:.1f}, {hi*100:.1f}] / {ts}")
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
write("tab_c2.tex", [
    r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
    "rung & " + " & ".join(f"$n{{=}}{n}$" for n in SIZES_C2) + r" \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"])
facts["c2_m3_n200_mae"] = pp(c2["M3_mirt"]["200"]["model_cal"]["mae"])
facts["c2_m2_tau_range"] = [tau(c2["M2_graded"][n]["model_cal"]["tau"]) for n in SIZES_C2]

# ---- Table 3: stratum coverage (EA.4, n=200) --------------------------------
ea4 = json.load(open(R / "ea4_strata_n200.json"))
u = ea4["methods"]["fisher_unconstrained"]
s = ea4["methods"]["fisher_stratified"]
write("tab_strata.tex", [
    r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
    r"metric & unconstrained & stratified \\", r"\midrule",
    f"strata with zero coverage & {u['strata_zero_coverage']} & {s['strata_zero_coverage']} \\\\",
    f"aggregate MAE (pp) & {pp(u['aggregate_mae'], 2)} & {pp(s['aggregate_mae'], 2)} \\\\",
    f"aggregate Kendall $\\tau$ & {tau(u['aggregate_tau'])} & {tau(s['aggregate_tau'])} \\\\",
    f"per-stratum MAE, mean (pp) & {pp(u['per_stratum_mae_mean'])} & {pp(s['per_stratum_mae_mean'])} \\\\",
    f"per-stratum MAE, max (pp) & {pp(u['per_stratum_mae_max'])} & {pp(s['per_stratum_mae_max'])} \\\\",
    r"\bottomrule", r"\end{tabular}"])
facts["ea4_zero_cov"] = u["strata_zero_coverage"]
facts["ea4_n_strata"] = ea4["n_strata"]
facts["ea4_worst_stratum"] = u["worst_strata"][0][0]
facts["ea4_agg_cost_pp"] = pp(s["aggregate_mae"] - u["aggregate_mae"], 2)

# ---- Table 4: sub-score compression (n=200) ---------------------------------
sub = json.load(open(R / "subscore_compression_n200.json"))["methods"]
SEL_LABEL = {"random": "random", "m0_stratified": "flat M0 information",
             "m3_stratified": "M3 information"}
rows = []
for sel, label in SEL_LABEL.items():
    d = sub[sel]["200"]
    cells = [f"{pp(d[k]['mae'])} / {tau(d[k]['tau'])}" for k in DIMS]
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
write("tab_subscore.tex", [
    r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
    r"selection & recognition & perception & safety \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"])
facts["sub_m3_taus"] = [tau(sub["m3_stratified"]["200"][k]["tau"]) for k in DIMS]
facts["sub_m3_safety_mae"] = pp(sub["m3_stratified"]["200"]["safety_reasoning"]["mae"])
facts["sub_m0_taus"] = [tau(sub["m0_stratified"]["200"][k]["tau"]) for k in DIMS]

# ---- Table 5: LOMO aggregate + per-dimension --------------------------------
lomo = json.load(open(R / "ea3_lomo.json"))["by_size"]
lsub = json.load(open(R / "ea3_lomo_subscores.json"))["by_size"]
rows = []
for n in ["100", "150", "200"]:
    d = lomo[n]
    lo, hi = d["mae_ci95"]
    agg = (f"{pp(d['mae_all'], 2)} [{lo*100:.2f}, {hi*100:.2f}] & "
           f"{pp(d['mae_interior'], 2)} & {pp(d['err_best_model'], 2)} & "
           f"{pp(d['err_worst_model'], 2)} & "
           f"{tau(d['kendall_tau_recon_vs_true'])}")
    dsub = lsub[n]
    per = " & ".join(f"{pp(dsub[k]['mae'])} / {tau(dsub[k]['tau'])}" for k in DIMS)
    rows.append(f"{n} & {agg} & {per} \\\\")
write("tab_lomo.tex", [
    r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
    r" & \multicolumn{5}{c}{aggregate score} & \multicolumn{3}{c}{sub-score MAE (pp) / $\tau$} \\",
    r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
    r"$n$ & MAE [95\% CI] & interior & best & worst & $\tau$ & recognition & perception & safety \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
facts["lomo_n200_mae"] = pp(lomo["200"]["mae_all"], 2)
facts["lomo_n200_ci"] = [f"{x*100:.1f}" for x in lomo["200"]["mae_ci95"]]
facts["lomo_n200_tau"] = tau(lomo["200"]["kendall_tau_recon_vs_true"])
facts["lomo_sub_n200_taus"] = [tau(lsub["200"][k]["tau"]) for k in DIMS]

# ---- Table 6: dimension correlations, raw and latent ------------------------
om = json.load(open(R / "omega_ci.json"))
raw = om["raw_subscores_default_map"]["pearson"]
PAIRS = [("recognition", "perception_detail"), ("recognition", "safety_reasoning"),
         ("perception_detail", "safety_reasoning")]
PAIR_LABEL = {("recognition", "perception_detail"): "recognition--perception",
              ("recognition", "safety_reasoning"): "recognition--safety",
              ("perception_detail", "safety_reasoning"): "perception--safety"}
rows = []
for a, b in PAIRS:
    key = f"{a}__{b}"
    r_raw = raw[key]
    r_def = om["conditions"]["default_merge"]["omega"][key]["mean"]
    r_drop = om["conditions"]["abstain_drop"]["omega"][key]["mean"]
    rows.append(f"{PAIR_LABEL[(a, b)]} & {r_raw:.2f} & {r_def:.2f} & {r_drop:.2f} \\\\")
write("tab_omega.tex", [
    r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
    r"dimension pair & raw Pearson & $\Omega$ (default) & $\Omega$ (abstain-drop) \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
facts["omega_rec_safety_default"] = f"{om['conditions']['default_merge']['omega']['recognition__safety_reasoning']['mean']:.2f}"
facts["omega_rec_safety_drop"] = f"{om['conditions']['abstain_drop']['omega']['recognition__safety_reasoning']['mean']:.2f}"
facts["omega_rec_safety_strict"] = f"{om['conditions']['strict_map']['omega']['recognition__safety_reasoning']['mean']:.2f}"
facts["raw_pearson"] = {PAIR_LABEL[(a, b)]: raw[f'{a}__{b}'] for a, b in PAIRS}

# ---- caption facts from remaining sources -----------------------------------
ea1 = json.load(open(R / "ea1_ladder.json"))
facts["waic_m2"] = round(ea1["models"]["M2_graded"]["elpd_waic"] / 1000, 1)
facts["waic_m3"] = round(ea1["models"]["M3_mirt"]["elpd_waic"] / 1000, 1)
facts["dwaic"] = ea1["M2_vs_M3_compare"]["elpd_diff_M3_minus_M2"]
facts["dwaic_dse"] = ea1["M2_vs_M3_compare"]["compare_table"]["M2_graded"]["dse"]
facts["n_field_obs"] = ea1["n_field_obs"]
wc = json.load(open(R / "waic_clustered.json"))
facts["dwaic_dse_clip"] = wc["dse"]["clip"]["se"]
facts["dwaic_z_clip"] = wc["z"]["clip"]
facts["dwaic_strict"] = json.load(open(R / "ea1_strict.json"))["M2_vs_M3_compare"]["elpd_diff_M3_minus_M2"]
facts["dwaic_absdrop"] = json.load(open(R / "ea1_absdrop.json"))["M2_vs_M3_compare"]["elpd_diff_M3_minus_M2"]
ea5 = json.load(open(R / "ea5_dims.json"))
facts["theta_sub_taus"] = [tau(ea5["per_dim"][k]["theta_vs_subscore_tau"]) for k in DIMS]
facts["unidim_taus"] = [tau(ea5["unidim_mis_ranking"][k]["m2_unidim_vs_this_subscore_tau"]) for k in DIMS]
facts["n_swapping_pairs_theta"] = ea5["fragmentation"]["n_swapping_pairs"]
pa = json.load(open(R / "pseudo_ablation_m3.json"))
facts["pseudo_agreement"] = pa["agreement_corr"]
facts["pseudo_shrink_a"] = pa["sd_shrinkage_pct"]["a"]
facts["pseudo_shrink_b"] = pa["sd_shrinkage_pct"]["b"]
facts["pseudo_tau"] = [tau(pa["conditions"]["base_only"]["downstream_tau_n150"]),
                       tau(pa["conditions"]["augmented"]["downstream_tau_n150"])]
ea7 = json.load(open(R / "ea7_provenance.json"))["methods"]["fisher_stratified"]
facts["provenance_pp"] = {k: pp(v) for k, v in
                          (dict(ea7["worst_strata"]) | {}).items()}
ea2 = json.load(open(R / "ea2_gp_cal.json"))["curves"]
r150 = [r for r in ea2["random"] if r["n_clips"] == 150][0]
f150 = [r for r in ea2["fisher_stratified"] if r["n_clips"] == 150][0]
f200 = [r for r in ea2["fisher_stratified"] if r["n_clips"] == 200][0]
facts["ea2_random_n150"] = {"mae": pp(r150["mae_mean"]), "tau": tau(r150["tau_mean"]),
                            "cp": tau(r150["close_pair_acc"])}
facts["ea2_fisher_n150"] = {"mae": pp(f150["mae_mean"]), "tau": tau(f150["tau_mean"]),
                            "cp": tau(f150["close_pair_acc"])}
facts["ea2_fisher_n200_tau"] = tau(f200["tau_mean"])
facts["ea2_fisher_n200_cp"] = tau(f200["close_pair_acc"])



# ---- NEW tables (review revision, 2026-08-17) --------------------------------
def _opt(path):
    return json.load(open(path)) if Path(path).exists() else None

# LOMO with baselines: selector x estimator at n=200, M2 rotations (paper protocol) and
# stabilised-M3 rotations
lomo = _opt(R / "lomo_full.json"); lomo3 = _opt(R / "lomo_full_m3stab.json")
SEL_LABEL = {"fisher_stratified": "Fisher, stratum floors", "fisher_unconstrained": "Fisher, unconstrained",
             "stratified_random": "stratified random", "random": "uniform random",
             "kmedoids": "$k$-medoids anchors", "disagree_stratified": "disagreement, stratum floors"}
if lomo:
    blocks = [("M2 rotation fits (paper protocol)", lomo["summary"]["m2"])]
    if lomo3:
        blocks.append(("M3 rotation fits (stabilised init, 9/9)", lomo3["summary"]["m3"]))
    rows = []
    for label, S_ in blocks:
        rows.append(r"\multicolumn{4}{@{}l}{\textit{" + label + r"}} \\")
        for sel, sl in SEL_LABEL.items():
            cells = []
            for est in ("subset_mean", "irt_cal", "gp_cal"):
                e = S_[sel]["200"]["estimators"][est]
                lo, hi = e["mae_ci95"]
                cells.append(f"{pp(e['mae'], 2)} [{lo*100:.1f}, {hi*100:.1f}] / {tau(e['tau_mean_over_seeds'])}")
            rows.append(f"{sl} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    write("tab_lomo_baselines.tex", [
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"selector & subset mean & IRT calibrated & blend (0.7/0.3) \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"])
    e = lomo["summary"]["m2"]
    facts["lomo_m2_fisher_sm_n200"] = pp(e["fisher_stratified"]["200"]["estimators"]["subset_mean"]["mae"], 2)
    facts["lomo_m2_random_sm_n200"] = pp(e["random"]["200"]["estimators"]["subset_mean"]["mae"], 2)
    facts["lomo_m2_stratrandom_sm_n200"] = pp(e["stratified_random"]["200"]["estimators"]["subset_mean"]["mae"], 2)
    if lomo3:
        e3 = lomo3["summary"]["m3"]["fisher_stratified"]["200"]["estimators"]
        facts["lomo_m3stab_fisher_gpcal_n200"] = [pp(e3["gp_cal"]["mae"], 2), tau(e3["gp_cal"]["tau_mean_over_seeds"])]
        facts["lomo_m3stab_fisher_irtcal_n200"] = [pp(e3["irt_cal"]["mae"], 2), tau(e3["irt_cal"]["tau_mean_over_seeds"])]
        # sub-scores held-out under M3 stabilised
        ss = lomo3["summary"]["m3"]["fisher_stratified"]["200"]["sub_scores"]
        facts["lomo_m3stab_subscore_tau_naive_n200"] = [tau(ss[d]["subset_mean"]["tau"]) for d in DIMS]
        facts["lomo_m3stab_subscore_mae_naive_n200"] = [pp(ss[d]["subset_mean"]["mae"]) for d in DIMS]
        facts["lomo_m3stab_subscore_mae_cal_n200"] = [pp(ss[d]["irt_cal"]["mae"]) for d in DIMS]

# Reliability
rel = _opt(R / "reliability.json")
if rel:
    w = rel["within_dimension"]; rows = []
    for k, lab in (("recognition", "recognition"), ("perception_detail", "perception"),
                   ("safety_reasoning", "safety"), ("aggregate", "aggregate")):
        rows.append(f"{lab} & {tau(w[k]['tau_boot_vs_full_mean'])} & {tau(w[k]['split_half_tau_mean'])} & "
                    f"{w[k]['n_pairs_stable_95']}/36 & {w[k]['median_se_pp']:.2f} \\\\")
    write("tab_reliability.tex", [
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"quantity & $\tau$(boot, full) & split-half $\tau$ & stable pairs & median SE (pp) \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"])
    x = rel["cross_dimension"]
    facts["rel_cross_tau"] = {k: [tau(v["tau_full"]), [tau(v["tau_boot_ci95"][0]), tau(v["tau_boot_ci95"][1])]] for k, v in x.items()}
    facts["rel_reversals"] = {k: rel["reversals"][k] for k in ("observed_swaps", "expected_under_independence", "stable_swaps_95", "stable_swaps_80", "boot_swaps_mean")}

# Released subset (fixed 200 clips): in-sample and fixed-subset held-out (M3 stabilised rotations)
mini = _opt(R / "released_subset_eval_m3stab.json") or _opt(R / "released_subset_eval.json")
if mini:
    A = mini["A_in_sample"]; B = mini["B_fixed_subset_heldout"].get("m3")
    rows = []
    for est, lab in (("subset_mean", "subset mean"), ("irt_cal", "IRT calibrated"), ("gp_cal", "blend")):
        a = A["aggregate"][est]
        c = [f"{pp(a['mae'], 2)} / {tau(a['tau'])}"]
        c += [f"{pp(A['sub_scores'][d][est]['mae'])} ({A['sub_scores'][d][est]['signed_bias']*100:+.1f}) / {tau(A['sub_scores'][d][est]['tau'])}" for d in DIMS]
        rows.append(f"in-sample & {lab} & " + " & ".join(c) + r" \\")
    if B:
        for est, lab in (("subset_mean", "subset mean"), ("irt_cal", "IRT calibrated"), ("gp_cal", "blend")):
            a = B["aggregate"][est]
            c = [f"{pp(a['mae'], 2)} / {tau(a['tau'])}"]
            c += [f"{pp(B['sub_scores'][d][est]['mae'])} / {tau(B['sub_scores'][d][est]['tau'])}" if est in B["sub_scores"][d] else "--" for d in DIMS]
            rows.append(f"held-out items & {lab} & " + " & ".join(c) + r" \\")
    write("tab_released.tex", [
        r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"validation & estimator & aggregate & recognition & perception & safety \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"])
    facts["mini_safety_naive_mae_bias"] = [pp(A["sub_scores"]["safety_reasoning"]["subset_mean"]["mae"]),
                                           f"{A['sub_scores']['safety_reasoning']['subset_mean']['signed_bias']*100:+.1f}"]
    facts["mini_safety_cal_mae"] = pp(A["sub_scores"]["safety_reasoning"]["irt_cal"]["mae"])

(OUT / "caption_facts.json").write_text(json.dumps(facts, indent=1))
print("wrote tables/caption_facts.json")
print(json.dumps(facts, indent=1))
