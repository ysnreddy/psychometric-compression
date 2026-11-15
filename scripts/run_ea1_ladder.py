"""EA.1 — response-model ladder M0->M3 on the full response matrix.

Fits M0 (clip-level 2PL, threshold sweep), M1 (field 2PL), M2 (field graded + clip
effect), M3 (confirmatory MIRT). Reports LOO (M2 vs M3 is the clean multidimensionality
test — same ordinal obs set; M1 is a different response coding, reference only), PPC
category marginals, and SVI seed stability. Saves fitted posteriors for the selection /
reconstruction modules (Milestone 3). The C2 "structure beats flat" downstream curve
(reconstruction MAE / tau vs subset size) is produced in EA.2 using these fits.

Usage:
    python scripts/run_ea1_ladder.py                 # augmented matrix, D=3
    python scripts/run_ea1_ladder.py --matrix data/response_matrix_v1_base.parquet --steps 8000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, diagnostics, prep

_T0 = time.time()
def log(msg):
    print(f"[EA.1 +{time.time()-_T0:6.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=3, help="M3 ability dimensions")
    ap.add_argument("--abstain", default="merge", choices=["merge", "drop"])
    ap.add_argument("--out", default="results/ea1_ladder.json")
    ap.add_argument("--save-fits", default="results/fits")
    ap.add_argument("--stability", action="store_true",
                    help="also run the 5-seed ELBO-spread stability diagnostic")
    ap.add_argument("--hier-gap", action="store_true",
                    help="family-pooled hierarchical log-gap prior for the ternary cutpoints "
                         "(fixes the partial-category PPC under-prediction; revision default)")
    ap.add_argument("--m3-init", default=None, choices=[None, "median"],
                    help="M3 guide initialisation ('median' = stabilised init_to_median)")
    args = ap.parse_args()

    mpath = HERE / args.matrix
    log(f"loading {args.matrix}")
    fd = io.load_field_data(mpath, abstain=args.abstain)
    # Independent per-respondent ability: pseudo-respondents contribute rows to ITEM
    # estimation but get their own (unused) theta, keeping the base-model leaderboard clean.
    # Grouping shrinks base theta toward pseudo-contaminated means and inverts the ranking.
    ug = False
    A = prep.field_arrays(fd)
    item_dim = prep.item_dim_array(fd, args.dim)
    fitkw = dict(n_steps=args.steps, n_seeds=args.seeds, n_post=200)
    log(f"data ready: N={len(fd.y):,} items={fd.n_item} resp={fd.n_resp} base={fd.n_group}")
    Path(HERE / args.save_fits).mkdir(parents=True, exist_ok=True)

    report = {"matrix": args.matrix, "abstain": args.abstain, "dim": args.dim,
              "hier_gap": bool(args.hier_gap), "m3_init": args.m3_init,
              "steps": args.steps, "seeds": args.seeds, "use_group": ug,
              "n_field_obs": int(len(fd.y)), "n_items": fd.n_item, "n_resp": fd.n_resp,
              "n_base_models": fd.n_group, "families": fd.family_names,
              "resp_names": fd.resp_names, "group_names": fd.group_names, "models": {}}
    t_all = time.time()

    # ---- M0: clip-level dichotomous 2PL, threshold sweep -------------------
    report["M0"] = {"thresholds": {}}
    m0_fits = {}
    for t in (0.5, 0.6, 0.7):
        cd = io.load_clip_data(mpath, threshold=t)
        r0 = fit.fit_svi(models.m0, prep.clip_args(cd), {"use_group": False},  # independent, clean
                         **fitkw)
        log(f"M0 t={t} done ({r0['seconds']}s, elbo={r0['loss']:.0f}, pass={cd.y.mean():.2f})")
        m0_fits[t] = (cd, r0)
        report["M0"]["thresholds"][str(t)] = {"clip_pass_rate": round(float(cd.y.mean()), 4),
                                              "final_elbo": round(r0["loss"], 1),
                                              "seconds": r0["seconds"]}
    headline_t = min((0.5, 0.6, 0.7), key=lambda t: abs(m0_fits[t][0].y.mean() - 0.5))
    cd_h, r0_h = m0_fits[headline_t]
    np.savez(HERE / args.save_fits / "m0.npz", theta=np.asarray(r0_h["derived"]["theta"]),
             a=np.asarray(r0_h["derived"]["a"]), b=np.asarray(r0_h["posterior"]["b"]),
             threshold=headline_t, resp_names=np.array(cd_h.resp_names))
    report["M0"]["headline_threshold"] = headline_t

    # ---- field models M1/M2/M3 --------------------------------------------
    from irtlib import harness as H, params as PMOD
    Yfull = H.response_table(fd)

    def run(name, model, margs, mkwargs):
        fk = dict(fitkw)
        if name == "M3_mirt" and args.m3_init:
            fk["init_values"] = args.m3_init
        r = fit.fit_svi(model, margs, mkwargs, **fk)
        log(f"{name} fit done ({r['seconds']}s, elbo={r['loss']:.0f}); computing WAIC...")
        ll = fit.pointwise_loglik(model, r["posterior"], margs, mkwargs)
        summ = diagnostics.elpd_summary(ll)
        log(f"{name} WAIC elpd={summ['elpd_waic']:.0f} bad_frac={summ['pareto_k_bad_frac']:.3f}")
        entry = {**summ, "final_elbo": round(r["loss"], 1), "fit_seconds": r["seconds"]}
        report["models"][name] = entry
        return r, ll

    def save_graded(name, r, item_dim=None, extra=None):
        """Build item params, marginalise the clip effect (u=0), refine abilities by
        frozen-item MLE, and save. theta is the refined per-respondent estimate (M,) or (M,D)."""
        p = PMOD.ItemParams.from_fit(r["derived"], fd, item_dim=item_dim)
        p.u = np.zeros_like(p.u)                      # marginalise clip effect for prediction
        theta_refined = H.refine_abilities(p, fd, Yfull)
        d = {"theta": theta_refined, "a": p.a, "b": p.b, "k0": p.k0, "k1": p.k1,
             "u": p.u, "theta_vi": np.asarray(r["derived"]["theta"]).mean(0)}
        if extra:
            d.update(extra)
        np.savez(HERE / args.save_fits / f"{name}.npz", **d,
                 item_family=fd.item_family, item_clip=fd.item_clip, item_keff=fd.item_keff,
                 resp_names=np.array(fd.resp_names), family_names=np.array(fd.family_names))
        return theta_refined

    def save_m1(r):
        np.savez(HERE / args.save_fits / "m1.npz",
                 theta=np.asarray(r["derived"]["theta"]).mean(0), a=np.asarray(r["derived"]["a"]),
                 b=np.asarray(r["derived"]["b"]), resp_names=np.array(fd.resp_names))

    # No clip effect: items are self-consistent at u=0, so the frozen-item ability refinement
    # is clean (a fitted clip effect absorbs clip baselines and, marginalised to 0, rails the
    # refined abilities to the grid boundary). Stable via the tighter cutpoint-gap prior.
    mk = {"use_group": ug, "hier_gap": bool(args.hier_gap)}
    mk1 = {"use_group": ug}
    r1, ll1 = run("M1_field_2pl", models.m1, prep.m1_args(A), mk1)
    save_m1(r1)                                      # save as we go so a later crash keeps M1/M2
    r2, ll2 = run("M2_graded", models.m2, prep.m2_args(A), mk)
    theta2_refined = save_graded("m2", r2)

    # M3 is the fragile one (multidimensional); if it won't stabilise, ship M2 (plan fallback).
    r3 = ll3 = None
    try:
        r3, ll3 = run("M3_mirt", models.m3, prep.m3_args(A, item_dim, args.dim), mk)
    except Exception as e:
        log(f"M3 FAILED ({type(e).__name__}: {e}); shipping M2 — paper stands on C1/C2/C4.")
        report["M3_status"] = f"failed: {type(e).__name__}"

    if r3 is not None:
        log("comparing M2 vs M3...")
        cmp = diagnostics.compare_models({"M2_graded": ll2, "M3_mirt": ll3})
        report["M2_vs_M3_compare"] = {
            "elpd_diff_M3_minus_M2": round(float(report["models"]["M3_mirt"]["elpd_waic"]
                                                 - report["models"]["M2_graded"]["elpd_waic"]), 1),
            "best_by_waic": cmp["best"], "compare_table": cmp["models"],
        }
    report["note_M1"] = ("M1 LOO is on a BINARIZED response coding, not comparable to M2/M3 "
                         "ordinal LOO; the C2 flat-vs-structured verdict is the downstream "
                         "reconstruction curve in EA.2, not this number.")

    # ---- PPC: fitted models reproduce observed category marginals ---------
    log("posterior-predictive checks...")
    ybin = np.asarray(fd.y)[np.where(fd.binary_mask)[0]]
    ytern = np.asarray(fd.y)[np.where(~fd.binary_mask)[0]]
    yblocks = {"y_bin": ybin, "y_tern": ytern}
    report["PPC_M2"] = diagnostics.ppc_category_marginals(r2["posterior"], models.m2,
                                                          prep.m2_args(A), mk, yblocks)
    Lc = None
    if r3 is not None:
        report["PPC_M3"] = diagnostics.ppc_category_marginals(r3["posterior"], models.m3,
                                                              prep.m3_args(A, item_dim, args.dim),
                                                              mk, yblocks)
        # M3 extras: correlation among ability dimensions
        Lc = np.asarray(r3["posterior"]["L_corr"]).mean(0)
        save_graded("m3", r3, item_dim=np.asarray(item_dim),
                    extra={"L_corr": Lc, "corr": Lc @ Lc.T, "item_dim": np.asarray(item_dim)})

    # base-model ability leaderboard from the REFINED (frozen-item) M2 abilities
    grp = {fd.resp_names[i]: round(float(theta2_refined[i]), 3)
           for i in range(fd.n_resp) if fd.resp_names[i] in fd.group_names}
    report["M2_base_ability"] = dict(sorted(grp.items(), key=lambda kv: -kv[1]))
    # validate against the true ordinal score ranking
    truesc = {fd.resp_names[i]: float((Yfull[i][Yfull[i] >= 0] /
              (fd.item_keff[Yfull[i] >= 0] - 1)).mean()) for i in range(fd.n_resp)
              if fd.resp_names[i] in fd.group_names}
    bn = list(grp)
    report["ability_vs_truescore_corr"] = round(float(np.corrcoef(
        [grp[n] for n in bn], [truesc[n] for n in bn])[0, 1]), 3)

    report["total_seconds"] = round(time.time() - t_all, 1)
    (HERE / args.out).write_text(json.dumps(report, indent=2))

    # ---- console summary ---------------------------------------------------
    print(f"\n[EA.1] matrix={args.matrix}  N_obs={len(fd.y):,}  items={fd.n_item}  "
          f"resp={fd.n_resp} (base={fd.n_group})  use_group={ug}")
    print(f"[EA.1] M0 headline threshold t={headline_t} "
          f"(pass rates {[report['M0']['thresholds'][str(t)]['clip_pass_rate'] for t in (0.5,0.6,0.7)]})")
    print("[EA.1] WAIC elpd (higher=better) + high-variance obs frac:")
    for m in ("M1_field_2pl", "M2_graded", "M3_mirt"):
        if m not in report["models"]:
            continue
        d = report["models"][m]
        cv = f"  elbo_cv={d['elbo_seed_cv']:.4f}" if "elbo_seed_cv" in d else ""
        print(f"   {m:14s} elpd_waic={d['elpd_waic']:>10.0f}  p_waic={d['p_waic']:>7.0f}  "
              f"bad_frac={d['pareto_k_bad_frac']:.3f}{cv}")
    if "M2_vs_M3_compare" in report:
        print(f"[EA.1] M3 - M2 elpd diff = {report['M2_vs_M3_compare']['elpd_diff_M3_minus_M2']:+.0f} "
              f"(positive => multidimensionality preferred; C3)")
        print(f"[EA.1] dimension correlation (M3):\n{np.round(Lc @ Lc.T, 2)}")
    else:
        print(f"[EA.1] M3 status: {report.get('M3_status', 'not run')} — C3 falls back to M2 (C1/C2/C4 hold)")
    print(f"[EA.1] M2 base-model ability (refined, top->bottom): "
          f"{list(report['M2_base_ability'].items())}")
    print(f"[EA.1] ability vs true-ordinal-score corr = {report['ability_vs_truescore_corr']} "
          f"(should be strongly positive; qwen/llama near top)")
    print(f"[EA.1] PPC M2 y_tern obs vs pred: {report['PPC_M2'].get('y_tern')}")
    print(f"[EA.1] total {report['total_seconds']}s  -> {args.out}")


if __name__ == "__main__":
    main()
