"""NUTS validation tier — the appendix answer to "does mean-field SVI distort the
posterior?" SVI is the production estimator (M2/M3 at ~10^5 obs are infeasible under
NUTS); here NUTS is run where it IS affordable — M0 on the full clip matrix, and M2 on a
small clip sub-matrix — and its posterior is compared to SVI's.

Reports Pearson correlation of base-model ability (theta) between SVI and NUTS, and the
family-level discrimination hyper-means. High agreement => the SVI posteriors the paper
relies on are not artefacts of the mean-field approximation.

Usage:
    python scripts/fit_nuts_check.py --sub-clips 200 --out results/nuts_check.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, prep

_T0 = time.time()
def log(m): print(f"[NUTS +{time.time()-_T0:6.1f}s] {m}", flush=True)


def base_theta(theta_draws, resp_names, group_names):
    """Mean ability restricted to the base (leaderboard) respondents, name->value."""
    tm = np.asarray(theta_draws).mean(0)
    return {resp_names[i]: float(tm[i]) for i in range(len(resp_names))
            if resp_names[i] in set(group_names)}


def aligned_corr(a, b):
    keys = sorted(set(a) & set(b))
    va, vb = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    return float(np.corrcoef(va, vb)[0, 1]), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--sub-clips", type=int, default=200)
    ap.add_argument("--svi-steps", type=int, default=8000)
    ap.add_argument("--nuts-warmup", type=int, default=300)
    ap.add_argument("--nuts-samples", type=int, default=300)
    ap.add_argument("--out", default="results/nuts_check.json")
    args = ap.parse_args()
    mpath = HERE / args.matrix
    rep = {"matrix": args.matrix, "sub_clips": args.sub_clips,
           "nuts_warmup": args.nuts_warmup, "nuts_samples": args.nuts_samples}

    # ---- M0 full: SVI vs NUTS (M0 is small enough for NUTS on the full matrix) ----
    cd = io.load_clip_data(mpath, threshold=0.6)
    ug0 = False   # final config: independent abilities
    log(f"M0 full: {cd.n_clip} clips, {len(cd.y)} obs — SVI")
    s0 = fit.fit_svi(models.m0, prep.clip_args(cd), {"use_group": ug0},
                     n_steps=args.svi_steps, n_seeds=2, n_post=400)
    log(f"M0 SVI done ({s0['seconds']}s) — NUTS")
    n0 = fit.fit_nuts(models.m0, prep.clip_args(cd), {"use_group": ug0},
                      n_warmup=args.nuts_warmup, n_samples=args.nuts_samples)
    log(f"M0 NUTS done ({n0['seconds']}s)")
    svi_t = base_theta(s0["derived"]["theta"], cd.resp_names, [n for n in cd.resp_names])  # M0 groups==base for base-only; use resp
    # M0 theta site name is "theta"
    svi_theta0 = {cd.resp_names[i]: float(np.asarray(s0["derived"]["theta"]).mean(0)[i]) for i in range(cd.n_resp)}
    nuts_theta0 = {cd.resp_names[i]: float(np.asarray(n0["samples"].get("theta", n0["samples"].get("z_theta"))).mean(0)[i]) for i in range(cd.n_resp)}
    r0, keys0 = aligned_corr(svi_theta0, nuts_theta0)
    rep["M0"] = {"n_clips": cd.n_clip, "svi_seconds": s0["seconds"], "nuts_seconds": n0["seconds"],
                 "theta_corr_svi_vs_nuts": round(r0, 4), "n_respondents": len(keys0)}
    log(f"M0 theta corr(SVI,NUTS) = {r0:.4f}")

    # ---- M2 on a small clip sub-matrix: SVI vs NUTS ----
    df = pd.read_parquet(mpath)
    keep = df["clip_id"].drop_duplicates().head(args.sub_clips)
    subp = HERE / "data/_nuts_sub.parquet"
    df[df.clip_id.isin(keep)].to_parquet(subp, index=False)
    fd = io.load_field_data(subp)
    ug = False    # final config: independent abilities
    A = prep.field_arrays(fd)
    log(f"M2 sub: {fd.n_clip} clips, {len(fd.y)} obs, {fd.n_item} items — SVI")
    s2 = fit.fit_svi(models.m2, prep.m2_args(A), {"use_group": ug},
                     n_steps=args.svi_steps, n_seeds=2, n_post=400)
    log(f"M2 SVI done ({s2['seconds']}s) — NUTS")
    n2 = fit.fit_nuts(models.m2, prep.m2_args(A), {"use_group": ug},
                      n_warmup=args.nuts_warmup, n_samples=args.nuts_samples)
    log(f"M2 NUTS done ({n2['seconds']}s)")
    svi_t2 = base_theta(s2["derived"]["theta"], fd.resp_names, fd.group_names)
    nuts_t2 = base_theta(n2["samples"].get("theta", n2["samples"].get("z_theta")), fd.resp_names, fd.group_names)
    r2, keys2 = aligned_corr(svi_t2, nuts_t2)
    # family discrimination hyper-means
    svi_mu = np.asarray(s2["posterior"]["mu_log_a"]).mean(0).tolist()
    nuts_mu = np.asarray(n2["samples"]["mu_log_a"]).mean(0).tolist()
    rep["M2_sub"] = {"n_clips": fd.n_clip, "n_obs": int(len(fd.y)), "n_items": fd.n_item,
                     "svi_seconds": s2["seconds"], "nuts_seconds": n2["seconds"],
                     "theta_corr_svi_vs_nuts": round(r2, 4), "n_base_models": len(keys2),
                     "families": fd.family_names,
                     "mu_log_a_svi": [round(x, 3) for x in svi_mu],
                     "mu_log_a_nuts": [round(x, 3) for x in nuts_mu]}
    log(f"M2 theta corr(SVI,NUTS) = {r2:.4f}")
    subp.unlink(missing_ok=True)

    rep["verdict"] = ("SVI reproduces NUTS ability ordering" if min(r0, r2) > 0.9
                      else "check disagreement — SVI may distort at this scale")
    rep["total_seconds"] = round(time.time() - _T0, 1)
    (HERE / args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n[NUTS] M0 theta corr={r0:.3f}  M2 theta corr={r2:.3f}  -> {rep['verdict']}")
    print(f"[NUTS] wrote {args.out}  ({rep['total_seconds']}s)")


if __name__ == "__main__":
    main()
