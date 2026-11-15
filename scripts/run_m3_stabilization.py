"""M3 (confirmatory MIRT) SVI stabilisation on the 8-model LOMO rotations.

The paper's LOMO table uses M2 to drive selection because M3 "occasionally diverges on
eight-model pools" (the gemma4_31b rotation: all seeds non-finite after 3 lr back-offs).
This script tries the standard remedies on the failing rotation(s) and reports, per config,
whether the fit is finite, its final ELBO, and its cost:

  baseline        AutoNormal, lr 0.01, init_scale 0.1 (paper config, no back-off)
  lowlr           lr 0.005 with exponential decay to 10%
  warm            AutoNormal initialised at the M2 rotation fit's posterior medians
                  (shared item sites; per-dimension abilities tiled from M2's theta;
                  identity correlation)
  warm_decay      warm start + lr decay
  warm_lowrank    warm start + low-rank multivariate-normal guide (rank 8)

Usage:
    python scripts/run_m3_stabilization.py --holdouts gemma4_31b,gemini25_pro --steps 3000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, models, fit, prep, harness as H

_T0 = time.time()
def log(m): print(f"[STAB +{time.time()-_T0:6.1f}s] {m}", flush=True)

SHARED = ["mu_log_a", "sigma_log_a", "z_log_a", "mu_b", "sigma_b", "z_b",
          "mu_k0", "sigma_k0", "z_k0", "k_gap"]


def m2_to_m3_init(m2_median, D):
    init = {k: jnp.asarray(m2_median[k]) for k in SHARED if k in m2_median}
    zt = np.asarray(m2_median["z_theta"])                       # (M,)
    init["z_theta"] = jnp.asarray(np.tile(zt[:, None], (1, D)))
    init["L_corr"] = jnp.eye(D)
    return init


def first_nonfinite(losses):
    bad = np.where(~np.isfinite(np.asarray(losses)))[0]
    return int(bad[0]) if len(bad) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--holdouts", default="gemma4_31b")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--configs", default="baseline,median_init,lowlr,warm,warm_decay,warm_lowrank")
    ap.add_argument("--out", default="results/m3_stabilization.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    item_dim = prep.item_dim_array(fd, 3)
    base_idx = H.base_indices(fd)
    holdouts = args.holdouts.split(",")
    if holdouts == ["all"]:
        holdouts = [fd.resp_names[i] for i in base_idx]
    report = {"steps": args.steps, "seeds": args.seeds, "rotations": {}}

    for h in holdouts:
        excl = [i for i, nm in enumerate(fd.resp_names) if nm == h or nm.startswith(h + "__")]
        A = prep.rotation_field_arrays(fd, excl)
        m3a = prep.m3_args(A, item_dim, 3)
        log(f"rotation holdout={h}: fitting M2 for warm start")
        r2 = fit.fit_svi(models.m2, prep.m2_args(A), {"use_group": False},
                         n_steps=args.steps, n_seeds=args.seeds, n_post=50)
        warm = m2_to_m3_init(r2["median"], 3)
        rot = {"m2_elbo": r2["loss"], "m2_seconds": r2["seconds"], "configs": {}}
        cfgs = {
            "baseline": dict(),
            "median_init": dict(init_values="median"),
            "lowlr": dict(lr=0.005, lr_decay=0.1),
            "warm": dict(init_values=warm),
            "warm_decay": dict(init_values=warm, lr_decay=0.1),
            "warm_lowrank": dict(init_values=warm, guide_kind="lowrank", lowrank=8),
        }
        for cname in args.configs.split(","):
            kw = cfgs[cname]
            t0 = time.time()
            try:
                r = fit.fit_svi(models.m3, m3a, {"use_group": False}, n_steps=args.steps,
                                n_seeds=args.seeds, n_post=50, max_backoff=0, **kw)
                # per-seed diagnostics are hidden inside fit_svi; re-run the losing seed
                # cheaply is unnecessary — record the winner
                entry = {"finite": True, "final_elbo": r["loss"], "seed": r["seed"], "lr": r["lr"],
                         "seconds": round(time.time() - t0, 1),
                         "first_nonfinite_step": first_nonfinite(r["losses"]),
                         "loss_at_10pct": float(np.asarray(r["losses"])[args.steps // 10]),
                         "loss_at_50pct": float(np.asarray(r["losses"])[args.steps // 2])}
                L = np.asarray(r["posterior"]["L_corr"]).mean(0); om = L @ L.T
                dd = np.sqrt(np.diag(om)); entry["omega"] = (om / np.outer(dd, dd)).round(3).tolist()
            except RuntimeError as e:
                entry = {"finite": False, "error": str(e)[:120], "seconds": round(time.time() - t0, 1)}
            rot["configs"][cname] = entry
            log(f"  {cname:14s} " + (f"ELBO={entry['final_elbo']:.0f} ({entry['seconds']}s) omega={entry.get('omega')}"
                                     if entry["finite"] else f"DIVERGED ({entry['seconds']}s)"))
        report["rotations"][h] = rot

    (HERE / args.out).write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
