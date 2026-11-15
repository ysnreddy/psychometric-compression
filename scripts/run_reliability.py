"""Reliability-before-validity check for the non-transfer (multidimensionality) claim.

Zero cross-dimension agreement between capability rankings is only evidence of TRUE
multidimensionality if each dimension ranks the models reliably in the first place. This
script bootstraps CLIPS (the sampling unit of the benchmark) and asks:

  1. within-dimension reliability: how well does a resampled sub-score ranking agree with
     the full-data ranking (tau), and how well do two independent resamples agree with
     each other (split-half tau, Spearman-Brown corrected Pearson)?
  2. cross-dimension agreement: tau between dimensions on the full data and its bootstrap
     distribution; number of the 108 cross-dimension model-pair orderings that swap, the
     expected count under independent rankings (54), and how many observed swaps are
     STABLE (hold in >= 95% of resamples).
  3. per-model bootstrap SE of every sub-score and of the aggregate, and how many adjacent
     model pairs are separated by more than 2 SE.
  4. same for the aggregate ranking (the near-tie ceiling).

Model-free: uses only observed graded responses (score map correct=1, partial=0.5, wrong=0).

Usage:
    python scripts/run_reliability.py --boot 2000
"""
import argparse, json, sys, time
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import kendalltau, pearsonr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, harness as H, prep

_T0 = time.time()
def log(m): print(f"[REL +{time.time()-_T0:6.1f}s] {m}", flush=True)

DIMS = ["recognition", "perception_detail", "safety_reasoning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/reliability.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    keff = np.asarray(fd.item_keff); item_clip = np.asarray(fd.item_clip)
    dim_of_item = np.array(prep.item_dim_array(fd, 3))
    base_idx = H.base_indices(fd)
    names = [fd.resp_names[i] for i in base_idx]
    M, C = len(base_idx), fd.n_clip
    log(f"models={M} clips={C} items={fd.n_item}")

    # per (model, dim, clip): score sum and count; dim index 3 = aggregate (all items)
    ssum = np.zeros((M, 4, C)); cnt = np.zeros((M, 4, C))
    for r, i in enumerate(base_idx):
        yi = Y[i]; pres = yi >= 0
        sc = yi[pres] / (keff[pres] - 1)
        cl = item_clip[pres]; dm = dim_of_item[pres]
        for d in range(3):
            m = dm == d
            ssum[r, d] = np.bincount(cl[m], weights=sc[m], minlength=C)
            cnt[r, d] = np.bincount(cl[m], minlength=C)
        ssum[r, 3] = np.bincount(cl, weights=sc, minlength=C)
        cnt[r, 3] = np.bincount(cl, minlength=C)

    def scores(w):
        """(M,4) item-pooled scores under clip weights w (C,)."""
        return (ssum * w).sum(-1) / np.maximum((cnt * w).sum(-1), 1e-9)

    full = scores(np.ones(C))
    rng = np.random.default_rng(args.seed)
    B = args.boot
    boot = np.zeros((B, M, 4))
    for b in range(B):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        boot[b] = scores(w)
    log("bootstrap done")

    def tau(a, b):
        return float(kendalltau(a, b).statistic)

    pairs = list(combinations(range(M), 2))
    labels = DIMS + ["aggregate"]
    out = {"n_models": M, "n_clips": C, "n_boot": B, "models": names,
           "full_scores": {labels[k]: {names[r]: round(float(full[r, k]), 4) for r in range(M)} for k in range(4)},
           "bootstrap_se": {labels[k]: {names[r]: round(float(boot[:, r, k].std()), 4) for r in range(M)} for k in range(4)},
           "within_dimension": {}, "cross_dimension": {}, "reversals": {}}

    # ---- within-dimension reliability ------------------------------------------------
    half_seeds = np.random.default_rng(args.seed + 1)
    for k, lab in enumerate(labels):
        t_vs_full = np.array([tau(boot[b, :, k], full[:, k]) for b in range(B)])
        # boot-vs-boot (independent resamples)
        t_bb = np.array([tau(boot[b, :, k], boot[(b + B // 2) % B, :, k]) for b in range(B // 2)])
        # split-half over clips
        sh_tau, sh_r = [], []
        for _ in range(500):
            perm = half_seeds.permutation(C)
            w1 = np.zeros(C); w1[perm[: C // 2]] = 1.0
            w2 = 1.0 - w1
            s1, s2 = scores(w1)[:, k], scores(w2)[:, k]
            sh_tau.append(tau(s1, s2)); sh_r.append(pearsonr(s1, s2)[0])
        r = float(np.mean(sh_r)); sb = 2 * r / (1 + r) if r > -1 else float("nan")
        # adjacent-pair separation in SE units
        order = np.argsort(-full[:, k]); se = boot[:, :, k].std(0)
        gaps = []
        for a, b_ in zip(order[:-1], order[1:]):
            gaps.append(float((full[a, k] - full[b_, k]) / np.sqrt(se[a] ** 2 + se[b_] ** 2)))
        # pairwise: fraction of the 36 model pairs whose ordering is stable (>=95% of resamples)
        stable = 0
        for a, b_ in pairs:
            frac = float((np.sign(boot[:, a, k] - boot[:, b_, k]) == np.sign(full[a, k] - full[b_, k])).mean())
            stable += int(frac >= 0.95)
        out["within_dimension"][lab] = {
            "tau_boot_vs_full_mean": float(t_vs_full.mean()),
            "tau_boot_vs_full_q05_q95": [float(np.quantile(t_vs_full, .05)), float(np.quantile(t_vs_full, .95))],
            "tau_boot_vs_boot_mean": float(t_bb.mean()),
            "split_half_tau_mean": float(np.mean(sh_tau)),
            "split_half_pearson_mean": r, "spearman_brown_reliability": float(sb),
            "score_range_pp": float((full[:, k].max() - full[:, k].min()) * 100),
            "median_se_pp": float(np.median(se) * 100),
            "adjacent_gap_over_se": [round(g, 2) for g in gaps],
            "n_adjacent_pairs_gt_2se": int(sum(g > 2 for g in gaps)),
            "n_pairs_stable_95": stable, "n_pairs": len(pairs),
        }
        log(f"{lab:18s} tau(boot,full)={t_vs_full.mean():.2f} split-half tau={np.mean(sh_tau):.2f} "
            f"SB={sb:.2f} stable pairs={stable}/{len(pairs)}")

    # ---- cross-dimension ------------------------------------------------------------
    # disattenuation: r_xy / sqrt(rel_x rel_y) with rel = Spearman-Brown split-half reliability
    rel = {k: out["within_dimension"][k]["spearman_brown_reliability"] for k in labels}
    for (i, j) in combinations(range(3), 2):
        t_full = tau(full[:, i], full[:, j])
        t_b = np.array([tau(boot[b, :, i], boot[b, :, j]) for b in range(B)])
        r_raw = float(pearsonr(full[:, i], full[:, j])[0])
        r_b = np.array([pearsonr(boot[b, :, i], boot[b, :, j])[0] for b in range(B)])
        denom = np.sqrt(max(rel[DIMS[i]], 1e-6) * max(rel[DIMS[j]], 1e-6))
        out["cross_dimension"][f"{DIMS[i]}__{DIMS[j]}"] = {
            "tau_full": t_full, "tau_boot_mean": float(t_b.mean()),
            "tau_boot_ci95": [float(np.quantile(t_b, .025)), float(np.quantile(t_b, .975))],
            "pearson_full": r_raw, "pearson_boot_ci95": [float(np.quantile(r_b, .025)), float(np.quantile(r_b, .975))],
            "pearson_disattenuated": float(r_raw / denom),
            "reliabilities_used": [rel[DIMS[i]], rel[DIMS[j]]],
        }
        log(f"cross {DIMS[i]} vs {DIMS[j]}: tau={t_full:+.2f} boot CI {out['cross_dimension'][f'{DIMS[i]}__{DIMS[j]}']['tau_boot_ci95']}")

    # ---- reversals: 36 pairs x 3 dimension-pairs = 108 orderings ---------------------
    obs_swaps, stable_swaps, swap_details = 0, 0, []
    boot_swaps = np.zeros(B)
    for (i, j) in combinations(range(3), 2):
        for a, b_ in pairs:
            s_full_i = np.sign(full[a, i] - full[b_, i]); s_full_j = np.sign(full[a, j] - full[b_, j])
            swapped = (s_full_i != 0) and (s_full_j != 0) and (s_full_i != s_full_j)
            si = np.sign(boot[:, a, i] - boot[:, b_, i]); sj = np.sign(boot[:, a, j] - boot[:, b_, j])
            bs = (si != 0) & (sj != 0) & (si != sj)
            boot_swaps += bs
            frac = float(bs.mean())
            if swapped:
                obs_swaps += 1
                stable_swaps += int(frac >= 0.95)
                swap_details.append({"pair": [names[a], names[b_]], "dims": [DIMS[i], DIMS[j]],
                                     "gap_pp": [round(float(abs(full[a, i] - full[b_, i])) * 100, 2),
                                                round(float(abs(full[a, j] - full[b_, j])) * 100, 2)],
                                     "frac_resamples_swapped": round(frac, 3)})
    swap_details.sort(key=lambda d: -d["frac_resamples_swapped"])
    # expected swaps under independent rankings: each of the 108 orderings swaps w.p. 1/2
    out["reversals"] = {
        "n_orderings": 3 * len(pairs), "observed_swaps": obs_swaps,
        "expected_under_independence": 1.5 * len(pairs),
        "stable_swaps_95": stable_swaps,
        "stable_swaps_80": int(sum(d["frac_resamples_swapped"] >= 0.80 for d in swap_details)),
        "boot_swaps_mean": float(boot_swaps.mean()),
        "boot_swaps_ci95": [float(np.quantile(boot_swaps, .025)), float(np.quantile(boot_swaps, .975))],
        "details": swap_details,
    }
    log(f"reversals: observed={obs_swaps}/108, stable(>=95%)={stable_swaps}, "
        f">=80%={out['reversals']['stable_swaps_80']}, boot mean={boot_swaps.mean():.1f}")

    # ---- a null: how many swaps would two RELIABLE-but-independent rankings show?
    # (permutation null: shuffle model labels within one dimension)
    rng2 = np.random.default_rng(args.seed + 2)
    null = []
    for _ in range(2000):
        cnt_ = 0
        for (i, j) in combinations(range(3), 2):
            perm = rng2.permutation(M)
            for a, b_ in pairs:
                si = np.sign(full[a, i] - full[b_, i]); sj = np.sign(full[perm[a], j] - full[perm[b_], j])
                cnt_ += int(si != 0 and sj != 0 and si != sj)
        null.append(cnt_)
    out["reversals"]["permutation_null_mean"] = float(np.mean(null))
    out["reversals"]["permutation_null_ci95"] = [float(np.quantile(null, .025)), float(np.quantile(null, .975))]

    (HERE / args.out).write_text(json.dumps(out, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
