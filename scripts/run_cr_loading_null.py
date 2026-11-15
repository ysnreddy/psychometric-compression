"""Camera-ready (reviewer D4yq #1): is the multidimensionality claim an artefact of the
confirmatory loading structure? Model-free tests on the raw response matrix, no IRT fit.

  1. Family-level rankings with NO grouping: per-family raw scores of the nine base models
     and the Kendall tau between every pair of families. If families already disagree on
     model order, any grouping of them into dimensions produces cross-dimension swaps.
  2. All 25 partitions of the 5 field families into 3 non-empty dimensions: swap count
     for each. Shows where the paper's confirmatory partition sits among the alternatives.
  3. Item-level permutation null: dimension labels shuffled across items (sizes kept),
     2,000 draws. If the 56/108 swaps came from grouping alone, random groupings would
     match it.
  4. Clip bootstrap of the paper's swap count (1,000 resamples of clips) for a CI.

Usage:
    python scripts/run_cr_loading_null.py
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from irtlib import io, harness as H, prep


def scores_by_label(Y, rows, keff, label, n_label):
    """(len(rows), n_label) mean ordinal score per label group."""
    out = np.zeros((len(rows), n_label))
    for r, i in enumerate(rows):
        yi = Y[i]; pres = yi >= 0
        sc = yi[pres] / (keff[pres] - 1)
        s = np.bincount(label[pres], weights=sc, minlength=n_label)
        c = np.bincount(label[pres], minlength=n_label)
        out[r] = s / np.maximum(c, 1)
    return out


def swap_count(S):
    """Cross-dimension model-pair order swaps; S is (models, dims). Ties count as no swap."""
    M, D = S.shape
    n = tot = 0
    for a, b in itertools.combinations(range(M), 2):
        for d1, d2 in itertools.combinations(range(D), 2):
            tot += 1
            if np.sign(S[a, d1] - S[b, d1]) * np.sign(S[a, d2] - S[b, d2]) < 0:
                n += 1
    return n, tot


def partitions_into_3(items):
    """All set partitions of `items` into exactly 3 non-empty blocks (Stirling S(5,3)=25)."""
    items = list(items)
    seen = set()
    for lab in itertools.product(range(3), repeat=len(items)):
        if len(set(lab)) < 3:
            continue
        # canonical relabel: order blocks by first appearance
        m, canon = {}, []
        for l in lab:
            m.setdefault(l, len(m)); canon.append(m[l])
        canon = tuple(canon)
        if canon not in seen:
            seen.add(canon)
            yield dict(zip(items, canon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/response_matrix_v1.parquet")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/camera_ready/loading_null.json")
    args = ap.parse_args()

    fd = io.load_field_data(HERE / args.matrix)
    Y = H.response_table(fd)
    keff = np.asarray(fd.item_keff)
    fam = np.asarray(fd.item_family)
    fams = list(fd.family_names)
    base = H.base_indices(fd)
    names = [fd.resp_names[i] for i in base]
    dim_paper = np.asarray(prep.item_dim_array(fd, 3))
    paper_map = prep.FAM_DIM[3]

    # 1. family-level rankings, no grouping
    F = scores_by_label(Y, base, keff, fam, len(fams))
    fam_tau = {f"{fams[i]}__{fams[j]}": float(kendalltau(F[:, i], F[:, j]).statistic)
               for i, j in itertools.combinations(range(len(fams)), 2)}
    fam_swaps, fam_tot = swap_count(F)

    # 2. all 3-block partitions of the 5 families
    part_rows = []
    for pm in partitions_into_3(fams):
        lab = np.array([pm[fams[f]] for f in fam])
        n, tot = swap_count(scores_by_label(Y, base, keff, lab, 3))
        blocks = sorted(["+".join(sorted(f for f in fams if pm[f] == b)) for b in range(3)])
        part_rows.append({"blocks": blocks, "swaps": n, "total": tot})
    paper_blocks = sorted(["+".join(sorted(f for f in fams if paper_map[f] == b)) for b in range(3)])
    part_rows.sort(key=lambda r: -r["swaps"])
    paper_rank = 1 + [r["blocks"] for r in part_rows].index(paper_blocks)

    # paper partition
    S_paper = scores_by_label(Y, base, keff, dim_paper, 3)
    paper_swaps, tot = swap_count(S_paper)

    # 3. item-level permutation null
    rng = np.random.default_rng(0)
    null = np.array([swap_count(scores_by_label(Y, base, keff, rng.permutation(dim_paper), 3))[0]
                     for _ in range(args.n_perm)])

    # 4. clip bootstrap of the paper's swap count
    clip = np.asarray(fd.item_clip)
    item_by_clip = [np.where(clip == c)[0] for c in range(fd.n_clip)]
    boot = []
    for _ in range(args.n_boot):
        cs = rng.integers(0, fd.n_clip, fd.n_clip)
        idx = np.concatenate([item_by_clip[c] for c in cs])
        Yb = Y[:, idx]
        boot.append(swap_count(scores_by_label(Yb, base, keff[idx], dim_paper[idx], 3))[0])
    boot = np.array(boot)

    report = {
        "matrix": args.matrix, "models": names, "families": fams,
        "paper_partition": paper_blocks,
        "paper_swaps": paper_swaps, "total_pairs": tot,
        "family_level": {"scores": {n: dict(zip(fams, map(float, F[k]))) for k, n in enumerate(names)},
                         "kendall_tau_between_families": fam_tau,
                         "swaps": fam_swaps, "total_pairs": fam_tot},
        "partitions": {"n": len(part_rows), "paper_rank_by_swaps": paper_rank,
                       "swaps_min": min(r["swaps"] for r in part_rows),
                       "swaps_median": float(np.median([r["swaps"] for r in part_rows])),
                       "swaps_max": max(r["swaps"] for r in part_rows), "rows": part_rows},
        "item_permutation_null": {"n_perm": args.n_perm, "mean": float(null.mean()),
                                  "q95": float(np.quantile(null, 0.95)), "max": int(null.max()),
                                  "p_value_ge_paper": float((1 + (null >= paper_swaps).sum()) / (1 + len(null)))},
        "clip_bootstrap": {"n_boot": args.n_boot, "mean": float(boot.mean()),
                           "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]},
    }
    (HERE / args.out).parent.mkdir(parents=True, exist_ok=True)
    (HERE / args.out).write_text(json.dumps(report, indent=2))
    print(f"paper partition {paper_blocks}: {paper_swaps}/{tot} swaps")
    print(f"family-level (no grouping): {fam_swaps}/{fam_tot} swaps; tau between families:")
    for k, v in fam_tau.items():
        print(f"  {k:24s} {v:+.2f}")
    print(f"25 partitions: paper ranks {paper_rank}/{len(part_rows)}; swaps min/median/max "
          f"{report['partitions']['swaps_min']}/{report['partitions']['swaps_median']}/{report['partitions']['swaps_max']}")
    print(f"item-permutation null: mean {null.mean():.1f}, q95 {np.quantile(null, .95):.0f}, max {null.max()}, "
          f"p={report['item_permutation_null']['p_value_ge_paper']:.4f}")
    print(f"clip bootstrap: mean {boot.mean():.1f}, 95% CI {report['clip_bootstrap']['ci95']}")


if __name__ == "__main__":
    main()
