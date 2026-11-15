"""Subset selection over CLIPS (SteelBench-mini is a set of clips).

Fisher information is additive over independent items, so a subset's total information is
the sum of its clips' informations — the objective is MODULAR, and greedy top-information
selection under per-stratum FLOOR quotas is exact (no exchange step needed for the additive
objective). Baselines: uniform random, stratified random, k-medoids on clip item-parameter
vectors (the tinyBenchmarks anchor-clustering comparator).
"""
import numpy as np
from . import params as P


# --------------------------------------------------------------------------- information
def clip_information(p: P.ItemParams, base_thetas):
    """Per-clip total + per-dimension Fisher information, averaged over the base models.

    base_thetas: (M,) for 1-D fits or (M, n_dim) for M3. Returns (clip_info (C,),
    clip_info_by_dim (C, n_dim)).
    """
    base_thetas = np.atleast_1d(base_thetas)
    J = len(p.a)
    C = int(p.clip.max()) + 1
    D = p.n_dim
    item_info = np.zeros(J)
    for th in base_thetas:
        item_info += P.item_information(p, P.theta_for_items(p, th))
    item_info /= len(base_thetas)
    clip_info = np.bincount(p.clip, weights=item_info, minlength=C)
    by_dim = np.zeros((C, D))
    if p.item_dim is not None and D > 1:
        for d in range(D):
            m = p.item_dim == d
            by_dim[:, d] = np.bincount(p.clip[m], weights=item_info[m], minlength=C)
    else:
        by_dim[:, 0] = clip_info
    return clip_info, by_dim


# --------------------------------------------------------------------------- quota helper
def stratum_floors(strata, n, min_k=3):
    """Per-stratum floor counts summing to <= n (proportional, min_k each, capped by size)."""
    uniq, counts = np.unique(strata, return_counts=True)
    shares = counts / counts.sum()
    floors = {s: min(int(c), max(min_k, int(np.round(sh * n))))
              for s, c, sh in zip(uniq, counts, shares)}
    total = sum(floors.values())
    if total > n:                                  # scale down proportionally, keep >=1
        scale = n / total
        floors = {s: max(1, int(np.floor(f * scale))) for s, f in floors.items()}
    return floors


# --------------------------------------------------------------------------- selectors
def greedy_stratified(clip_info, strata, n, min_k=3):
    """Floors satisfied by top-info per stratum, remainder filled by top-info globally."""
    strata = np.asarray(strata)
    order = np.argsort(-clip_info)                  # high info first
    floors = stratum_floors(strata, n, min_k)
    selected, per_stratum = [], {s: 0 for s in floors}
    # pass 1: satisfy floors
    for c in order:
        s = strata[c]
        if per_stratum.get(s, 0) < floors.get(s, 0):
            selected.append(c); per_stratum[s] += 1
    # pass 2: fill remainder by global info
    sel = set(selected)
    for c in order:
        if len(selected) >= n:
            break
        if c not in sel:
            selected.append(c); sel.add(c)
    return np.array(selected[:n])


def select_random(n_clips, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(n_clips, size=n, replace=False)


def select_stratified_random(strata, n, seed=0, min_k=3):
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    floors = stratum_floors(strata, n, min_k)
    selected = []
    for s, f in floors.items():
        idx = np.where(strata == s)[0]
        selected += list(rng.choice(idx, size=min(f, len(idx)), replace=False))
    remaining = np.setdiff1d(np.arange(len(strata)), selected)
    if len(selected) < n and len(remaining):
        selected += list(rng.choice(remaining, size=min(n - len(selected), len(remaining)),
                                    replace=False))
    return np.array(selected[:n])


def clip_feature_matrix(p: P.ItemParams):
    """Per-clip item-parameter summary for k-medoids (mean a/b/k0/k1, ternary frac, size)."""
    C = int(p.clip.max()) + 1
    def agg(w):
        s = np.bincount(p.clip, weights=w, minlength=C)
        n = np.bincount(p.clip, minlength=C).clip(1)
        return s / n
    feats = np.stack([agg(p.a), agg(p.b), agg(p.k0), agg(p.k1),
                      agg((p.keff == 3).astype(float)),
                      np.bincount(p.clip, minlength=C).astype(float)], axis=1)
    # standardise
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    return (feats - mu) / sd


def select_kmedoids(p: P.ItemParams, n, seed=0):
    """Anchor-clustering baseline: KMeans on clip features, take the clip nearest each center."""
    from sklearn.cluster import KMeans
    X = clip_feature_matrix(p)
    km = KMeans(n_clusters=n, random_state=seed, n_init=4).fit(X)
    medoids = []
    for c in range(n):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1)
        medoids.append(members[np.argmin(d)])
    return np.array(medoids)


def build_strata(strata_df, cols=("visibility_primary", "layer", "freq_band")):
    """Compose a stratum label per clip from the clip_strata_v1.csv rows (clip-ordered).

    fillna BEFORE astype: a column of NaN floats (e.g. calibration_condition for
    non-calibration clips) must become the literal stratum "none", not crash the join.
    """
    df = strata_df[list(cols)].fillna("none").astype(str)
    return df.agg("|".join, axis=1).to_numpy()


def greedy_dim_balanced(clip_info_by_dim, strata, n, min_k=3, weights=None):
    """Dimension-balanced information selection under stratum floors.

    Pass 1 satisfies the per-stratum floor quotas by TOTAL information (as
    greedy_stratified). Pass 2 fills the remaining budget round-robin over ability
    dimensions: each turn takes the not-yet-selected clip with the highest information for
    that dimension. `weights` (D,) sets how many turns each dimension gets per cycle
    (default equal). This guarantees every capability dimension receives a share of
    high-information clips instead of letting the largest family dominate the total.
    """
    by_dim = np.asarray(clip_info_by_dim, float)
    C, D = by_dim.shape
    strata = np.asarray(strata)
    total = by_dim.sum(1)
    order = np.argsort(-total)
    floors = stratum_floors(strata, n, min_k)
    selected, per_stratum = [], {s: 0 for s in floors}
    for c in order:
        s = strata[c]
        if per_stratum.get(s, 0) < floors.get(s, 0):
            selected.append(int(c)); per_stratum[s] += 1
    sel = set(selected)
    orders = [np.argsort(-by_dim[:, d]) for d in range(D)]
    ptr = [0] * D
    w = np.ones(D, int) if weights is None else np.asarray(weights, int)
    d = 0
    while len(selected) < n:
        for _ in range(int(w[d])):
            while ptr[d] < C and int(orders[d][ptr[d]]) in sel:
                ptr[d] += 1
            if ptr[d] < C and len(selected) < n:
                c = int(orders[d][ptr[d]]); selected.append(c); sel.add(c); ptr[d] += 1
        d = (d + 1) % D
        if all(pt >= C for pt in ptr):
            break
    return np.array(selected[:n])
