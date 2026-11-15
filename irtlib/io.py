"""Benchmark-agnostic loading of a long-format response matrix into IRT arrays.

Consumes any table with columns (respondent, group_id[=clip], item_id, family,
response, n_categories, structural_miss, base_model). Column names are configurable;
defaults match scripts/build_response_matrix.py output.

Ordinal coding: raw graded_response is 3=correct / 2=partial / 1=wrong / 0=abstain.
For the GRM likelihood we map to a contiguous ordinal ignoring abstain:
  K_eff = 3 if the family can produce partial (raw n_categories==4) else 2.
  correct(3) -> K_eff-1 ; partial(2) -> 1 ; wrong(1) -> 0 ; abstain(0) -> 0 (merge) or dropped.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FieldData:
    """Field-level arrays for M1/M2/M3. Indices are 0-based contiguous."""
    resp: np.ndarray          # (N,) respondent index
    item: np.ndarray          # (N,) item index
    clip: np.ndarray          # (N,) clip index (per observation, via its item)
    y: np.ndarray             # (N,) ordinal response in {0..K_eff-1}
    keff: np.ndarray          # (N,) effective #categories for the obs's item (2 or 3)
    item_family: np.ndarray   # (J,) family index per item
    item_clip: np.ndarray     # (J,) clip index per item
    item_keff: np.ndarray     # (J,) K_eff per item
    resp_group: np.ndarray    # (M,) base-model group index per respondent
    n_resp: int
    n_item: int
    n_clip: int
    n_family: int
    n_group: int
    family_names: list
    resp_names: list
    group_names: list
    clip_names: list          # (n_clip,) clip_id per clip index — for aligning strata
    binary_mask: np.ndarray   # (N,) bool: obs whose item is binary (K_eff==2)


def _codes(series):
    """Factorize to contiguous ints; return (codes, uniques list)."""
    cat = pd.Categorical(series)
    return cat.codes.astype(np.int64), list(cat.categories)


def load_field_data(path, abstain="merge",
                    col_resp="respondent", col_item="field_id", col_clip="clip_id",
                    col_family="field_family", col_resp_group="base_model",
                    col_y="graded_response", col_ncat="n_categories",
                    col_struct="structural_miss"):
    """Load a long matrix (parquet/csv) into FieldData for the field-level models."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

    # drop structural misses (respondent never saw the clip) — no information about ability
    if col_struct in df.columns:
        df = df[~df[col_struct].astype(bool)].copy()

    df["_keff"] = np.where(df[col_ncat].to_numpy() == 4, 3, 2)

    # ordinal remap (abstain handling)
    raw = df[col_y].to_numpy()
    keff = df["_keff"].to_numpy()
    ordv = np.full(len(df), -1, dtype=np.int64)
    ordv[raw == 3] = keff[raw == 3] - 1          # correct -> top
    ordv[raw == 2] = 1                            # partial (only where keff==3)
    ordv[raw == 1] = 0                            # wrong
    if abstain == "merge":
        ordv[raw == 0] = 0                        # abstain merged with wrong
    elif abstain == "drop":
        keep = raw != 0
        df, ordv, keff = df[keep].copy(), ordv[keep], keff[keep]
    else:
        raise ValueError(f"abstain={abstain!r} (use 'merge' or 'drop')")
    df["_ord"] = ordv
    df["_keff"] = keff

    # contiguous index maps
    resp_codes, resp_names = _codes(df[col_resp])
    item_codes, item_names = _codes(df[col_item])
    clip_codes, clip_names = _codes(df[col_clip])
    fam_codes, fam_names = _codes(df[col_family])

    # per-item attributes (family, clip, keff) — take first occurrence per item
    J = len(item_names)
    item_family = np.zeros(J, np.int64)
    item_clip = np.zeros(J, np.int64)
    item_keff = np.zeros(J, np.int64)
    first = ~df.duplicated(subset=[col_item])
    item_family[item_codes[first]] = fam_codes[first]
    item_clip[item_codes[first]] = clip_codes[first]
    item_keff[item_codes[first]] = df["_keff"].to_numpy()[first]

    # respondent -> base-model group
    resp_to_group_name = (df[[col_resp, col_resp_group]].drop_duplicates()
                          .set_index(col_resp)[col_resp_group])
    group_cat = pd.Categorical(resp_to_group_name.loc[resp_names])
    resp_group = group_cat.codes.astype(np.int64)
    group_names = list(group_cat.categories)

    return FieldData(
        resp=resp_codes, item=item_codes, clip=clip_codes,
        y=df["_ord"].to_numpy(), keff=df["_keff"].to_numpy(),
        item_family=item_family, item_clip=item_clip, item_keff=item_keff,
        resp_group=resp_group,
        n_resp=len(resp_names), n_item=J, n_clip=len(clip_names),
        n_family=len(fam_names), n_group=len(group_names),
        family_names=fam_names, resp_names=resp_names, group_names=group_names,
        clip_names=clip_names,
        binary_mask=(df["_keff"].to_numpy() == 2),
    )


@dataclass
class ClipData:
    """Clip-level dichotomous arrays for M0 at a given threshold."""
    resp: np.ndarray          # (N,) respondent index
    clip: np.ndarray          # (N,) clip index
    y: np.ndarray             # (N,) binary {0,1}
    resp_group: np.ndarray    # (M,)
    n_resp: int
    n_clip: int
    n_group: int
    resp_names: list
    threshold: float


def load_clip_data(path, threshold=0.6,
                   col_resp="respondent", col_clip="clip_id",
                   col_resp_group="base_model", col_y="graded_response",
                   col_struct="structural_miss"):
    """M0 data: clip 'correct' iff mean partial-credit field score >= threshold.

    Field score map: correct=1.0, partial=0.5, wrong=0.0, abstain=0.0.
    """
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if col_struct in df.columns:
        df = df[~df[col_struct].astype(bool)].copy()
    smap = {3: 1.0, 2: 0.5, 1: 0.0, 0: 0.0}
    df["_s"] = df[col_y].map(smap)
    agg = df.groupby([col_resp, col_clip, col_resp_group])["_s"].mean().reset_index()
    agg["_bin"] = (agg["_s"] >= threshold).astype(np.int64)

    resp_codes, resp_names = _codes(agg[col_resp])
    clip_codes, clip_names = _codes(agg[col_clip])
    group_cat = pd.Categorical(
        agg[[col_resp, col_resp_group]].drop_duplicates()
        .set_index(col_resp)[col_resp_group].loc[resp_names])
    return ClipData(
        resp=resp_codes, clip=clip_codes, y=agg["_bin"].to_numpy(),
        resp_group=group_cat.codes.astype(np.int64),
        n_resp=len(resp_names), n_clip=len(clip_names),
        n_group=len(group_cat.categories), resp_names=resp_names,
        threshold=threshold,
    )
