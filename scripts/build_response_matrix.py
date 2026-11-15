#!/usr/bin/env python3
"""Build the D23 response matrix: respondent x clip x field -> graded ordinal response.

Long-format output consumed by irtlib. One row per (respondent, field_id). Items
(field_id) are defined by the canonical GT structure and shared across respondents;
each respondent supplies a graded category (3=correct/2=partial/1=wrong/0=abstain)
or abstains. See grading_maps/*.yaml for the ordinal map and configs/matrix_v1.yaml.

Usage:
    python scripts/build_response_matrix.py --config configs/matrix_v1.yaml
    python scripts/build_response_matrix.py --config configs/matrix_v1.yaml --grading-map grading_maps/v2_strict.yaml
    python scripts/build_response_matrix.py --config configs/matrix_v1.yaml --no-pseudo
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent          # CODS Paper-1/subset
PPE_SLOTS = ["helmet", "high_vis_vest", "welding_protection", "harness", "safety_shoes"]
PPE_SKIP_GT = {"not_applicable", "cannot_determine", "", None}   # GT slot carries no elicitable info


# --------------------------------------------------------------------------- taxonomy
def load_taxonomy(repo_root):
    sys.path.insert(0, str(repo_root))
    from annotation_tool.agreement import CODE_TO_GROUP
    return CODE_TO_GROUP


def taxonomic_distance(a, b, code_to_group):
    if a == b:
        return 0.0
    ga, gb = code_to_group.get(a, "X"), code_to_group.get(b, "X")
    if ga == gb:
        return 0.33
    order = ["A", "B", "C", "D", "E", "F", "X"]
    try:
        return min(0.50 + abs(order.index(ga) - order.index(gb)) * 0.10, 1.0)
    except ValueError:
        return 1.0


# --------------------------------------------------------------------------- grading
# Each returns an int category in {3,2,1,0}. Rule names come from the grading map.
def grade_action(gt, pred, rule, c2g):
    if not pred:
        return 0
    if rule == "exact_strict":
        return 3 if gt == pred else 1
    d = taxonomic_distance(gt, pred, c2g)
    if d == 0.0:
        return 3
    if d == 0.33:
        return 2
    return 1


def grade_scene(gt, pred, rule):
    if not pred:
        return 0
    return 3 if str(gt).upper() == str(pred).upper() else 1


def grade_ppe(gtv, predv, rule):
    if predv in (None, ""):
        return 0
    if predv == "cannot_determine":
        return 1 if rule == "ppe_strict" else 0
    return 3 if predv == gtv else 1


def grade_count(gt, pred, rule):
    if pred is None:
        return 0
    try:
        delta = abs(int(gt) - int(pred))
    except (TypeError, ValueError):
        return 0
    if delta == 0:
        return 3
    if rule == "count_strict":
        return 1
    return 2 if delta == 1 else 1


def _violation(txt):
    t = (txt or "").strip().lower()
    return t not in ("", "none")


def grade_safety(gt_txt, pred_txt, pred_missing):
    if pred_missing:
        return 0
    return 3 if _violation(gt_txt) == _violation(pred_txt) else 1


# --------------------------------------------------------------------------- loaders
def load_predictions(path):
    """Last-success-wins per clip_id (absorbs reruns appended to the base jsonl)."""
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = r.get("clip_id")
            if cid and r.get("success"):
                preds[cid] = r                      # later line overwrites earlier
    return preds


def load_gt(gt_dir):
    gt = {}
    for fn in sorted(Path(gt_dir).glob("*.json")):
        try:
            d = json.load(open(fn))
        except json.JSONDecodeError:
            continue
        cid = d.get("clip_id")
        if cid:
            gt[cid] = d
    return gt


def load_sites(manifest_path):
    sites = {}
    if Path(manifest_path).exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                sites[row.get("clip_id", "")] = row.get("site", "")
    return sites


# --------------------------------------------------------------------------- freq bands
def derive_freq_bands(pool, gt, head_cut, tail_cut):
    counts = Counter()
    for cid in pool:
        d = gt[cid]
        if d.get("annotation_layer") == 2:
            for p in d.get("persons", []) or []:
                c = p.get("action_code")
                if c:
                    counts[c] += 1
        else:
            dom = d.get("dominant_actions") or []
            if dom:
                counts[dom[0]] += 1
    total = sum(counts.values()) or 1
    ranked = counts.most_common()
    band, cum = {}, 0.0
    for code, n in ranked:
        share = n / total
        prev = cum                      # cumulative BEFORE adding this code
        cum += share
        if prev < head_cut:             # inclusive: the code that tips past the cutoff stays head
            band[code] = "head"
        elif share < tail_cut:
            band[code] = "tail"
        else:
            band[code] = "mid"
    return band, dict(counts)


def clip_band(gt_rec, band):
    codes = []
    if gt_rec.get("annotation_layer") == 2:
        codes = [p.get("action_code") for p in (gt_rec.get("persons") or []) if p.get("action_code")]
    else:
        dom = gt_rec.get("dominant_actions") or []
        codes = dom[:1]
    order = {"tail": 0, "mid": 1, "head": 2}
    bands = [band.get(c, "mid") for c in codes] or ["mid"]
    return min(bands, key=lambda b: order[b])          # rarest band present


# --------------------------------------------------------------------------- item enumeration
def clip_items(gt_rec, families):
    """Yield (field_id_suffix, family, person_index, slot, gt_value) for one clip's GT."""
    layer = gt_rec.get("annotation_layer", 2)
    if "scene_type" in families and gt_rec.get("scene_type"):
        yield ("scene_type", "scene_type", -1, None, gt_rec.get("scene_type"))
    if "count" in families and gt_rec.get("num_workers") is not None:
        yield ("count", "count", -1, None, gt_rec.get("num_workers"))
    if layer == 2 and gt_rec.get("persons"):
        for i, p in enumerate(gt_rec["persons"]):
            code = p.get("action_code")
            if "action" in families and code:
                yield (f"action_p{i}", "action", i, None, code)
            if "ppe" in families:
                for slot in PPE_SLOTS:
                    gv = (p.get("ppe") or {}).get(slot)
                    if gv not in PPE_SKIP_GT:
                        yield (f"ppe_p{i}_{slot}", "ppe", i, slot, gv)
            if "safety" in families:
                yield (f"safety_p{i}", "safety", i, None, p.get("unsafe_act"))
    else:  # layer 1 (scene-level)
        dom = gt_rec.get("dominant_actions") or []
        if "action" in families and dom:
            yield ("action_dom0", "action", -1, None, dom[0])
        if "safety" in families:
            yield ("safety_scene", "safety", -1, None, gt_rec.get("scene_unsafe_act"))


def grade_one(family, gt_val, person_idx, slot, pred_norm, pred_missing, rule_by_fam, c2g):
    """Grade a single item for one respondent. pred_missing => whole-cell failure."""
    rule = rule_by_fam[family]
    if pred_missing:
        return 0
    persons = pred_norm.get("persons") or []
    if family == "scene_type":
        return grade_scene(gt_val, pred_norm.get("scene_type"), rule)
    if family == "count":
        return grade_count(gt_val, pred_norm.get("num_workers"), rule)
    if family == "action":
        if person_idx == -1:                                    # layer-1 dominant
            dom = pred_norm.get("dominant_actions") or []
            return grade_action(gt_val, dom[0] if dom else "", rule, c2g)
        pv = persons[person_idx].get("action_code", "") if person_idx < len(persons) else ""
        return grade_action(gt_val, pv, rule, c2g)
    if family == "ppe":
        if person_idx >= len(persons):
            return 0
        return grade_ppe(gt_val, (persons[person_idx].get("ppe") or {}).get(slot), rule)
    if family == "safety":
        if person_idx == -1:                                    # layer-1 scene
            return grade_safety(gt_val, pred_norm.get("unsafe_act"), False)
        if person_idx >= len(persons):
            return 0
        return grade_safety(gt_val, persons[person_idx].get("unsafe_act"), False)
    raise ValueError(family)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/matrix_v1.yaml")
    ap.add_argument("--grading-map", default=None, help="override the map named in the config")
    ap.add_argument("--no-pseudo", action="store_true", help="base respondents only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / args.config))
    roster = yaml.safe_load(open(HERE / cfg["roster"]))
    gmap_path = args.grading_map or cfg["grading_map"]
    gmap = yaml.safe_load(open(HERE / gmap_path))
    rule_by_fam = {fam: spec["rule"] for fam, spec in gmap["families"].items()}
    ncat_by_fam = {fam: spec["n_categories"] for fam, spec in gmap["families"].items()}
    families = cfg["families"]

    repo = Path(roster["repo_root"])
    results_dir = repo / roster["results_dir"]
    c2g = load_taxonomy(repo)
    gt = load_gt(repo / cfg["gt_dir"])
    sites = load_sites(repo / cfg["manifest"])

    # respondent roster
    resp = list(roster["base"])
    include_pseudo = cfg.get("include_pseudo", True) and not args.no_pseudo
    if include_pseudo:
        resp += list(roster["pseudo"])
    resp_kind = {r["id"]: ("base" if r in roster["base"] else "pseudo") for r in resp}
    preds = {r["id"]: load_predictions(results_dir / r["file"]) for r in resp}

    # pool = intersection of base-model success clips, restricted to GT
    base_ids = [r["id"] for r in roster["base"]]
    base_clip_sets = [set(preds[i].keys()) for i in base_ids]
    if cfg["pool"] == "intersection":
        pool = set.intersection(*base_clip_sets) & set(gt.keys())
    else:
        want = json.load(open(repo / cfg["pool"]))
        want = set(want if isinstance(want, list) else want.keys())
        pool = want & set(gt.keys())
    pool = sorted(pool)

    band, code_counts = derive_freq_bands(
        pool, gt, cfg["freq_bands"]["head_mass_cutoff"], cfg["freq_bands"]["tail_share_cutoff"])
    (HERE / cfg["freq_bands"]["out"]).write_text(
        json.dumps({"band": band, "counts": code_counts}, indent=2))

    # ---- enumerate items once (shared identity), then grade every respondent
    strata_rows, catalog_rows, matrix_rows = [], [], []
    field_index = {}                                            # field_id -> family for catalog
    for cid in pool:
        g = gt[cid]
        stratum = {
            "clip_id": cid,
            "layer": g.get("annotation_layer"),
            "site": sites.get(cid, ""),
            "tier": g.get("tier"),
            "severity": g.get("severity_level"),
            "visibility": "|".join(g.get("visibility_conditions") or []) or "clear",
            "visibility_primary": next((v for v in (g.get("visibility_conditions") or []) if v != "clear"),
                                       (g.get("visibility_conditions") or ["clear"])[0]),
            "calibration_condition": g.get("calibration_condition"),
            "freq_band": clip_band(g, band),
            "gt_source": g.get("annotator_id"),
        }
        strata_rows.append(stratum)

        items = list(clip_items(g, families))
        for suffix, fam, pidx, slot, gtv in items:
            fid = f"{cid}::{suffix}"
            field_index[fid] = fam
            catalog_rows.append({"field_id": fid, "clip_id": cid, "family": fam,
                                 "person_index": pidx, "slot": slot,
                                 "gt_value": gtv, "n_categories": ncat_by_fam[fam]})
            for rid in (r["id"] for r in resp):
                pred = preds[rid].get(cid)
                pred_missing = pred is None
                pnorm = (pred or {}).get("normalized", {}) or {}
                cat = grade_one(fam, gtv, pidx, slot, pnorm, pred_missing, rule_by_fam, c2g)
                # structural miss: pseudo/partial respondent simply never saw this clip
                structural = pred_missing and cid not in preds[rid]
                matrix_rows.append({
                    "respondent": rid, "base_model": next(r["base_model"] for r in resp if r["id"] == rid),
                    "respondent_kind": resp_kind[rid],
                    "clip_id": cid, "field_id": fid, "field_family": fam,
                    "person_index": pidx, "graded_response": cat,
                    "n_categories": ncat_by_fam[fam], "structural_miss": structural,
                    "layer": stratum["layer"], "site": stratum["site"], "tier": stratum["tier"],
                    "severity": stratum["severity"], "visibility_primary": stratum["visibility_primary"],
                    "calibration_condition": stratum["calibration_condition"],
                    "freq_band": stratum["freq_band"],
                })

    import pandas as pd
    mat = pd.DataFrame(matrix_rows)
    strata = pd.DataFrame(strata_rows)
    catalog = pd.DataFrame(catalog_rows)

    # Variant builds (base-only, non-default grading map) write to tagged paths so the
    # canonical augmented+default matrix is never clobbered by an ablation/sensitivity run.
    tag = ("" if include_pseudo else "_base")
    if gmap_path != cfg["grading_map"]:
        tag += "_" + gmap["version"]
    out = dict(cfg["outputs"])
    if tag:
        def _suffix(p):
            name = Path(p).name
            for ext in (".csv.gz", ".parquet", ".csv", ".json"):
                if name.endswith(ext):
                    return str(Path(p).with_name(name[:-len(ext)] + tag + ext))
            return str(Path(p).with_name(Path(p).stem + tag + Path(p).suffix))
        out = {k: _suffix(v) for k, v in out.items()}
    (HERE / "data").mkdir(exist_ok=True)
    mat.to_parquet(HERE / out["matrix_parquet"], index=False)
    mat.to_csv(HERE / out["matrix_csv"], index=False, compression="gzip")
    strata.to_csv(HERE / out["clip_strata"], index=False)
    catalog.to_csv(HERE / out["field_catalog"], index=False)

    # ---- QC report (the sign-off gate) --------------------------------------
    cat_names = {3: "correct", 2: "partial", 1: "wrong", 0: "abstain"}
    qc = {
        "matrix_version": cfg["matrix_version"], "grading_map": gmap_path,
        "include_pseudo": include_pseudo,
        "n_clips": len(pool), "n_field_items": len(catalog),
        "n_respondents": len(resp), "n_graded_responses": len(mat),
        "layer_split": strata["layer"].value_counts().to_dict(),
        "items_per_family": catalog["family"].value_counts().to_dict(),
        "per_respondent": {}, "category_marginals_per_family": {},
        "stratum_crosstab_visibility_x_layer_x_band": {},
        "code_freq_bands": {"head": [c for c, b in band.items() if b == "head"],
                            "mid": [c for c, b in band.items() if b == "mid"],
                            "tail": [c for c, b in band.items() if b == "tail"]},
    }
    for rid in (r["id"] for r in resp):
        sub = mat[mat.respondent == rid]
        real = sub[~sub.structural_miss]
        vc = real.graded_response.value_counts().to_dict()
        n = len(real) or 1
        qc["per_respondent"][rid] = {
            "kind": resp_kind[rid], "n_real_responses": int(len(real)),
            "n_structural_miss": int(sub.structural_miss.sum()),
            "abstain_rate": round(vc.get(0, 0) / n, 4),
            "correct_rate": round(vc.get(3, 0) / n, 4),
        }
    for fam in families:
        sub = mat[(mat.field_family == fam) & (~mat.structural_miss)]
        vc = sub.graded_response.value_counts().to_dict()
        tot = len(sub) or 1
        qc["category_marginals_per_family"][fam] = {
            cat_names[k]: round(vc.get(k, 0) / tot, 4) for k in (3, 2, 1, 0)}
    ct = strata.groupby(["visibility_primary", "layer", "freq_band"]).size()
    qc["stratum_crosstab_visibility_x_layer_x_band"] = {
        "|".join(map(str, k)): int(v) for k, v in ct.items()}
    (HERE / out["qc_report_json"]).write_text(json.dumps(qc, indent=2))

    print(f"[D23] pool={len(pool)} clips | items={len(catalog)} | respondents={len(resp)} "
          f"({sum(1 for r in resp if resp_kind[r['id']]=='base')} base + "
          f"{sum(1 for r in resp if resp_kind[r['id']]=='pseudo')} pseudo)")
    print(f"[D23] graded responses={len(mat):,}  ->  {HERE/out['matrix_parquet']}")
    print(f"[D23] QC report -> {HERE/out['qc_report_json']}")
    print("[D23] category marginals per family:")
    for fam, m in qc["category_marginals_per_family"].items():
        print(f"       {fam:11s} {m}")
    print(f"[D23] freq bands: head={len(qc['code_freq_bands']['head'])} "
          f"mid={len(qc['code_freq_bands']['mid'])} tail={len(qc['code_freq_bands']['tail'])}")


if __name__ == "__main__":
    main()
