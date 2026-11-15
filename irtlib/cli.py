"""Benchmark-agnostic CLI for the compression tool (C5 deliverable).

Consumes ANY long-format (respondent x item x graded-response) matrix — not SteelBench-
specific. All SteelBench logic lives in scripts/build_response_matrix.py; irtlib itself
only needs the columns below.

    python -m irtlib.cli fit       --matrix any.csv --out fit.npz
    python -m irtlib.cli select    --matrix any.csv --fit fit.npz --n 150 --strata strata.csv
    python -m irtlib.cli reconstruct --matrix any.csv --fit fit.npz --subset subset.txt

Required matrix columns (override with flags): respondent, clip_id (group), field_id (item),
field_family, graded_response (0..3), n_categories. Optional: base_model, structural_miss.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np


def _fit(args):
    from . import io, models, fit, prep, params as P
    fd = io.load_field_data(args.matrix, abstain=args.abstain)
    ug = fd.n_group < fd.n_resp
    A = prep.field_arrays(fd)
    if args.model == "m3":
        idim = prep.item_dim_array(fd, args.dim)
        margs, model = prep.m3_args(A, idim, args.dim), models.m3
    elif args.model == "m2":
        idim, margs, model = None, prep.m2_args(A), models.m2
    else:
        idim, margs, model = None, prep.m1_args(A), models.m1
    r = fit.fit_svi(model, margs, {"use_group": ug}, n_steps=args.steps, n_seeds=args.seeds, n_post=200)
    d = {k: np.asarray(r["derived"][k]) for k in ("theta", "a", "b", "k0", "k1", "u")
         if k in r["derived"]}
    np.savez(args.out, item_family=fd.item_family, item_clip=fd.item_clip, item_keff=fd.item_keff,
             resp_names=np.array(fd.resp_names), family_names=np.array(fd.family_names),
             **({"item_dim": np.asarray(idim)} if idim is not None else {}), **d)
    print(f"fit {args.model} -> {args.out}  (elbo={r['loss']:.0f}, {r['seconds']}s)")


def _select(args):
    from . import io, params as P, selection as S
    import pandas as pd
    fd = io.load_field_data(args.matrix)
    p = P.ItemParams.from_npz(args.fit)
    tload = np.load(args.fit, allow_pickle=True)["theta"]
    tmean = tload.mean(0) if tload.ndim in (2, 3) else tload
    base = [i for i, n in enumerate(fd.resp_names) if n in set(fd.group_names)]
    clip_info, _ = S.clip_information(p, tmean[base])
    if args.strata:
        sdf = pd.read_csv(args.strata).set_index("clip_id").loc[fd.clip_names].reset_index()
        sel = S.greedy_stratified(clip_info, S.build_strata(sdf), args.n, min_k=args.min_k)
    else:
        sel = np.argsort(-clip_info)[:args.n]
    Path(args.out).write_text("\n".join(fd.clip_names[c] for c in sel))
    print(f"selected {len(sel)} items -> {args.out}")


def _reconstruct(args):
    from . import io, params as P, harness as H, reconstruct as R
    fd = io.load_field_data(args.matrix)
    p = P.ItemParams.from_npz(args.fit)
    Y = H.response_table(fd)
    subset_ids = set(Path(args.subset).read_text().split())
    S = [c for c, name in enumerate(fd.clip_names) if name in subset_ids]
    Sset = set(S)
    item_in_S = np.array([c in Sset for c in p.clip])
    out = {}
    for i in H.base_indices(fd):
        yi = Y[i]; present = yi >= 0
        idx = np.where(item_in_S & present)[0]
        out[fd.resp_names[i]] = R.reconstruct(p, idx, yi[idx])["gp_irt"]
    print(json.dumps({k: round(v, 4) for k, v in out.items()}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="irtlib.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--matrix", required=True)
    f.add_argument("--out", default="fit.npz"); f.add_argument("--model", default="m2")
    f.add_argument("--dim", type=int, default=3); f.add_argument("--steps", type=int, default=4000)
    f.add_argument("--seeds", type=int, default=3); f.add_argument("--abstain", default="merge")
    f.set_defaults(fn=_fit)
    s = sub.add_parser("select"); s.add_argument("--matrix", required=True)
    s.add_argument("--fit", required=True); s.add_argument("--n", type=int, default=150)
    s.add_argument("--strata", default=None); s.add_argument("--min-k", type=int, default=3)
    s.add_argument("--out", default="subset.txt"); s.set_defaults(fn=_select)
    r = sub.add_parser("reconstruct"); r.add_argument("--matrix", required=True)
    r.add_argument("--fit", required=True); r.add_argument("--subset", required=True)
    r.set_defaults(fn=_reconstruct)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
