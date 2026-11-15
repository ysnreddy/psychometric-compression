# Psychometric Compression of Structured Benchmarks

Code, data, and released subsets for the CODS 2026 paper *"Psychometric Compression of
Structured Benchmarks: Multidimensional IRT and Stratum-Constrained Subset Selection for
Multi-Field VLM Evaluation"* by Suryanarayana R Yarrabothula, Manisha Chawla, Sandeep
Vadlamudi, Tokachichu Kedarnath, G K Agrawal, Gagan Raj Gupta, and Antariksh Gupta.

This is the **camera-ready** version of the artifact. The released SteelBench-mini clip
list is unchanged from the submission; the held-out validation protocol, the guarantee
sheet, and several analyses were added or revised in response to the reviews (see
"Camera-ready changes" below).

## What is here

| Path | Contents |
|---|---|
| `irtlib/` | **Benchmark-agnostic library**: graded/multidimensional IRT models (NumPyro), SVI fitting, WAIC diagnostics, Fisher-information stratified selection, score reconstruction, CLI. No benchmark-specific imports. |
| `data/` | The frozen graded response matrix (`response_matrix_v1.parquet`: 20 respondents × 13,364 field items over 1,345 clips; `_base` = 9 base models only), clip stratum table, frequency-band map. |
| `configs/`, `grading_maps/` | Versioned build configuration and the ordinal grading maps (default + strict sensitivity variant). |
| `scripts/` | Experiment runners for every number in the paper, table/figure generators, and the guarantee-sheet builder. |
| `results/` | The JSON outputs behind every number in the paper; `results/camera_ready/` holds the camera-ready analyses and the cached LOMO rotation fits. |
| `artifacts/` | **SteelBench-mini** (200-clip manifest, 31/31 strata), the disjoint 194-clip dev subset, and the guarantee sheet (validated score ranges, held-out errors, usage conditions). The submitted sheet is kept as `*_submitted.*`. |

The artifact contains no benchmark media, only graded responses, clip identifiers, and
stratum metadata. SteelBench itself is described in its own paper (TODO: citation).

## Setup

```bash
git clone https://github.com/ysnreddy/psychometric-compression.git
cd psychometric-compression
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export JAX_PLATFORMS=cpu
```

CPU-only. Tested with Python 3.12, NumPyro 0.21.0, JAX 0.11.0 on an Intel i7-13700K: a
full-pool M3 fit takes about one minute and the nine-rotation LOMO run about eight minutes.

## Using SteelBench-mini

Read `artifacts/steelbench_mini_guarantee.md` first. In short:

- The subset is designed for **ranking models** and recovering **capability profiles**.
- Estimate **score levels** with the calibrated estimator (aggregate: blend of calibrated
  IRT and subset mean; capabilities: calibrated per-dimension). The naive subset mean is
  biased on this subset (safety naive MAE about 13 pp).
- Estimate **capability rankings** with the naive per-dimension subset means.
- If you only need aggregate score levels, a uniform random subset of the same size is at
  least as accurate.
- Results are validated for models whose scores fall inside the calibration pool's range
  (aggregate [0.475, 0.575]; recognition [0.355, 0.570]; perception [0.476, 0.661];
  safety [0.374, 0.575]). Outside it, confirm on the full benchmark. Any change to the
  grading map requires revalidation.

## Reproduce the paper

Section numbers refer to the camera-ready paper.

```bash
# §5.1 model ladder (writes results/fits/; the shipped fits are the ones used in the paper)
python scripts/run_ea1_ladder.py
python scripts/run_waic_clustered.py                  # clip-clustered WAIC (z=27)
python scripts/run_partial_misfit.py                  # PPC partial-category misfit, gap prior
# §5.2 capability structure
python scripts/run_ea5_dims.py                        # per-dimension vs unidimensional rankings
python scripts/run_omega_ci.py                        # latent correlations, three grading conventions
python scripts/run_cr_loading_null.py                 # family-level swaps, 25 partitions, permutation null
python scripts/run_sensitivity_fits.py                # cross-loadings and nested pseudo-respondents
# §5.3 structure vs flatness (Table 3)
python scripts/run_c2_headtohead.py
# §5.4 selection, coverage, quota strength (Figures 1-2, Table 4)
python scripts/run_ea2_selection.py --estimator gp_cal
for k in 0 1 2 3 5; do
  python scripts/run_ea4_strata.py --n 200 --min-k $k --estimator gp_irt \
    --out results/camera_ready/quota_insample/n200_k${k}_gp_irt.json
done
# §5.5 capability-level reconstruction (Table 5)
python scripts/run_subscore_compression.py
# §5.6 leave-one-model-out with M3 refit in every rotation (Tables 6-7)
bash scripts/run_cr_quota_lomo.sh                     # k=3 fits and caches; k=0,1,2,5 reuse the cache
python scripts/run_lomo_full.py --models m3 --m3-init median --lr-decay 0.1 --steps 3000 --seeds 2 \
  --sizes 100,150,200,300 --min-k 3 --kmed-seeds 10 --rand-seeds 10 --dump-raw \
  --cache results/camera_ready/lomo_fits --out results/camera_ready/lomo_m3_k3_seeds10.json
python scripts/run_cr_validity_range.py               # validated score ranges
python scripts/run_released_subset_eval.py --lomo-cache results/camera_ready/lomo_fits \
  --out results/camera_ready/released_subset_eval_m3.json
python scripts/run_reliability.py                     # full-benchmark ranking reliability
# §5.7 pseudo-respondents and inference checks
python scripts/run_pseudo_ablation.py --model m3
python scripts/run_cr_clip_effect_m3.py --sub-clips 200 --chains 4 --nuts-warmup 500 --nuts-samples 500 \
  --out results/camera_ready/clip_effect_m3_200_4ch.json   # ~45 min; also --sub-clips 100
python scripts/run_ea4_strata.py --strata-cols calibration_condition --out results/ea7_provenance.json
# §5.8 artifacts, tables, figures
python scripts/run_ea8_artifacts.py --mini 200 --dev 200   # reproduces the released mini + 194-clip dev
                                                      # manifests byte for byte; it also rewrites the
                                                      # guarantee sheet, so run the next line after it
python scripts/make_guarantee_cr.py                   # camera-ready guarantee sheet
TABLES_DIR=tables python scripts/make_tables_cr.py    # camera-ready table bodies + caption facts
python scripts/make_figures.py
```

`results/camera_ready/lomo_fits/` caches the nine M3 rotation fits, so the LOMO, quota,
baseline, and released-subset evaluations rerun in a few minutes without refitting.
Delete it to refit from scratch.

**Numerical note.** The three-category likelihood in `irtlib/models.py` is evaluated in
log space (σ(b) − σ(a) = σ(b)σ(−a)(1 − e^{a−b})), which is algebraically identical to the
ordered-logistic form used for the shipped full-pool fits but cannot underflow in float32.
The camera-ready LOMO fits use this form and reproduce the earlier runs to within 0.05 pp.

## Camera-ready changes

- **Quota-strength ablation** (k ∈ {0, 1, 2, 3, 5}) in sample and held out.
- **LOMO baselines** under the same M3 rotation fits, ten seeds for stochastic selectors.
- **Loading-structure tests**, **clip-effect NUTS check**, and **validated score ranges**.
- **Guarantee sheet** regenerated with the above and an explicit usage statement.

## Use on YOUR benchmark

`irtlib` consumes any long-format table with columns
`(respondent, clip_id, field_id, field_family, graded_response∈{0..3}, n_categories,
structural_miss, base_model)`:

```bash
python -m irtlib.cli fit         --matrix your_matrix.parquet --model m3 --out fit.npz
python -m irtlib.cli select      --matrix your_matrix.parquet --fit fit.npz --n 150 --strata your_strata.csv
python -m irtlib.cli reconstruct --matrix your_matrix.parquet --fit fit.npz --subset subset.txt
```

`scripts/build_response_matrix.py` shows how a structured benchmark's predictions and
ground truth are converted into this format (grading maps are versioned YAML).

## Responsible use

Subset-based scores are estimates, not full-benchmark measurements. Report them with the
guarantee sheet's LOMO interval and the subset version, and state whether the model lies
inside the validated score range.

## Citation

```bibtex
@inproceedings{yarrabothula2026psychometric,
  title     = {Psychometric Compression of Structured Benchmarks: Multidimensional {IRT} and
               Stratum-Constrained Subset Selection for Multi-Field {VLM} Evaluation},
  author    = {Yarrabothula, Suryanarayana R and Chawla, Manisha and Vadlamudi, Sandeep and
               Kedarnath, Tokachichu and Agrawal, G K and Gupta, Gagan Raj and Gupta, Antariksh},
  booktitle = {Proceedings of the ACM India Joint International Conference on Data Science
               and Management of Data (CODS)},
  year      = {2026},
  note      = {TODO: DOI and pages from the ACM rights form}
}
```

## License

MIT (see `LICENSE`).
