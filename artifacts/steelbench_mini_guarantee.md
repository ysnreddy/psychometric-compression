# SteelBench-mini — Guarantee Sheet (camera-ready)

- **Clips:** 200 of 1345 (unchanged from the submitted version), 2680 field items, 31/31 strata
- **Selection:** Fisher-information greedy under per-stratum floor quotas (visibility x layer x freq-band, min 3/stratum)
- **Response model:** M3 (confirmatory 3-D graded MIRT); grading map v1_default (sha in grading_maps/)
- **Dev subset:** 194 clips, disjoint from the evaluation subset
- **Held-out protocol:** LOMO: M3 refit on 8 primary models + their pseudo-respondents in every rotation (median init, lr decay; 9/9 converged), selection rerun, affine calibration fitted on the 8 training models. Headline estimator: blend 0.7*calibrated + 0.3*subset mean.

## How to use
- Designed for RANKING models and recovering CAPABILITY PROFILES.
- Score levels: use the calibrated estimator (aggregate: blend; capability: calibrated per-dimension); the naive subset mean is biased on this subset (safety naive MAE ~13 pp).
- Capability rankings: use the naive per-dimension subset means.
- If only aggregate score LEVELS are needed, a uniform random subset of equal size is at least as accurate.

## Validity conditions
1. Validated for models whose aggregate and capability scores fall inside the calibrated ranges of the nine-model pool (validity_range.calibrated_score_ranges). Outside them the estimate is an extrapolation: held-out error roughly doubles to quadruples per dimension; confirm on the full benchmark.
2. Item parameters assume the v1 ordinal grading map; any change to the grading map requires revalidation.
3. Pseudo-respondents stabilise item parameters but are variants of the nine models, not independent ability points.
4. The subset is not a substitute for full evaluation when a new capability dimension is introduced.

## Validated score ranges (nine calibration models)

| quantity | range |
|---|---|
| aggregate | [0.475, 0.575] |
| recognition | [0.355, 0.570] |
| perception | [0.476, 0.661] |
| safety | [0.374, 0.575] |

Inside the range the blend errs by 1.06 pp on average (max 1.92 pp); edge rotations by 1.7 pp. Calibrated sub-score MAE inside vs outside the range: recognition 2.6 vs 4.9 pp, perception 1.8 vs 7.5 pp, safety 1.9 vs 4.7 pp.

## Held-out fidelity (LOMO, M3 in every rotation, Fisher + k=3 floors)

| n | blend MAE [95% CI] | interior | top model | tau | naive MAE | recognition cal. MAE / tau | perception | safety |
|---|---|---|---|---|---|---|---|---|
| 100 | 1.50 pp [0.9, 2.1] | 1.47 | 2.82 | 0.83 | 3.06 | 3.0 pp / 1.00 | 3.1 pp / 1.00 | 2.6 pp / 0.94 |
| 150 | 1.42 pp [0.9, 1.9] | 1.33 | 2.76 | 0.89 | 2.80 | 2.9 pp / 1.00 | 3.0 pp / 0.94 | 2.7 pp / 1.00 |
| 200 | 1.21 pp [0.8, 1.7] | 1.06 | 2.82 | 0.89 | 2.20 | 3.1 pp / 0.94 | 3.1 pp / 0.94 | 2.5 pp / 1.00 |
| 300 | 1.10 pp [0.6, 1.6] | 1.00 | 2.41 | 0.94 | 1.63 | 3.1 pp / 0.94 | 3.0 pp / 0.94 | 2.6 pp / 0.94 |

## This exact clip list, held out (M3 rotation item parameters and calibration)

Aggregate blend: 1.34 pp [0.8, 2.0], tau 0.94.

| dimension | naive MAE / tau | calibrated MAE |
|---|---|---|
| recognition | 4.4 pp / 0.94 | 2.9 pp |
| perception | 3.4 pp / 0.89 | 3.1 pp |
| safety | 12.9 pp / 1.00 | 1.9 pp |

Per-stratum MAE bound (in-sample, n=200): 17.6 pp with the released k=3 floors (10.9 pp with k=5).

## Held-out baselines (LOMO, same M3 rotation fits, n=200; stochastic selectors 10 seeds)

| selector | naive mean MAE [CI] / tau | blend MAE [CI] / tau |
|---|---|---|
| fisher_stratified | 2.20 [1.1, 3.5] / 0.67 | 1.21 [0.8, 1.7] / 0.89 |
| fisher_unconstrained | 1.42 [0.8, 2.1] / 0.56 | 1.05 [0.6, 1.6] / 0.89 |
| stratified_random | 1.05 [1.0, 1.1] / 0.68 | 1.23 [0.9, 1.6] / 0.76 |
| random | 1.07 [0.9, 1.2] / 0.70 | 1.07 [0.7, 1.5] / 0.75 |
| kmedoids | 1.01 [0.8, 1.3] / 0.71 | 0.97 [0.7, 1.4] / 0.84 |
| disagree_stratified | 1.19 [0.8, 1.6] / 0.67 | 1.37 [0.7, 2.1] / 0.78 |

## Ranking reliability of the FULL benchmark (clip bootstrap)

| quantity | tau(bootstrap vs full) | split-half tau | model pairs stable at 95% |
|---|---|---|---|
| recognition | 0.98 | 0.96 | 33/36 |
| perception | 0.93 | 0.87 | 32/36 |
| safety | 0.91 | 0.81 | 32/36 |
| aggregate | 0.81 | 0.63 | 24/36 |

The aggregate ranking of the nine models is the least reliable quantity even on the full benchmark; subset-based aggregate rank claims cannot exceed this ceiling.

**Responsible use.** Report subset-based scores as estimates with the LOMO interval, disclose the subset version and size, and state whether the model lies inside the validated range.
