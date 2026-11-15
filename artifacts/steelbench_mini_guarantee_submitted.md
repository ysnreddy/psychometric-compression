# SteelBench-mini — Guarantee Sheet

- **Clips:** 200 (of 1345)  ~14.9% compression
- **Selection:** Fisher-information greedy under per-stratum floor quotas (visibility x layer x freq-band, min 3/stratum)
- **Response model:** m3;  grading map: v1_default (sha in grading_maps/)
- **Dev subset:** 194 clips, disjoint; dev-vs-full sanity MAE 1.0pp

## Validity conditions
1. Estimates are calibrated on a 9-model black-box respondent pool; extrapolation beyond that ability range is not guaranteed (see EA.3 best-model holdout).
2. Item parameters assume the v1 ordinal grading map; a stricter map (v2_strict) shifts absolute scores — re-derive the subset if the map changes.
3. The subset reproduces AGGREGATE and per-stratum scores; it is not a substitute for full evaluation when a new capability dimension is introduced.
4. Pseudo-respondents (prompt/frame perturbations) stabilise item parameters but do not add independent ability points; ranking claims rest on the 9 base models.

**Responsible use.** Report subset-based scores as estimates with the EA.3 CI, not as exact full-benchmark numbers; disclose the subset size and version.

## Held-out fidelity (EA.3 LOMO)
| n | MAE all | MAE interior | err best-model | tau |
|---|---|---|---|---|
| 100 | 2.33pp | 2.26pp | 3.98pp | 0.33 |
| 150 | 1.63pp | 1.48pp | 3.10pp | 0.39 |
| 200 | 1.39pp | 1.34pp | 2.13pp | 0.61 |

## Held-out capability sub-scores (EA.3b LOMO; naive subset-mean levels, ranking tau)
| n | recognition MAE/tau | perception MAE/tau | safety MAE/tau |
|---|---|---|---|
| 100 | 4.24pp / 0.89 | 4.72pp / 0.72 | 2.11pp / 0.83 |
| 150 | 4.67pp / 1.00 | 3.67pp / 0.78 | 2.14pp / 0.78 |
| 200 | 4.43pp / 1.00 | 3.00pp / 0.78 | 1.70pp / 0.83 |

<!-- generated: fixed-subset evaluation -->

## Fixed-subset evaluation (this exact 200-clip list)

In-sample (selection and item parameters saw all nine models):

| estimator | aggregate MAE / tau | recognition MAE (bias) / tau | perception | safety |
|---|---|---|---|---|
| subset_mean | 2.38pp / 0.72 | 4.4pp (-4.3) / 0.94 | 3.4pp (+3.3) / 0.89 | 12.9pp (-3.5) / 1.00 |
| irt_cal | 1.43pp / 0.72 | 2.5pp (+0.1) / 0.61 | 2.9pp (+0.0) / 0.44 | 1.7pp (-0.1) / 0.83 |
| gp_cal | 1.36pp / 0.89 | 2.3pp (-1.2) / 0.78 | 2.3pp (+1.0) / 0.67 | 4.2pp (-1.1) / 0.94 |

Fixed-subset held-out (M2 rotation item parameters; only the selection saw the model):

| estimator | aggregate MAE [95% CI] / tau | recognition MAE / tau | perception | safety |
|---|---|---|---|---|
| subset_mean | 2.38pp [1.3, 3.5] / 0.72 | 4.4pp / 0.94 | 3.4pp / 0.89 | 12.9pp / 1.00 |
| irt_cal | 3.08pp [1.6, 4.7] / -0.39 | 7.3pp / -0.44 | 3.8pp / 0.17 | 5.7pp / 0.17 |
| gp_cal | 1.93pp [0.9, 2.9] / 0.56 | -- | -- | -- |

Fixed-subset held-out (M3 rotation item parameters; only the selection saw the model):

| estimator | aggregate MAE [95% CI] / tau | recognition MAE / tau | perception | safety |
|---|---|---|---|---|
| subset_mean | 2.38pp [1.3, 3.5] / 0.72 | 4.4pp / 0.94 | 3.4pp / 0.89 | 12.9pp / 1.00 |
| irt_cal | 1.31pp [0.6, 2.1] / 0.78 | 3.4pp / 0.56 | 2.5pp / 0.39 | 2.1pp / 0.78 |
| gp_cal | 1.27pp [0.7, 1.9] / 0.83 | -- | -- | -- |

**Capability levels.** The naive per-dimension subset mean on this subset is biased for safety (signed bias -3.5pp, MAE 12.9pp); use the calibrated per-dimension estimator for levels (safety MAE 1.7pp) and the naive per-dimension mean for rankings.

## Held-out baselines (LOMO, M2 rotation fits (paper protocol), per-rotation selection, n=200)

| selector | subset-mean MAE [CI] / tau | IRT-calibrated MAE [CI] / tau | blend MAE [CI] / tau |
|---|---|---|---|
| fisher_stratified | 1.39pp [1.0, 1.7] / 0.61 | 2.86pp [1.4, 4.5] / -0.56 | 1.77pp [0.8, 2.9] / 0.50 |
| fisher_unconstrained | 1.37pp [1.0, 1.7] / 0.56 | 2.94pp [1.5, 4.5] / -0.56 | 2.05pp [1.0, 3.2] / 0.50 |
| stratified_random | 1.14pp [1.0, 1.3] / 0.73 | 3.03pp [1.6, 4.6] / -0.50 | 2.16pp [1.1, 3.3] / 0.52 |
| random | 0.99pp [0.8, 1.2] / 0.68 | 2.98pp [1.6, 4.6] / -0.44 | 2.08pp [1.1, 3.2] / 0.59 |
| kmedoids | 0.91pp [0.6, 1.2] / 0.67 | 2.94pp [1.5, 4.5] / -0.44 | 2.00pp [1.0, 3.1] / 0.69 |
| disagree_stratified | 1.19pp [0.8, 1.6] / 0.67 | 3.04pp [1.6, 4.7] / -0.44 | 2.35pp [1.4, 3.4] / 0.50 |

## Held-out baselines (LOMO, M3 rotation fits (stabilised init, 9/9 rotations), per-rotation selection, n=200)

| selector | subset-mean MAE [CI] / tau | IRT-calibrated MAE [CI] / tau | blend MAE [CI] / tau |
|---|---|---|---|
| fisher_stratified | 2.20pp [1.1, 3.5] / 0.67 | 1.41pp [0.7, 2.2] / 0.78 | 1.20pp [0.8, 1.7] / 0.89 |
| fisher_unconstrained | 1.43pp [0.9, 2.1] / 0.56 | 1.51pp [0.9, 2.3] / 0.83 | 1.04pp [0.5, 1.5] / 0.89 |
| stratified_random | 1.14pp [1.0, 1.3] / 0.73 | 1.56pp [0.9, 2.2] / 0.72 | 1.30pp [0.9, 1.7] / 0.79 |
| random | 0.99pp [0.8, 1.2] / 0.68 | 1.47pp [0.8, 2.2] / 0.70 | 1.09pp [0.7, 1.6] / 0.73 |
| kmedoids | 0.76pp [0.5, 1.2] / 0.67 | 1.83pp [1.2, 2.6] / 0.72 | 1.26pp [0.8, 1.8] / 0.86 |
| disagree_stratified | 1.19pp [0.8, 1.6] / 0.67 | 1.70pp [0.7, 2.8] / 0.72 | 1.37pp [0.7, 2.1] / 0.78 |

## Ranking reliability of the FULL benchmark (clip bootstrap)

| quantity | tau(bootstrap vs full) | split-half tau | model pairs stable at 95% |
|---|---|---|---|
| recognition | 0.98 | 0.96 | 33/36 |
| perception | 0.93 | 0.87 | 32/36 |
| safety | 0.91 | 0.81 | 32/36 |
| aggregate | 0.81 | 0.63 | 24/36 |

The aggregate ranking of the nine models is the least reliable quantity even on the full benchmark; subset-based aggregate rank claims cannot exceed this ceiling.
