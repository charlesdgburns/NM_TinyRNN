# Weight similarity sensitivity to pairwise scaling

All comparisons use selected raw fitted weights. No minimum-norm balancing or subject bootstrap is applied.

| Comparison | Within subjects | Across subjects |
|---|---:|---:|
| ReLU: discrete → continuous scaling | +0.14–+1.14% | +0.10–+2.13% |
| tanh: unscaled → hypothetical continuous scaling | -0.47–+1.35% | -0.12–+2.08% |

Ranges describe percentage changes in variant means, not uncertainty intervals. Each variant mean equally averages subjects or unordered subject dyads. Percent change = 100 × (scaled mean − baseline mean) / |baseline mean|. Individual near-zero baseline scores make percentage changes unstable; figures show absolute cosine differences.

ReLU: exhaustive permutation/grid search using {0.5, 0.875, 1, 1.25, 1.625, 2}, then TRF least-squares refinement from the winning scales with its permutation fixed. Tanh: exhaustive permutation/sign baseline; the hypothetical scaled condition refines positive scales in every sign/permutation branch, starting at unit scales. Tanh scaling changes function. Both use the core Euclidean alignment objective; cosine includes all trainable parameters, so scaling can lower the reported cosine.

Both continuous procedures use analytical Jacobians, log-scale bounds [-20,20], ftol/xtol/gtol 1e-10 and at most 300 residual evaluations. A successful ReLU refinement is retained only if it improves the grid objective. The hypothetical tanh helper retains the baseline if a candidate does not improve the distance. Neither local search proves a global continuous optimum.

The combined discrete figure reproduces the previous standard-method contrast. The continuous figure instead uses grid-refined ReLU scores in its lower panels. Both keep tanh unscaled in the activation contrast; tanh scaling appears only in the sensitivity panels.

Selected fits: 1,320; fit pairs: 71,940; failed optimisation branches: 0; selected scales at numerical bounds: 0.

## Per-variant changes

| Variant | Scope | Baseline | With continuous scaling | Δ cosine | % change |
|---|---|---:|---:|---:|---:|
| GRU+BC | across_subject | 0.32064 | 0.32096 | +0.00031 | +0.10% |
| GRU+BC | within_subject | 0.60619 | 0.60716 | +0.00098 | +0.16% |
| GRU+BC-DB | across_subject | 0.35403 | 0.35498 | +0.00095 | +0.27% |
| GRU+BC-DB | within_subject | 0.58462 | 0.58545 | +0.00084 | +0.14% |
| GRU | across_subject | 0.32417 | 0.32456 | +0.00040 | +0.12% |
| GRU | within_subject | 0.63793 | 0.63745 | -0.00048 | -0.08% |
| GRU-DB | across_subject | 0.28903 | 0.28869 | -0.00034 | -0.12% |
| GRU-DB | within_subject | 0.59374 | 0.59094 | -0.00280 | -0.47% |
| lightGRU+BC | across_subject | 0.41383 | 0.41742 | +0.00359 | +0.87% |
| lightGRU+BC | within_subject | 0.68972 | 0.69264 | +0.00292 | +0.42% |
| lightGRU+BC-DB | across_subject | 0.50084 | 0.50190 | +0.00106 | +0.21% |
| lightGRU+BC-DB | within_subject | 0.74430 | 0.74625 | +0.00195 | +0.26% |
| lightGRU | across_subject | 0.54816 | 0.55956 | +0.01140 | +2.08% |
| lightGRU | within_subject | 0.77160 | 0.77603 | +0.00444 | +0.57% |
| lightGRU-DB | across_subject | 0.51077 | 0.52041 | +0.00964 | +1.89% |
| lightGRU-DB | within_subject | 0.72852 | 0.73839 | +0.00987 | +1.35% |
| monoGRU+BC | across_subject | 0.49479 | 0.50533 | +0.01054 | +2.13% |
| monoGRU+BC | within_subject | 0.75700 | 0.76566 | +0.00866 | +1.14% |
| monoGRU+BC-DB | across_subject | 0.63873 | 0.64113 | +0.00240 | +0.38% |
| monoGRU+BC-DB | within_subject | 0.83249 | 0.83458 | +0.00210 | +0.25% |
| monoGRU | across_subject | 0.55118 | 0.55926 | +0.00808 | +1.47% |
| monoGRU | within_subject | 0.79932 | 0.80598 | +0.00666 | +0.83% |
| monoGRU-DB | across_subject | 0.47631 | 0.48267 | +0.00636 | +1.34% |
| monoGRU-DB | within_subject | 0.87411 | 0.88445 | +0.01034 | +1.18% |
