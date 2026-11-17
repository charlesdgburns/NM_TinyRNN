# Weight similarity sensitivity to pairwise scaling

All comparisons use selected raw fitted weights. No minimum-norm balancing or subject bootstrap is applied.

| Comparison | Within subjects | Across subjects |
|---|---:|---:|
| ReLU: discrete → continuous scaling | -0.25–+0.22% | -0.12–+0.65% |
| tanh: unscaled → hypothetical continuous scaling | -0.21–+1.02% | -0.83–+2.22% |

Ranges describe percentage changes in variant means, not uncertainty intervals. Each variant mean equally averages subjects or unordered subject dyads. Percent change = 100 × (scaled mean − baseline mean) / |baseline mean|. Individual near-zero baseline scores make percentage changes unstable; figures show absolute cosine differences.

ReLU: exhaustive permutation/grid search using {0.5, 0.875, 1, 1.25, 1.625, 2}, then TRF least-squares refinement from the winning scales with its permutation fixed. Tanh: exhaustive permutation/sign baseline; the hypothetical scaled condition refines positive scales in every sign/permutation branch, starting at unit scales. Tanh scaling changes function. Both use the core Euclidean alignment objective; cosine includes all trainable parameters, so scaling can lower the reported cosine.

Both continuous procedures use analytical Jacobians, log-scale bounds [-20,20], ftol/xtol/gtol 1e-10 and at most 300 residual evaluations. A successful ReLU refinement is retained only if it improves the grid objective. The hypothetical tanh helper retains the baseline if a candidate does not improve the distance. Neither local search proves a global continuous optimum.

The combined discrete figure reproduces the previous standard-method contrast. The continuous figure instead uses grid-refined ReLU scores in its lower panels. Both keep tanh unscaled in the activation contrast; tanh scaling appears only in the sensitivity panels.

Selected fits: 1,320; fit pairs: 71,940; failed optimisation branches: 103; selected scales at numerical bounds: 0.

## Per-variant changes

| Variant | Scope | Baseline | With continuous scaling | Δ cosine | % change |
|---|---|---:|---:|---:|---:|
| GRU+BC | across_subject | 0.38199 | 0.38305 | +0.00106 | +0.28% |
| GRU+BC | within_subject | 0.76106 | 0.76218 | +0.00113 | +0.15% |
| GRU+BC-DB | across_subject | 0.42054 | 0.42003 | -0.00051 | -0.12% |
| GRU+BC-DB | within_subject | 0.66531 | 0.66436 | -0.00095 | -0.14% |
| GRU | across_subject | 0.45259 | 0.45264 | +0.00005 | +0.01% |
| GRU | within_subject | 0.74686 | 0.74838 | +0.00152 | +0.20% |
| GRU-DB | across_subject | 0.44661 | 0.44292 | -0.00369 | -0.83% |
| GRU-DB | within_subject | 0.74822 | 0.74661 | -0.00161 | -0.21% |
| lightGRU+BC | across_subject | 0.41755 | 0.41911 | +0.00157 | +0.38% |
| lightGRU+BC | within_subject | 0.73071 | 0.72904 | -0.00167 | -0.23% |
| lightGRU+BC-DB | across_subject | 0.51945 | 0.52006 | +0.00060 | +0.12% |
| lightGRU+BC-DB | within_subject | 0.78497 | 0.78302 | -0.00195 | -0.25% |
| lightGRU | across_subject | 0.57077 | 0.57008 | -0.00069 | -0.12% |
| lightGRU | within_subject | 0.84671 | 0.85175 | +0.00505 | +0.60% |
| lightGRU-DB | across_subject | 0.51097 | 0.50963 | -0.00134 | -0.26% |
| lightGRU-DB | within_subject | 0.81668 | 0.81504 | -0.00164 | -0.20% |
| monoGRU+BC | across_subject | 0.59504 | 0.59892 | +0.00388 | +0.65% |
| monoGRU+BC | within_subject | 0.82796 | 0.82976 | +0.00180 | +0.22% |
| monoGRU+BC-DB | across_subject | 0.78034 | 0.78258 | +0.00224 | +0.29% |
| monoGRU+BC-DB | within_subject | 0.90447 | 0.90573 | +0.00126 | +0.14% |
| monoGRU | across_subject | 0.49226 | 0.49982 | +0.00756 | +1.54% |
| monoGRU | within_subject | 0.86034 | 0.86910 | +0.00876 | +1.02% |
| monoGRU-DB | across_subject | 0.44877 | 0.45872 | +0.00995 | +2.22% |
| monoGRU-DB | within_subject | 0.90570 | 0.91274 | +0.00704 | +0.78% |
