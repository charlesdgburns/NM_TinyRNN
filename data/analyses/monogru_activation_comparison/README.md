# Relabeling versus medoid alignment: hidden activations

110 selected two-unit monoGRU +BC-DB ReLU models from nested_DA_S32_B8: 11 subjects × ten outer folds. Both conditions were freshly replayed on identical raw subject histories.

Condition 1 applies only the h1 → A1, h2 → A2 permutation using mechanistic_variability.align_rnns with scales fixed at one. It does not independently balance weights. Condition 2 reproduces the maintained minimum-norm initialization followed by positive-scale Euclidean/Frobenius core-parameter fitting to the fixed balanced medoid (WS14, outer 3), preserving the action labels. Every refit was checked against its previously published aligned model.

Primary summaries include all trials (forced and free, all original splits). State runs continuously across the complete subject history. train/val/eval summaries apply masks afterward and are provided as sensitivities, not reset-based hidden-state evaluations.

| Component | Mean, relabeled | Mean, aligned | Mean change | Pointwise fold SD, relabeled | Pointwise fold SD, aligned | SD change |
|---|---:|---:|---:|---:|---:|---:|
| h1 | 0.3498 | 0.3833 | +9.6% | 0.0883 | 0.0691 | -21.7% |
| h2 | 0.2942 | 0.3372 | +14.6% | 0.0919 | 0.0694 | -24.5% |
| sum | 0.6440 | 0.7205 | +11.9% | 0.1692 | 0.1261 | -25.5% |

Means first average trials/folds within each subject, then subjects equally. Pointwise fold SD means: at each common input trial, take the sample SD across ten outer-fold responses, then average over trials and subjects. This quantifies across-fit amplitude variability; it is not uncertainty of the group mean. Folds share training data and no independent-fold significance test is used.

Summed activity (h1+h2 at each trial) has a median between-condition trajectory correlation of 0.998546, with range [0.938584, 0.999999]. The smallest correlation occurs for WS02, outer fold 1; this is an exploratory extreme, not the plotted example. Summed pointwise fold SD decreases in 10/11 subjects.

Permutation alone preserves the original sum (maximum numerical discrepancy 9.19e-07). For every model, h_aligned,j(t) = d_j h_relabelled,j(t), checked by direct replay. Thus individual-unit Pearson correlations are one up to numerical precision, while the unweighted sum becomes d1*h1+d2*h2 and can change its time course when d1 differs from d2.

Behavior is preserved: maximum logit difference 1.91e-06, update-gate difference 7.15e-07, and outer continuous NLL difference 2.35e-08 nats/free choice.

The sum is dependent on the hidden-coordinate scale convention. A change in it under this exact symmetry is not a change in predictive computation, and the parameter-distance objective does not guarantee reduced activation variability for every subject.

## Files

- fit_metrics.htsv: matched all/train/val/eval unit and sum metrics for every model.
- fit_checks.htsv: permutations, scale gains, prediction/gate preservation and NLL checks.
- subject_summary.htsv: equal-subject aggregation and pointwise across-fold SD.
- subject_fit_metrics.htsv: subject averages of fit metrics.
- trajectories/<subject>.npz: both freshly recomputed hidden trajectories, outer-fold IDs, original split masks and shared trial identifiers. Array shape is (outer fold, trial, hidden unit).
- manifest.json: source model, raw input and reference hashes.
- Figures: /home/charlesdgburns/Coding/NM_TinyRNN/results/appendix/monogru_activation_comparison

The time-series example is the predefined notebook example WS20, outer fold 10; the last session is cropped to its first 200 trials only for display, after full-history inference. The across-fold display includes all ten fits on those same inputs. All data and inference checks use complete histories.

Reproduce: `.venv/bin/python code/analyses/monogru_activation_comparison.py --jobs -1`.
