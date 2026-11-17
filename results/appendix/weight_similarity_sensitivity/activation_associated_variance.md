# Activation-associated variance in aligned weight similarity

Primary scores: continuously refined ReLU alignment and permutation/sign-only tanh alignment. No balancing. The discrete-ReLU version is saved as a sensitivity comparison.

## Estimand and model

Observations are cosine scores averaged over fold pairs, for each architecture × decoder-bias × activation × subject or subject-pair block. Six architecture/bias configurations are matched across activations. Within-subject analysis has 132 observations (11 subjects × 6 configurations × 2 activations); across-subject analysis has 660 (55 dyads × 6 × 2). These are not 132 or 660 independent subjects.
The reduced fixed-effect model includes configuration and subject/dyad intercepts. The next model adds one overall activation contrast. The full model allows a separate activation contrast in each architecture/bias configuration. “Overall activation” is the reduction in residual sum of squares after adding the single activation contrast, divided by the TOTAL observed sum of squares. “Additional configuration-specific” is the further reduction when allowing those six contrasts to differ. Their sum is the combined activation-associated share of total variance. Full data are balanced, so the activation terms are orthogonal to configuration and block intercepts.
The separate partial-R² column in the data file divides the combined reduction by the reduced model’s residual variance instead of total variance. It therefore answers a different question and is larger. These descriptive sample R² increments are not causal effects, bias corrections, measurement-equivalence tests or estimates of population variance components. Adding interactions can fit sample noise; the numbers should not be read as significance tests.

| Scope | Overall activation (% total variance) | Additional configuration-specific (% total) | Combined (% total) | Combined leave-one-subject-out range |
|---|---:|---:|---:|---|
| across_subject | 0.506% | 10.776% | 11.282% | 7.322–15.848% |
| within_subject | 1.568% | 1.910% | 3.478% | 2.597–6.669% |

## Dependence and stability

No inferential p-values or bootstrap confidence intervals are calculated. Instead, remove each of the 11 subjects in turn and recompute the decomposition. For across-subject analysis, deletion removes every dyad involving that subject. The displayed range is a stability check, NOT a confidence interval; it does not assume dyads or folds are independent.

## Interpretation

A small overall activation term can coexist with stronger activation-by-configuration differences of opposing signs. Neither a small nor a large association establishes score comparability: observed differences may reflect fitted parameter patterns, training/selection differences or properties of the measurement procedure. Conversely, absence of a large association is not evidence of equivalence. The existing function-preserving reference-rescaling sensitivity remains a separate diagnostic.

Files: `activation_associated_variance.htsv` contains both ReLU alignment versions and both denominators; `activation_variance_subject_deletion.htsv` contains all deletion results.

```bash
.venv/bin/python code/analyses/weight_similarity_sensitivity.py --stage variance
```
