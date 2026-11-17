# monoGRU +BC-DB alignment

This is the corrected no-decoder-bias cohort: 110 selected two-unit monoGRU ReLU models from nested_DA_S32_B8 (11 subjects × ten outer folds). All model objects explicitly have decoder_bias=False and decoder.bias=None. The parent-directory mixed-bias reference from the initial run is superseded for this task.

Independent minimum-full-parameter-norm balancing and a hard decoder-contrast label convention were applied using the existing canonicalizer and mechanistic_variability alignment engine. All 110 models have finite balanced scales, opposing readout contrasts and two active units on synthetic probes. h1 supports choice 1; h2 supports choice 0. No models required a degenerate bounded fallback.

The equal-subject-weighted medoid is WS14, outer_fold_3, inner_fold_4 (zero-based inner index), decoder bias off. Leave-one-subject-out selects the same model in 8/11 cases, WS14 outer 6 in one case, and WS22 outer 8 in two cases. The representative model is therefore not uniquely stable to every subject deletion. This does not change any model's independently fixed scale or unit labels.

Across all original/aligned comparisons on the deterministic synthetic probe set, maximum absolute logit difference is 1.43e-6 and maximum probability difference is 2.98e-7. Gates and the expected hidden-state scale/permutation relationship passed the same checks. This is function-preserving coordinate alignment; scaling-dependent hidden statistics and regularization penalties can change.

Use reference_model.pickle and aligned_models.htsv in this directory for later analyses. Source checkpoints and notebook cells are untouched. Methods and notebook reuse instructions are in code/analyses/MONOGRU_ALIGNMENT.md. Six new alignment tests and four existing refinement tests pass.
