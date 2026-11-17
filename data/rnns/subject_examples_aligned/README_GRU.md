# Action-aligned GRU subject examples

110 two-unit tanh GRUs from `nested_DA_S32_B8`: WS01, WS02, WS05, WS08, WS09,
WS10, WS13, WS14, WS16, WS20 and WS22, each retaining its ten published outer
winners and exact stored inner indices. Selection remains the source minimum
block-reset train+validation NLL, with no test-based reselection or retraining.

Every exported model satisfies the requested convention:

- h1 has positive `decoder.weight[0,0] - decoder.weight[1,0]`: increasing h1
  favours A1, left, choice code 0.
- h2 has negative `decoder.weight[0,1] - decoder.weight[1,1]`: increasing h2
  favours A2, right, choice code 1.

This is decoder-contribution labelling, not a claim about empirical tuning or
positive firing rates. Tanh hidden states can be negative. This requested order
is the reverse of the current monoGRU export's documented unit order.

## Transformation and reference

Tanh allows signed permutations, not positive gain rescaling. The native
`mechanistic_variability.align_rnns` engine searches the two signed permutations
that satisfy the required action labels. All gate parameters, candidate
parameters, learned initial states and decoder columns transform consistently;
gate activations are permuted without sign flips, and decoder bias is unchanged.

The actual population medoid is **WS10, outer fold 8, stored inner index 4**.
It minimizes mean Euclidean distance over candidate input weights, candidate
bias, recurrent weights and decoder weights, after minimizing the permitted
alignment for every cohort fit. As in monoGRU, subjects have equal total weight
and their ten folds have equal within-subject weight. Gates, initialization and
decoder bias transform but are not included in that distance objective.

There are two globally equivalent action-labelled orientations, related by
negation and swapping of both units. They give identical medoid objectives.
The frozen reference uses the valid orientation with fewer source-unit sign
flips; native permutation order breaks an exact tie. This fixes a coordinate
convention, not a biological direction for hidden-state sums. Per-model matching
then minimizes distance to this frozen reference over both valid assignments.
All 110 fits were eligible reference candidates; no fits were excluded.

## Files

```text
<subject>/GRU/outer_fold_<N>/
  2_unit_GRU_tanh_unipolar_forced_info.json
  2_unit_GRU_tanh_unipolar_forced_model.pickle
  2_unit_GRU_tanh_unipolar_forced_trials_data.htsv
GRU_outer_fold_winners.htsv
GRU_regeneration_checks.htsv
GRU_export_manifest.json
GRU_population_medoid_model.pickle
GRU_average_aligned_model.pickle
GRU_average_parameters.htsv
GRU_average_parameters.md
```

GRU summary names are architecture-specific so these additions can coexist with
the existing monoGRU release files. Paths in the GRU winner table are relative
to the repository root. The original published training paths and fit metadata
are retained for provenance.

Trial tables are regenerated from the exact 204 raw session files, using
monoGRU's `load_subject_data` and `regenerate_trials` functions and the native
`mechanistic_variability._run_model_on_trials`. Histories are continuous over
the whole subject, including session and split boundaries. Each row is a trial
within one exact fit, with readouts after the current input predicting the next
choice. No lag shifts, recurrent resets, trial exclusions or fold averaging are
introduced. There are 84,861 distinct subject trials, represented separately in
ten fitted-model trajectories, for 848,610 saved trial rows.

All source trial identities, raw behaviour and original split labels match.
Hidden states equal `source_hidden[:, perm] * signs`; both gate vectors equal
their source vectors permuted by `perm`. Probabilities and logits are preserved
within float32 numerical error. All 110 saved models and trial tables were
reloaded and checked. Maximum source-versus-regenerated logit error is
2.03461e-6; probability error is 4.26806e-7. The ledger records every fit's
actual errors and row counts; the manifest records file, code and source hashes.
Original block-reset training metrics remain in info JSON files and do not
describe these continuous rollouts.

The 44-entry parameter summary first averages ten fits within each subject,
then averages the eleven subjects equally. Subject SD/SEM describe those eleven
subject means. Folds are dependent; no independent-fold inference is performed.
The parameter-mean pickle is descriptive, not a fitted model or an ensemble.

## Reproduction and scope

Use the code additions documented in `code/analyses/GRU_ALIGNMENT.md`, together
with the pinned published source release and raw-session provenance. No source
checkpoint or existing monoGRU artifact is overwritten by this addition.
The reference is descriptive and estimated from this whole cohort.

Direct model/trial loading uses the native pickle and tab-separated formats.
The separate modelling-reversal `rnn_data.aligned_subject_examples` importer is
currently monoGRU-specific and has not been extended by this export task.
