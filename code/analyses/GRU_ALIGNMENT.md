# GRU action-aligned export

This addition ports the current monoGRU alignment/export workflow to the
published two-unit tanh GRUs, using the user-approved hard action labels
h1 -> A1 and h2 -> A2 in every fitted model. See
`data/rnns/subject_examples_aligned/README_GRU.md` for scientific scope,
actual source/cohort/fit selection, timing, row units, aggregation and validation.

Baseline code is pinned to `f36568933d5cd937e3944092c63998e46a458819`, whose
parent is the requested `d5e02f6390e911f0eef6cc830f0dc2b2454e7af7` export
commit. The constrained alignment function is provided in the new
`code/measures/gru_alignment_constraints.py` module. It extracts the current
native alignment function with an optional joint signed-permutation restriction
and reuses the unchanged native gate-transformation helper. Existing
`mechanistic_variability.py`, all other existing code, and all monoGRU files
remain unchanged. The original monoGRU alignment/export tests pass.

The GRU script reuses monoGRU's synthetic probes, equal-subject weights, generic
parameter-transform/core helpers, raw-session loading, trial regeneration and
parameter averaging. The architecture-specific differences are tanh signed
permutations, vector update/reset gates, retained decoder bias and hard A1/A2
decoder signs. Tanh norms are already invariant to its permitted transformations,
so no ReLU gain-balancing optimization is applied.

## Reproduce from the repository root

```bash
python -m unittest discover -s code/tests -v
python code/analyses/GRU_alignment.py
python code/analyses/export_aligned_GRU.py --jobs 4
python code/tests/verify_gru_release.py
```

Required sources are the pinned current `data/rnns/subject_examples_latest`
release and all raw sessions listed in the frozen published monoGRU export
manifest. A copy of that upstream provenance manifest is supplied as
`data/rnns/subject_examples_aligned_monoGRU_provenance.json`. The scripts check
source artifact and raw-session hashes; another training run cannot be
substituted because it shares fold labels or filenames.

The alignment output defaults to
`data/analyses/GRU_alignment/nested_DA_S32_B8/GRU_tanh/`. It contains the
frozen reference, aligned copies, exact transformations, core vectors, medoid
distance matrix, leave-one-subject-out sensitivity and source/code/artifact
hashes. Its model copies and corresponding subject-example pickles have
identical parameters. Exact native source inner indices are retained, without
renumbering them.

Completed outputs are immutable. The packaged release is already complete;
rerunning requires empty `--output` locations within this repository and the
corresponding `--alignment-dir`. An existing completed GRU release is rejected
instead of overwritten. Existing monoGRU files can coexist with the GRU addition.

For a new GRU aligned to the frozen reference:

```python
from NM_TinyRNN.code.analyses.GRU_alignment import load_model, align_model

reference = load_model("data/rnns/subject_examples_aligned/GRU_population_medoid_model.pickle")
aligned_model, audit = align_model(model, reference)
```

Inspect the returned `perm` and `transform_vec` audit. Models with nonfinite
parameters or near-zero decoder contrasts stop rather than receiving arbitrary
action labels. New cohorts use the frozen reference unless a new reference is
explicitly intended; changing a coordinate convention changes its hidden summaries.

## Executed validation

The approved `cortical_photometry_1` interpreter passed 14 focused tests from the
upstream monoGRU and new GRU test modules, plus two independent synthetic tests
covering all eight tanh signed permutations. Native source replay passed for
all 110 original models before alignment. All 110 aligned full-history exports
then passed prediction, signed-state, both-gate, metadata, identity, split and
serialization checks. An independent serialized-release audit verified all
336 manifest-listed artifacts and all original source hashes.

The GRU alignment directory's `VALIDATION.md` and JSON ledgers retain exact
results and limitations. Network downloads needed DNS-interruption retries with identical
inputs. Imports emitted Matplotlib/Fontconfig cache warnings; no cache settings
were changed and no plotting validation is claimed. Black was unavailable;
the scientific checks above executed successfully without it. No figures,
notebooks or new model training were executed. The publication is additions-only;
pre-existing repository file paths and blob hashes are preserved.
