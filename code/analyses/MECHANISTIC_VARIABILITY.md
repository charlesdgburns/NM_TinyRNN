# Mechanistic variability: main paper and appendix

Two entry points own the reports. Appendix statistical computations live in the appendix script; reusable metrics live in `code/measures`;
plotting helpers live in `code/figures`. Both consume the same canonical frozen
similarities from `nested_DA_S32_B8` by default.

```bash
# Verify cached scores, update statistics, Table 1 and focused violins.
.venv/bin/python code/analyses/mechanistic_variability_main.py

# Appendix: variance breakdown and hierarchical intervals, all configurations,
# Cohen's d panels, and exploratory factorial analysis.
.venv/bin/python code/analyses/mechanistic_variability_appendix.py

# Rebuild just one appendix section.
.venv/bin/python code/analyses/mechanistic_variability_appendix.py --sections factorial

# Explicitly compute fresh similarities; model training is never rerun here.
.venv/bin/python code/analyses/mechanistic_variability_main.py --stage all --recompute
```

Main `--stage compute`, `statistics`, and `plots` isolate stages. Default
`report` reuses verified similarities. The appendix defaults to saved posterior
reuse (`--variance-stage plots`); `--variance-stage all` explicitly permits
new hierarchical fitting and can be expensive. Do not reuse posterior draws
when score files, selected models or the computation fingerprint change.
Metadata-only path/report changes are audited separately and preserve posterior
identities. The optional inclusion-evidence engine is invoked with
`code/analyses/mechanistic_variability_appendix.py --sections evidence`; ordinary report generation
does not launch nested sampling.

## Selection, alignment and trial sequences

Select one winner per outer fold by minimum free-choice training + validation
NLL, resetting to the learned initial state on each original sequence block.
No outer-test loss or regularization penalty enters selection. There are 11
subjects, 10 winners per subject and 12 two-unit configurations.

B is aligned to A's weights; both run on A's saved trial sequence, continuously,
including forced trials. Only the existing sorted comparison direction is used;
there is no reverse-direction averaging. Tanh uses permutation/sign alignment;
ReLU adds positive-scale continuous refinement from the grid winner. The
objective is Euclidean distance of the alignment's main parameters. Similarities
are parameter cosine, unitwise hidden-state Pearson means, and flattened gate
and logit-difference Pearson correlations.

## Output layout

- `data/analyses/mechanistic_variability/`: selected models, pairwise scores,
  subject-pair summaries, contrasts and canonical computation provenance.
- `.../main/`: focused intermediate subject summaries and methods.
- `.../appendix/variance/`: raw variance budgets, robust summaries, posterior
  draws, hierarchical summaries and diagnostics.
- `.../appendix/factorial/`: design, omnibus tests, planned contrasts, cell means
  and reconstructed attribution.
- `results/main/`: `table_1.csv`, `table_1.svg`, focused GRU/monoGRU violin SVGs.
- `results/appendix/mechanistic_variability/`: full-configuration violin and
  Cohen's d SVGs/tables, with `variance/` and `factorial/` subfolders.

All new publication figures are SVG. Historical/local experiments remain in
`.local` or gitignored `code/analyses/preliminary`. The folder migration from
`data/analysis` and `data/anlyses` is recorded under
`data/analyses/.local/layout_refactor_20261117`; historical manifests retain
original paths and hashes.

## Inference and interpretation

Within: 45 fold-pair scores become one mean per subject; exact two-sided sign
tests compare the 11 independent subject differences. Across: 100 scores become
one mean per dyad; 55 dyads share subjects, so delete each subject and its
incident dyads for a jackknife SE and approximate t(10) inference. Table 1 uses
BH across eight focused tests. Within stars describe direction/median changes;
across stars describe mean changes. Deletion ranges are sensitivity ranges,
not confidence intervals. Cohen's dz is supplementary/descriptive. All-model
Cohen panels use a separate BH family of 94 contrasts.

The raw variance budget is an exact descriptive decomposition of similarity
scores. Hierarchical Gaussian models operate on atanh(similarity); their
absolute variance units are squared atanh(similarity). Relative contributions
and absolute posterior intervals appear together; numerical tables are separate
CSV files. Inspect sampling diagnostics and posterior predictive checks: these
models are exploratory. Ratios of similarity-score variance do not establish
whether model parameters or mechanisms are more reproducible.

Factorial ANOVA uses architecture × BC × decoder-bias removal. Within-subject
omnibus tests use repeated-measures contrasts, Greenhouse–Geisser correction
and BH31; planned contrasts use BH94, with across-subject node-deletion
uncertainty. BC bundles activation and regularization changes. Reset removal
is lightGRU−GRU; scalar gating is monoGRU−lightGRU, conditional on reset removal.
Scalar gating with a reset gate was not observed, so an independent gate×gate
interaction is not estimable.

Attribution averages marginal changes over the 12 intervention orders in which
reset removal precedes scalar gating. It uses only the 12 observed cells; four
contributions sum to monoGRU+BC−DB minus standard GRU. It is a conditional,
exploratory allocation, not unconstrained four-player Shapley or a causal
analysis. Error bars use subject-deletion SE and t(10); contrast-point fills
use BH94 rather than uncorrected p-values.
