# Intermediate analysis outputs

One active directory per analysis. Dataset/run names belong in the analysis
script's configuration and saved metadata, not in directory or cache names.

| Directory | Contents | Maintained entry point |
|---|---|---|
| `performance/` | Shared inner-model path table and performance dataframe | `code/measures/analysis.py`, `code/measures/performance.py` (also used by notebooks) |
| `mechanistic_variability/` | Selected outer winners, reference inputs, similarities, intermediate statistics and provenance | `code/analyses/mechanistic_variability.py` |
| `weight_similarity_sensitivity/` | Selected models, scaling comparisons, intermediate summaries and provenance | `code/analyses/weight_similarity_sensitivity.py` |

The current analysis scripts default to `nested_DA_last10_P100_S32_B8` under
`data/rnns/`. `manifest.json`, `selected_models.htsv` and cache JSON sidecars
record the actual inputs. Model-path/fold checks reject cache reuse across
training runs. Post-training report/plot stages also reject a mismatched run
(and a mismatched hidden size for weight sensitivity).

Final figures, publication tables and narrative reports belong in `results/`;
fitted models remain in `data/rnns/`. Dataframes here are computational inputs
and intermediate summaries, rather than manuscript exports.

## Historical and exploratory outputs

`.local/` is gitignored. It retains older runs, superseded reference protocols,
selection diagnostics, alignment/solver benchmarks and exploratory balancing or
rescaling analyses. Their diagnostic scripts now point to these locations.
They are not current main-result caches. Existing historical manifests retain
their original computation hashes and descriptions.

The 2026-11-13 migration map is `.local/layout_migration/moves.json`;
`.local/layout_migration/deduplicated.json` records the only deleted data files:
two byte-identical copies, with retained paths, SHA-256 hashes and byte counts.
Scientific scores were not recomputed or changed by this reorganization.

## Cache use

Use the notebook APIs rather than constructing dataset-named filenames:

```python
analysis_df = analysis.get_analysis_df(experiment_df)
performance_df = perf.get_performance_df(analysis_df)
best_model_df = perf.select_best_outer(performance_df)
sim_df = mech_var.compute_similarities(best_model_df, use_cache=True)
```

Similarity notebooks and the maintained study share
`mechanistic_variability/pairwise_similarities.htsv`; there is no second full
notebook cache. Shared path/performance caches are under `performance/`.
There is one active output set per analysis; changing a run recomputes or
replaces that set. Preserve a completed output in `.local/` before switching
inputs if it is needed for a historical comparison. Plot-only/report-only
commands must match the saved manifest's run and hidden size.
