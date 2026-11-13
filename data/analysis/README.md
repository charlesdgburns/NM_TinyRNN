# Intermediate analysis outputs

Use one active folder per analysis. Dataset names belong in the analysis
configuration and cache contents, rather than in filenames.

- `performance/analysis_df.htsv`: saved inner-model paths.
- `performance/performance_df.htsv`: saved and trial-level performance metrics.
- `mechanistic_variability/selected_models.htsv`: selected outer-fold winners.
- `mechanistic_variability/pairwise_similarities.htsv`: their similarities;
  this is also the default notebook cache.
- `weight_similarity_sensitivity/`: generated scaling-comparison intermediates
  from `code/analyses/weight_similarity_sensitivity.py`.

The published caches use `nested_DA_last10_P100_S32_B8`; source model paths
are retained in the dataframes. Cache readers check source paths and fold
identities before reuse. The default similarity reader also checks the selected
models and expected pair counts. Explicit `cache_path` remains supported.

There is one active output set per analysis. Changing the input dataset or
selection regenerates the appropriate cache; report/plot-only weight-sensitivity
runs must match the saved run and hidden size. Preserve an old output locally
before switching datasets if it is needed for a historical comparison.

Historical runs, superseded similarity methods and exploratory outputs have
moved to gitignored `.local/`. They remain on the originating computer; old
tracked caches are also recoverable from Git history. Generated group caches
and diagnostic snapshots are not included in this focused change.

Use the existing notebook APIs to load the current cache locations:

```python
analysis_df = analysis.get_analysis_df(experiment_df)
performance_df = perf.get_performance_df(analysis_df)
best_model_df = perf.select_best_outer(performance_df)
sim_df = mech_var.compute_similarities(best_model_df, use_cache=True)
```

Fitted models remain in `data/rnns/`. Final manuscript figures, tables and
weight-sensitivity narrative reports go in `results/`.
