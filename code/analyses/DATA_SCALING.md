# Recency-based dataset scaling

`data_scaling.py` is a standalone local runner and implements the shared study
interface for later AWS execution. No scaling jobs are launched on AWS by running
it locally. Models:

- Two-unit GRU, tanh, decoder bias on.
- Two-unit monoGRU, ReLU, decoder bias off.

## Protocol

1. Reserve each subject's newest timestamp-named session as a fixed holdout.
2. Find the smallest pool of recent sessions that supports the largest training
   budget in **all 90** nested-CV splits after excluding that holdout. Subject
   pools may differ. Freeze this pool across checkpoints. There must be at least
   ten other sessions for ten nonempty session chunks.
3. Split those other sessions using the existing deterministic sorted-session
   ten-chunk policy. For each outer fold, rotate the nine inner validation chunks.
   Session proportions approximate 80/10/10; trial proportions need not match.
4. At each checkpoint, take the newest eligible **training blocks**, adding older
   blocks as needed. Budgets are 512, 1024, 1536, 2048 prediction targets, including
   forced-choice targets. They are exact multiples of sequence length 32. The
   oldest contributing session can be partial; blocks never cross sessions.
5. Validation, outer evaluation and fixed holdout blocks remain identical across
   checkpoints. Smaller training sets are nested subsets of larger ones within
   each outer/inner split. Training-session membership can differ across folds.
6. Fit from scratch using the existing trainer/regularization and weight-seed
   search, with patience 100 by default. Preserve single-best-inner selection by validation NLL, without refit.
   Optimizer/block-shuffle seeds are paired across budgets. State resets follow
   the existing block policy. Validation/NLL mask forced-choice targets.
7. Evaluate the same selected winner on outer evaluation and the fixed newest
   session. Holdout scores never select budgets, configurations, seeds or winners.

A 32-target block consumes 33 raw trials. Budgets count predicted targets, not
source rows or free-choice targets. Minibatch size does not drop the final batch.
This protocol measures adding progressively older data, rather than a random
sample-size learning curve; recency and session differences are part of the effect.

With the current data and default budgets, WS05 requires a 14-session pool;
all other DA subjects require 11, including the fixed newest holdout. These
counts are calculated from the data rather than hard-coded.

## Local usage

From the repository root, a diagnostic WS20 run:

```bash
.venv/bin/python code/analyses/data_scaling.py \
  --subjects WS20 --smoke \
  --output code/analyses/.local/ws20-scaling-smoke
```

Smoke mode executes both models and all four budgets, two outer folds and all
nine inner folds: 144 fits with two epochs, two initializations and a reduced
regularization grid. Its NLLs are pipeline diagnostics, not scientific estimates.
The config records `diagnostic_only: true`. Choose a new output directory per run.

For the actual complete WS20 run, omit `--smoke` (720 full-grid fits):

```bash
.venv/bin/python code/analyses/data_scaling.py \
  --subjects WS20 --output code/analyses/.local/ws20-scaling-full
```

Both model types use shared learnable `hidden_0` vectors initialized at zero.
Every block resets to the current learned vector; final activations do not carry
between blocks. Global and loader RNGs follow the inner-fold-derived seed, paired
across budgets. `--early-stop`, `--seed` and optional `--shuffle-seed` expose those
settings in the standalone CLI; defaults match the shared study/AWS path.

Use `--plan-only` to inspect session selection and split/block indices without
fitting. `--outer-folds 1 2` explicitly restricts execution; pool feasibility still
checks every split. Defaults for block length and minibatch size come from
`datasets.SEQUENCE_LENGTH` and `training.BATCH_SIZE`. Explicit budgets must be
multiples of the chosen block length. `analyses/run.py` also supports this study,
including individual tasks and its existing input-identity checks.

The standalone runner writes source/data hashes, frozen config and plan, native
inner-fold artifacts and checkpoints, task metadata, `reduced/fold_results.csv`,
`reduced/summary.csv`, and portable experiment/analysis tables. Summaries report
mean outer-fold NLL and mean fixed-holdout NLL separately. Checkpoint selection
always uses validation NLL. These repeated fixed-holdout scores are not separate
independent test sessions. Failed standalone runs retain diagnostics; use a new
output directory or the shared task CLI for task-oriented execution.

## Similarity after collection

```bash
.venv/bin/python code/measures/data_scaling_similarity.py \
  --output code/analyses/.local/ws20-scaling-smoke
```

Uses the same validation-selected winners, comparing models within each
architecture and checkpoint across outer folds and subjects. Probe inputs are
fixed newest-session holdout blocks, balanced to the same number of newest blocks
per subject and reused across all checkpoints. Existing parameter alignment is
checked for logit invariance. Outputs report parameter cosine (including biases
and learned initial states), per-unit hidden correlation, and logit-difference
correlation; undefined correlations remain missing. Subject/subject-pair means
precede across-subject summaries. WS20 alone tests within-subject comparisons;
across-subject comparisons require multiple subjects. Pairwise fold scores are
descriptive, not independent replicates or confidence intervals.

Similarity runs locally after collection because the existing alignment module
has analysis/plotting dependencies not needed by training workers. It is not yet
part of the AWS reducer. Its own implementation/source provenance is recorded
separately from the frozen training inputs.
