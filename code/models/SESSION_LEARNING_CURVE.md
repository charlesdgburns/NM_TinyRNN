This existing prototype now lives in `code/analyses/session_learning_curve.py`.
The old models import/CLI and shell launcher remain compatible. It is separate
from the deferred `data_scaling` study and does not settle that study's design.

Run from NM_TinyRNN (or invoke the launcher by its absolute path):

    ./run_session_learning_curve.sh

Defaults: WS20 only; latest 2, 4, 6, 8, 10 sessions sorted by parsed datetime;
sequence length 32; batch size 8; two-unit GRU/monoGRU, tanh/relu, decoder bias
on/off; 10 outer x 9 inner folds; 150 vectorized seed/penalty configurations.
12 CPU processes run independent fits. The local loops use Trainer.fit and the
same save_data.save_inner_fold_results called by submit_jobs -> nested_cv.
There are no sbatch/srun calls. Explicit --subjects allows additional mice.

Output layout:

    data/rnns/nested_session_learning_curve/WS20/
      run.json
      sessions_02/monoGRU/sparsity/outer_fold_1/inner_fold_0/
        2_unit_monoGRU_tanh_unipolar_forced_model.pickle
        2_unit_monoGRU_tanh_unipolar_forced_info.json
        2_unit_monoGRU_tanh_unipolar_forced_trials_data.htsv
        2_unit_monoGRU_tanh_unipolar_forced_complete.json

Model identifiers and inner fold numbering (0..8) match the existing saver.
The constraint folder follows submit_jobs: relu -> energy, tanh -> sparsity.
As in the supplied learning-curve script, BOTH sparsity and energy grids are
searched for every model; folder names do not change the hyperparameter grid.
Trial tables cover only the selected sessions, include original behavioural
columns, hidden/gate activations, logits, probabilities and train/val/eval/unused
labels. Session remainders are labelled unused. The saver retains its existing
continuous trial-by-trial forward-pass convention across selected sessions.
Trainer does not expose epoch histories, so no training_losses.htsv is fabricated.
Metadata includes the five winning hyperparameters, searched ranges, and losses.

Extra checkpoint files live in .checkpoints/<model_id>/ under each architecture/
constraint directory. Completed outer folds also have <model_id>_outer_result.json.
fold_results.csv is updated as outer folds finish. summary.csv and plots are
written when the mouse finishes. Re-run the same command to resume; code/data/
settings mismatches are rejected by run.json. Do not change code during a run.

The original learning-curve CV is retained: shuffled complete 33-trial source
blocks yield 32 input/target pairs; sessions can occur in multiple splits. This
is intentionally different from nested_cv's default session-level partitions,
which cannot support 10-fold CV with only 2 sessions.

Check first:

    ./run_session_learning_curve.sh --dry-run
    ./run_session_learning_curve.sh --smoke-test --sessions 2 --output /tmp/tiny-rnn-smoke

To import completed checkpoints from the earlier runner, add
--resume-from /path/to/old/WS20. The runner verifies data, partitions, settings,
versions and training implementation before exporting cached winners with the
native saver. Only the intentional sequence/batch default edits are allowed.
For later resumes, --resume-from is optional once checkpoints have been imported.

Active unattended run: data/rnns/nested_session_learning_curve/ws20.log
Process group ID: data/rnns/nested_session_learning_curve/ws20.pid
Stop: kill -- -$(cat data/rnns/nested_session_learning_curve/ws20.pid)
