# Server execution

`SLURM/submit_jobs.py` writes batch scripts and submits them with `sbatch`.
Experiment definitions and shared data/result paths live in
[`models/experiments.py`](../models/experiments.py).

```python
from NM_TinyRNN.code.models import experiments
from NM_TinyRNN.code.servers.SLURM import submit_jobs

# Build/filter the same backbone used by downstream analysis:
experiment_df = experiments.get_DA_info_df(sequence_length=64, batch_size=16)

# Submit when ready:
submit_jobs.run_training(experiment_df)
```

A supplied manifest is not rebuilt or truncated by test mode. Existing preset
calls such as `run_training(test=False, sequence_length=64, batch_size=16)` still
work. Builder functions remain re-exported for compatibility.

Generated scripts and logs still go to `Jobs/NM_TinyRNN/{slurm,out,err}`;
training results still use the manifest's `save_path`. Resource settings belong
to the SLURM script generator. No jobs are submitted by importing either module.

See [training configuration](../models/TRAINING.md) for defaults, folder naming,
and legacy result lookups. `NM_TinyRNN.code.models.submit_jobs` remains an alias
for external callers using the old import.
