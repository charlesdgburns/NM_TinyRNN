# Server execution

`SLURM/submit_jobs.py` builds training job tables, writes batch scripts, and
submits them with `sbatch`.

```python
from NM_TinyRNN.code.servers.SLURM import submit_jobs

# Inspect jobs without submitting:
jobs = submit_jobs.get_DA_info_df(sequence_length=64, batch_size=16)

# Submit when ready:
submit_jobs.run_training(test=False, sequence_length=64, batch_size=16)
```

Generated scripts and logs still go to `Jobs/NM_TinyRNN/{slurm,out,err}`;
training results still use the configured `save_path`. This refactor changes
the source location, not job configuration, execution, or output locations.

See [training configuration](../models/TRAINING.md) for defaults, folder naming,
and legacy result lookups. `NM_TinyRNN.code.models.submit_jobs` remains an alias
for compatibility with external scripts and notebooks using the old import.
