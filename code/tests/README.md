# Sanity and regression checks

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s code/tests -p 'test_*.py'
```

The training checks use small synthetic datasets and do not submit jobs, write
training artifacts, or run a full nested-CV sweep. They use CUDA when available.
Audit findings and explanations are recorded separately in `code/audit`.
