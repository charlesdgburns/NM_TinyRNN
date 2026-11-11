"""Compatibility import; submission code lives in code.servers.SLURM.submit_jobs.

Use ``from NM_TinyRNN.code.servers.SLURM import submit_jobs`` in new code.
Both paths resolve to the same module so settings and reloads stay consistent.
"""
import sys

from NM_TinyRNN.code.servers.SLURM import submit_jobs

sys.modules[__name__] = submit_jobs
