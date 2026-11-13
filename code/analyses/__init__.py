"""Named scientific workflows, independent of local, SLURM, or AWS execution."""
from importlib import import_module

STUDIES = ('baseline', 'pooled_ws', 'data_scaling')


def get_study(name):
    if name == 'pooled_ws_cv': # old workflow name
        name = 'pooled_ws'
    if name not in STUDIES:
        raise ValueError(f'Unknown study {name!r}; choose from {STUDIES}')
    return import_module(f'{__name__}.{name}')
