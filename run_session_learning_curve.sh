#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$repo_dir/.venv/bin/python" -u "$repo_dir/code/analyses/session_learning_curve.py" --jobs 12 --threads 1 "$@"
