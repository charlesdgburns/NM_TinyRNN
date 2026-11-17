"""Submit two-unit standard GRU and monoGRU jobs for the serotonin cohort.

Run on the cluster in the training environment; --dry-run only lists jobs.
Uses the existing SLURM resources, conda environment and nested-CV defaults.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO.parent))
from NM_TinyRNN.code.models import experiments
from NM_TinyRNN.code.servers.SLURM import submit_jobs

SUBJECTS = [f'WS{n:02d}' for n in
            [3, 4, 6, 7, 11, 12, 15, 17, 18, 21, 23, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--subjects', nargs='+', choices=SUBJECTS, default=SUBJECTS,
                        help='Submit only these serotonin subjects (default: all)')
    parser.add_argument('--partition', help='Comma-separated SLURM partitions')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--sequence-length', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(argv)
    missing = [s for s in args.subjects if not any((REPO/'data/AB_behaviour'/s).glob('*/trials.htsv'))]
    if missing:
        parser.error(f'Missing serotonin data: {missing}')
    folder = experiments.training_folder_name(
        args.sequence_length, args.batch_size, prefix='nested_5HT')
    frame = experiments.get_job_info_df(
        processed_data_path=REPO/'data/AB_behaviour', save_path=REPO/'data/rnns',
        folder_name=folder, sequence_length=args.sequence_length,
        batch_size=args.batch_size)
    frame = frame[frame.subject_id.isin(args.subjects)].query(
        "hidden_size == 2 and input_encoding == 'unipolar' and input_forced_choice and "
        "((model_type == 'GRU' and nonlinearity == 'tanh' and decoder_bias) or "
        "(model_type == 'monoGRU' and nonlinearity == 'relu' and not decoder_bias))")
    # Each job trains all outer folds; resubmit if any fold is incomplete.
    os.chdir(REPO.parent)  # Existing job/log paths and package imports use this directory.
    count = 0
    for _, rows in frame.groupby(['subject_id', 'model_id'], sort=True):
        if rows.completed.all():
            continue
        job = next(rows.itertuples(index=False))
        print(f'{job.subject_id}: {job.model_id} -> {job.save_path}', flush=True)
        if not args.dry_run:
            script = submit_jobs.get_NM_TinyRNN_SLURM_script(
                job, partitions=args.partition, cpu_only=args.cpu_only)
            subprocess.run(['sbatch', str(script)], check=True)
        count += 1
    print(f'{count} jobs {"planned" if args.dry_run else "submitted"}; results: {folder}')


if __name__ == '__main__':
    main()
